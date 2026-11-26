//! A draft head of the model itself on the executor (`kinds::MtpHead`, `lower::draft_head`: Qwen3.8-Flash-Next's MTP layer of the model's streams, GLM-5.3's
//! of DeepSeek-V3's joined form): a draft part whose input is the model's state of a row and the token after it (`Op::MtpIn`), its blocks the model's kinds,
//! its keys and values in rings after the model's, the model's own LM head. A head's row at position p reads the model's state at p (its last streams, or
//! its final normed hidden state: `state`) and the token at p + 1, and drafts the token at p + 2; its own state after its blocks is the next draft's input.
//!
//! The head runs a row behind the model: after each prompt chunk, on the chunk's rows whose next token is known (the row the last chunk left waiting, then this
//! chunk's but its last, which waits in the head's Hid for the token the model picks); a drafting round (`spec_round_head`) gives the waiting row that token
//! (or takes the draft the head made after the last round), drafts on its own streams, has the model check the drafts in one verify pass (each row as its
//! step: `Dyn::verify_rows`, the states gone back to after the rows kept: `Dyn::rewind`), and runs the head again on the kept rows' streams, its rows at those
//! positions in place of its drafts' (what its caches held past them is written again before a row reads it).
//!
//! A model split across GPUs (src/dynamic/split.rs) drafts with the head on its last stage, which holds the model's last state, its LM head and the embedding
//! the head reads: the same steps, a round's in its parts (`stage_chunk`, `stage_drafts`, every stage's verify pass and rewind, `stage_head_again`).

use super::{copy_d2d_async, copy_d2h_async, Draft, Dyn, HeadLast, IdxKeys, IdxPick, ModelDesc, MtpFront, Op, Slot, VERIFY_ROWS};
use crate::ffi::{err, Error, Result};

/// One pass of the head as `Dyn::trace_head` keeps it: the position of its first row, each row's token (the one after the model's row it continues), the
/// state each row read (bf16, [rows, the state's width]: `state`), its experts' choices ([rows, k], the router's order; empty where the block leaves none),
/// its indexers' picks (`mla::PickTrace`, a latent attention's with an indexer), the last row's logits (bf16 words) and its draft.
#[derive(Clone, Debug)]
pub struct HeadPass {
    pub pos: usize,
    pub tokens: Vec<u32>,
    pub streams: Vec<u16>,
    pub routes: Vec<i64>,
    pub k: usize,
    pub picks: Vec<super::PickTrace>,
    pub logits: Vec<u16>,
    pub draft: u32,
}

/// A block draft head's pass as `Dyn::trace_block` keeps it (`MtpFront::Block`, DSpark's): its context of rows the model kept (their first position, tokens and
/// taps, bf16 [rows, the taps' width]), or a block (the anchor's position and token, its drafts, each draft's logits with the Markov head's (bf16 words), and
/// each of its blocks of experts' choices of the block's rows: (its layer, [rows, k])).
#[derive(Clone, Debug)]
pub enum BlockTrace {
    Context { pos: usize, tokens: Vec<u32>, taps: Vec<u16> },
    Block { pos: usize, anchor: u32, drafts: Vec<u32>, logits: Vec<Vec<u16>>, routes: Vec<(usize, Vec<i64>)> },
}

/// A block draft head's front (`MtpFront::Block`): its block, noise token, Markov matrices and rank; None for another head.
fn block_of(h: &ModelDesc) -> Option<(usize, u32, (usize, usize), usize)> {
    match h.first {
        Op::MtpIn { front: MtpFront::Block { block, noise, markov, rank }, .. } => Some((block, noise, markov, rank)),
        _ => None,
    }
}

/// The slot of the state a draft head's row continues, the model's and the head's own alike: the residual streams (a head of the model's streams), or the
/// final normed hidden state (a joined head: its final norm's output, which the LM head reads).
pub(super) fn state(h: &ModelDesc) -> Slot {
    match h.first {
        Op::MtpIn { front: MtpFront::Joined { .. }, .. } => Slot::H,
        _ => Slot::S,
    }
}

/// Whether a draft head's description `h` runs beside the model's `big`: the model's width, vocabulary and streams, its rotary tables the model's (it names
/// them by index), no per-sequence state (its caches are rings), every ring an operation of it names one it holds.
pub(super) fn fits(big: &ModelDesc, h: &ModelDesc) -> Result<()> {
    if (h.hidden, h.vocab, h.streams()) != (big.hidden, big.vocab, big.streams()) {
        return err(format!(
            "{}: a width of {}, a vocabulary of {} and {} streams beside a model of {}, {} and {}",
            h.family,
            h.hidden,
            h.vocab,
            h.streams(),
            big.hidden,
            big.vocab,
            big.streams()
        ));
    }
    if !h.states.is_empty() || h.ropes.len() > big.ropes.len() || h.ropes.iter().zip(&big.ropes).any(|(a, b)| format!("{a:?}") != format!("{b:?}")) {
        return err(format!("{}: per-sequence states, or rotary tables that are not the model's", h.family));
    }
    Ok(())
}

/// The rings `d`'s operations name moved `by` places (a draft head's after the model's): its attention's, its indexer's keys and pools, a latent attention's
/// latents and its indexer's keys, a compressed attention's window (its context's keys too). An operation of another ring kind is refused.
pub(super) fn shift_rings(d: &mut ModelDesc, by: usize) -> Result<()> {
    let family = d.family;
    let moved = |k: &mut IdxKeys| match k {
        IdxKeys::Ring(r) | IdxKeys::Entries(r) | IdxKeys::Pools(r) => *r += by,
    };
    for op in std::iter::once(&mut d.first).chain(d.layers.iter_mut().flatten()).chain(d.context.iter_mut()) {
        match op {
            Op::Attn(a) => a.ring += by,
            Op::Mla(a) => {
                a.ring += by;
                for p in a.indexer.as_mut().map(|x| &mut x.pick).into_iter().chain(a.lent.as_mut()) {
                    moved(&mut p.keys);
                }
            }
            Op::IdxMean { keys, pools, .. } => (*keys, *pools) = (*keys + by, *pools + by),
            Op::IdxPick(IdxPick { keys: IdxKeys::Pools(r), .. }) => *r += by,
            Op::CompKv { ring, .. } | Op::CompAttn { ring, cache: None, .. } => *ring += by,
            op if !op.rings().is_empty() => return err(format!("{family}: a draft head's {} names rings of a kind not moved", super::rows::op_name(op))),
            _ => {}
        }
    }
    Ok(())
}

/// The model `d`'s taps for its block draft head (`kinds::DSpark::targets` of the description `kinds`): at each target layer's input, the streams the layer's
/// attention collapses (after an n-gram block before it, as the release's main_hidden), the k-th target's tap in its place of Blk (`Op::Tap`). A target past
/// the model's layers, or a layer of no attention on the streams, is refused.
pub(super) fn tap(d: &mut ModelDesc, kinds: &crate::runtime::describe::Described) -> Result<()> {
    let ds = kinds.desc.mtp.first().and_then(|h| h.dspark.as_ref()).ok_or_else(|| Error(format!("{}: taps for a draft head of no DSpark form", d.family)))?;
    let of = ds.targets.len();
    for (at, &l) in ds.targets.iter().enumerate() {
        let layers = d.layers.len();
        let ops = d.layers.get_mut(l).ok_or_else(|| Error(format!("{}: a draft head's target layer {l}, past the model's {layers}", d.family)))?;
        let i = ops
            .iter()
            .position(|op| matches!(op, Op::StreamsPre { .. }))
            .ok_or_else(|| Error(format!("{}: layer {l}'s input is no streams' mix", d.family)))?;
        ops.insert(i, Op::Tap { at, of });
    }
    d.widths[Slot::Blk as usize] = of * d.hidden;
    Ok(())
}

impl Dyn {
    /// A draft's part of a prompt chunk the model just ran (its `tokens` from position `start`; `last`: the prompt's last chunk): a draft model's first input,
    /// the hidden state of the prompt's last row; a draft head's rows whose next token is known (the row the last chunk left waiting, this chunk's but its
    /// last), its last row then waiting in the head's Hid.
    ///
    /// # Safety
    /// The model's slots hold the chunk's rows (its pass just ran on the engine's stream).
    pub(super) unsafe fn draft_after_chunk(&self, start: usize, tokens: &[u32], last: bool) -> Result<()> {
        let Some(dr) = &self.draft else { return Ok(()) };
        let n = tokens.len();
        let Some(hs) = &dr.head else {
            if last {
                // (the draft model's first input: the hidden state of the prompt's last position)
                self.kern.copy_row(self.dev.stream, self.big.slot(Slot::H), n as i32 - 1, self.big.d.hidden as i32, dr.p.slot(Slot::Hid))?;
            }
            return Ok(());
        };
        if n == 0 {
            return Ok(());
        }
        // (a block head: its context of the chunk's rows, every one's next token known or not)
        if block_of(&dr.p.d).is_some() {
            return self.head_context(dr, start, tokens);
        }
        let (s, w, at) = (self.dev.stream, dr.p.d.width(Slot::Hid) * 2, self.big.slot(state(&dr.p.d)));
        let pend = hs.pending.take();
        if pend.is_some_and(|p| p + 1 != start) {
            return err(format!("a draft head's row at {pend:?} waits, where the chunk begins at {start}"));
        }
        let first = pend.is_some() as usize;
        let next: Vec<u32> = pend.map(|_| tokens[0]).into_iter().chain(tokens[1..].iter().copied()).collect();
        if !next.is_empty() {
            copy_d2d_async(dr.p.slot(Slot::Hid) + (first * w) as u64, at, (n - 1) * w, s)?;
            self.head_pass(dr, pend.unwrap_or(start), &next)?;
        }
        copy_d2d_async(dr.p.slot(Slot::Hid), at + ((n - 1) * w) as u64, w, s)?;
        hs.pending.set(Some(start + n - 1));
        Ok(())
    }

    /// The head on the rows whose streams are in its Hid (row 0 on), at positions `pos`.., each with the token after the model's row it continues (`next`): the
    /// last row's draft (and the pass kept as the head's last, `HeadState::last`).
    ///
    /// # Safety
    /// The head's Hid holds `next.len()` rows of streams (copies on the engine's stream queued before this).
    pub(super) unsafe fn head_pass(&self, dr: &Draft, pos: usize, next: &[u32]) -> Result<u32> {
        let hs = dr.head.as_ref().ok_or_else(|| Error("a head's pass on a draft model".into()))?;
        let (m, s) = (next.len(), self.dev.stream);
        if m == 0 || m > self.chunk {
            return err(format!("a draft head's pass of {m} rows, where a chunk is {}", self.chunk));
        }
        self.dev.sync()?;
        self.pos.write(0, &(0..m).map(|i| (pos + i) as i32).collect::<Vec<i32>>())?;
        self.start.set(pos);
        let traced = hs.trace.borrow().is_some();
        let mut streams = Vec::new();
        if traced {
            streams = vec![0u16; m * dr.p.d.width(Slot::Hid)];
            copy_d2h_async(streams.as_mut_ptr() as *mut _, dr.p.slot(Slot::Hid), streams.len() * 2, s)?;
        }
        self.embed(next)?;
        let pp = self.pos.ptr;
        // (a traced pass records its indexers' picks apart from any the model's passes record)
        let others = traced.then(|| self.pick_trace.replace(Some(Vec::new())));
        let run = self.forward(&dr.p, m, (pp, pp), 1);
        let picks = others.map_or(Vec::new(), |o| self.pick_trace.replace(o).unwrap_or_default());
        run?;
        copy_d2h_async(self.host.ptr, dr.p.tok.ptr, 4, s)?;
        self.dev.sync()?;
        let draft = *(self.host.ptr as *const i32) as u32;
        hs.last.set(Some(HeadLast { pos: pos + m - 1, row: m - 1, token: next[m - 1], draft }));
        if traced {
            let k = dr.p.moe.first().and_then(|b| b.top_k()).unwrap_or(0);
            let mut routes = vec![0i64; m * k];
            if let (true, Some(sc)) = (k > 0, self.moe_sc.get()) {
                sc.ids.read(0, &mut routes)?;
            }
            let mut logits = vec![0u16; dr.p.d.vocab];
            dr.p.logits.read(0, &mut logits)?;
            if let Some(t) = hs.trace.borrow_mut().as_mut() {
                t.push(HeadPass { pos, tokens: next.to_vec(), streams, routes, k, picks, logits, draft });
            }
        }
        Ok(draft)
    }

    /// The draft head's passes run between `trace_head(true)` and `trace_head(false)`, which returns them (`HeadPass`); none where the engine has no draft head.
    /// A block head's are `trace_block`'s, which this turns on and off with them.
    pub fn trace_head(&self, on: bool) -> Vec<HeadPass> {
        let Some(hs) = self.draft.as_ref().and_then(|d| d.head.as_ref()) else { return Vec::new() };
        *hs.blocks.borrow_mut() = on.then(Vec::new);
        std::mem::replace(&mut *hs.trace.borrow_mut(), on.then(Vec::new)).unwrap_or_default()
    }

    /// A block draft head's passes (its contexts and its blocks: `BlockTrace`) run since `trace_head(true)` (which begins them again), taken (the trace goes on
    /// where it was on).
    pub fn trace_block(&self) -> Vec<BlockTrace> {
        let Some(hs) = self.draft.as_ref().and_then(|d| d.head.as_ref()) else { return Vec::new() };
        hs.blocks.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// A block draft head's context of the rows the model's last pass ran (`tokens` from position `start`; their taps, Blk's first rows): its operations on
    /// them (`ModelDesc::context`), each layer's keys of main_x into its window at the rows' positions. The context then holds the positions to `start` +
    /// the rows; a row at a position past what it holds is an error (a pass the head did not see).
    ///
    /// # Safety
    /// The model's Blk holds the rows' taps (its pass ran on the engine's stream).
    pub(super) unsafe fn head_context(&self, dr: &Draft, start: usize, tokens: &[u32]) -> Result<()> {
        let hs = dr.head.as_ref().ok_or_else(|| Error("a block head's context on a draft model".into()))?;
        let (n, s) = (tokens.len(), self.dev.stream);
        if n == 0 {
            return Ok(());
        }
        if start > hs.ctx.get() || n > self.chunk {
            return err(format!("a block draft head's context of {n} rows at {start}, where it holds {} positions and a chunk {}", hs.ctx.get(), self.chunk));
        }
        let w = dr.p.d.width(Slot::Hid);
        self.dev.sync()?;
        self.pos.write(0, &(start..start + n).map(|x| x as i32).collect::<Vec<i32>>())?;
        copy_d2d_async(dr.p.slot(Slot::Hid), self.big.slot(Slot::Blk), n * w * 2, s)?;
        let pp = self.pos.ptr;
        for (i, op) in dr.p.d.context.iter().enumerate() {
            self.op(&dr.p, 0, i, op, n, (pp, pp), 0)?;
        }
        hs.ctx.set(start + n);
        if hs.blocks.borrow().is_some() {
            let mut taps = vec![0u16; n * w];
            copy_d2h_async(taps.as_mut_ptr() as *mut _, dr.p.slot(Slot::Hid), taps.len() * 2, s)?;
            self.dev.sync()?;
            if let Some(t) = hs.blocks.borrow_mut().as_mut() {
                t.push(BlockTrace::Context { pos: start, tokens: tokens.to_vec(), taps });
            }
        }
        Ok(())
    }

    /// A block draft head's drafts after `last` (the token after the cache's, at its position): one pass of the block, `last` then the noise tokens, its rows
    /// seeing each other (`Dyn::ahead`), each draft's logits by the model's LM head; then the Markov head's, one draft after another: the embedding of the token
    /// before it (`last`, then the draft before) by the Markov head's product added to its logits, the draft their largest. The first `k` of the block's.
    fn block_drafts(&self, dr: &Draft, last: u32, k: usize) -> Result<Vec<u32>> {
        let hs = dr.head.as_ref().ok_or_else(|| Error("a block head's drafts from a draft model".into()))?;
        let (block, noise, (me, mh), rank) = block_of(&dr.p.d).ok_or_else(|| Error("a block's drafts of a head of no block".into()))?;
        let (n0, s, vocab) = (self.n, self.dev.stream, dr.p.d.vocab);
        if hs.ctx.get() != n0 {
            return err(format!("a block draft head's context holds {} positions, where the sequence holds {n0} (a pass the head did not see?)", hs.ctx.get()));
        }
        if n0 + block > self.ctx + self.chunk {
            return err(format!("a block of {block} at {n0}, past the context of {}", self.ctx));
        }
        let toks: Vec<u32> = std::iter::once(last).chain(std::iter::repeat_n(noise, block - 1)).collect();
        self.dev.sync()?;
        self.pos.write(0, &(n0..n0 + block).map(|x| x as i32).collect::<Vec<i32>>())?;
        self.start.set(n0);
        let traced = hs.blocks.borrow().is_some();
        if traced {
            *self.head_routes.borrow_mut() = Some(Vec::new());
        }
        // SAFETY: live buffers of the engine, sized for a chunk (the block is at most one); the head's slots are its own.
        let run = unsafe {
            self.embed(&toks)?;
            let pp = self.pos.ptr;
            self.ahead.set(true);
            let r = self.forward(&dr.p, block, (pp, pp), block);
            self.ahead.set(false);
            r
        };
        let routes = self.head_routes.borrow_mut().take().unwrap_or_default();
        run?;
        let (table, Some(markov)) = (&dr.p.mats[me], dr.p.mats.get(mh)) else { return err("a block head's Markov matrices are not loaded") };
        let crate::kern35::Mat::Dense(table, rows, cols) = table else { return err("a block head's Markov embedding is not a table held as it is") };
        if (*rows, *cols) != (vocab, rank) || dr.p.d.width(Slot::Proj) < rank {
            return err(format!(
                "a block head's Markov embedding of {rows}x{cols} (a row into Proj of {}), where its rank is {rank}",
                dr.p.d.width(Slot::Proj)
            ));
        }
        let (row, bias) = (|j: usize| dr.p.logits.ptr + (j * vocab * 2) as u64, dr.p.logits.ptr + (block * vocab * 2) as u64);
        let (mut drafts, mut prev, mut logits) = (Vec::with_capacity(k), last, Vec::new());
        for j in 0..k.min(block) {
            // SAFETY: the table's row `prev` (prev below the vocabulary: a token), the head's slots and its logits' rows (block + 1 of them); synchronized below.
            unsafe {
                copy_d2d_async(dr.p.slot(Slot::Proj), table.ptr + (prev as usize * rank * 2) as u64, rank * 2, s)?;
                self.lin(markov, dr.p.slot(Slot::Proj), 1, bias)?;
                self.kres.add(s, row(j), bias, vocab)?;
                self.kern.argmax_rows(s, row(j), vocab as i32, 1, dr.p.tok.ptr + (j * 4) as u64, dr.p.val.ptr + (j * 4) as u64)?;
                copy_d2h_async(self.host.ptr, dr.p.tok.ptr + (j * 4) as u64, 4, s)?;
            }
            self.dev.sync()?;
            // SAFETY: the pinned host buffer holds the token just copied (synchronized).
            prev = unsafe { *(self.host.ptr as *const i32) } as u32;
            if prev as usize >= vocab {
                return err(format!("a block head's draft {j}: token {prev} past the vocabulary"));
            }
            drafts.push(prev);
            if traced {
                let mut w = vec![0u16; vocab];
                dr.p.logits.read(j * vocab * 2, &mut w)?;
                logits.push(w);
            }
        }
        if let Some(t) = hs.blocks.borrow_mut().as_mut() {
            t.push(BlockTrace::Block { pos: n0, anchor: last, drafts: drafts.clone(), logits, routes });
        }
        Ok(drafts)
    }

    /// Whether the engine drafts with a head of the model itself (`lower::draft_head`).
    pub fn drafts_with_head(&self) -> bool {
        self.draft.as_ref().is_some_and(|d| d.head.is_some())
    }

    /// A draft head's state forgotten (a new sequence: `reset`).
    pub(super) fn head_reset(&self) {
        if let Some(hs) = self.draft.as_ref().and_then(|d| d.head.as_ref()) {
            (hs.pending.set(None), hs.last.set(None), hs.ctx.set(0));
        }
    }

    /// One drafting round with a draft head: from `last` (the token after the cache's, not in it yet), the head's first draft (its waiting row given `last`, or
    /// the draft it made after `last` at the end of the round before), its next ones each from its own streams (`head_drafts`); the model checks `last` and the
    /// drafts in one verify pass (`verify_drafts`). The tokens returned (1 to k + 1) are what steps would give one by one; the cache then holds `last` and the
    /// kept drafts.
    pub(super) fn spec_round_head(&mut self, dr: &Draft, last: u32, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let n0 = self.n;
        if n0 == 0 {
            return err("spec_round before a prompt");
        }
        let k = dr.k.min(VERIFY_ROWS - 1).min(self.ctx.saturating_sub(n0 + 1));
        let drafts = self.head_drafts(dr, last, k)?;
        self.verify_drafts(dr, last, &drafts, sampler)
    }

    /// The head's k drafts after `last`: its first from its waiting row given `last` (a prompt's last row), or the one it made after `last` at the end of the round
    /// before; each next from its own streams of the draft before, at the next position.
    fn head_drafts(&self, dr: &Draft, last: u32, k: usize) -> Result<Vec<u32>> {
        if block_of(&dr.p.d).is_some() {
            return self.block_drafts(dr, last, k);
        }
        let hs = dr.head.as_ref().ok_or_else(|| Error("a head's drafts from a draft model".into()))?;
        let (n0, s, w) = (self.n, self.dev.stream, dr.p.d.width(Slot::Hid) * 2);
        let mut drafts = Vec::with_capacity(k);
        if k == 0 {
            return Ok(drafts);
        }
        let mut at = match hs.pending.take() {
            Some(p) if p + 1 == n0 => {
                // SAFETY: the waiting row's streams are in the head's Hid (`draft_after_chunk`).
                unsafe { self.head_pass(dr, p, &[last])? };
                hs.last.get().ok_or_else(|| Error("a head's pass that left no draft".into()))?
            }
            Some(p) => return err(format!("a draft head's row at {p} waits, where the sequence holds {n0}")),
            None => match hs.last.get() {
                Some(l) if l.token == last && l.pos + 1 == n0 => l,
                l => return err(format!("no draft after token {last} at {n0}: the head's last pass {l:?} (a prompt not run through `prefill`?)")),
            },
        };
        drafts.push(at.draft);
        while drafts.len() < k {
            // SAFETY: the head's last pass left its rows' state in its slot (`state`); its Hid takes the last of them.
            unsafe {
                copy_d2d_async(dr.p.slot(Slot::Hid), dr.p.slot(state(&dr.p.d)) + (at.row * w) as u64, w, s)?;
                self.head_pass(dr, at.pos + 1, &[at.draft])?;
            }
            at = hs.last.get().ok_or_else(|| Error("a head's pass that left no draft".into()))?;
            drafts.push(at.draft);
        }
        Ok(drafts)
    }

    /// `last` and `drafts` checked by the model in one verify pass (each row as its step: `verify_rows`), kept up to its first disagreement and its own token there
    /// (greedy), or each by `sampler`; the states go back to after the rows kept (`rewind`), and the head runs again on those rows' streams with the token after
    /// each, its rows there in place of its drafts', its last row's draft the next round's first.
    fn verify_drafts(&mut self, dr: &Draft, last: u32, drafts: &[u32], sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let n0 = self.n;
        let seq: Vec<u32> = std::iter::once(last).chain(drafts.iter().copied()).collect();
        let n = seq.len();
        let mut io = Vec::new();
        self.verify_rows(&seq, n0, &mut io, n)?;
        let (rows, _) = self.head_out(n, false)?;
        let big: Vec<u32> = rows.iter().map(|r| r.0).collect();
        // (each draft's check made of its row of the pass's head rows on the GPU: `verify_rows` marked them)
        let (out, kept) = super::accept(&seq, &big, &mut crate::sched::Step(self), sampler)?;
        self.rewind(kept + 1)?;
        if block_of(&dr.p.d).is_some() {
            // (a block head: its context of the kept rows, the round's last token and the kept drafts)
            // SAFETY: the model's Blk holds the verify pass's rows (`rewind` writes other buffers).
            unsafe { self.head_context(dr, n0, &seq[..=kept])? };
        } else {
            let next: Vec<u32> = seq[1..=kept].iter().copied().chain(out.last().copied()).collect();
            self.head_again(dr, n0, kept, &next)?;
        }
        self.prof_pass(n, 0)?;
        Ok(out)
    }

    /// After a round's verify pass at `n0` and `rewind` to its `kept` drafts and the row before them: the head again on those rows' streams, each with the token
    /// after it (`next`), its rows there in place of its drafts', its last row's draft the next round's first. Not a block head's (its context of the kept rows
    /// is `verify_drafts`').
    fn head_again(&self, dr: &Draft, n0: usize, kept: usize, next: &[u32]) -> Result<()> {
        if block_of(&dr.p.d).is_some() {
            return err(format!("a block draft head's rows again at {n0} ({kept} kept): its context is the verify pass's rows (`verify_drafts`), on one GPU"));
        }
        let (s, w) = (self.dev.stream, dr.p.d.width(Slot::Hid) * 2);
        // SAFETY: the model's state slot holds the verify pass's rows (`rewind` writes other slots); the copy is queued before the head's pass.
        unsafe {
            copy_d2d_async(dr.p.slot(Slot::Hid), self.big.slot(state(&dr.p.d)), (kept + 1) * w, s)?;
            self.head_pass(dr, n0, next)?;
        }
        Ok(())
    }

    /// A prompt chunk's rows through this stage of a model split across GPUs (`stage_rows`), and on the stage that drafts with a head of the model (the last) the
    /// head's rows of the chunk, as `prefill` hands them on one GPU (`draft_after_chunk`).
    pub fn stage_chunk(&mut self, tokens: &[u32], at: usize, io: &mut Vec<u16>, head: usize) -> Result<()> {
        self.stage_rows(tokens, at, io, head)?;
        // SAFETY: the stage's slots hold the chunk's rows (its pass just ran; `stage_rows` synchronized the stream).
        unsafe { self.draft_after_chunk(at, tokens, head > 0) }
    }

    /// The drafts of a round across GPUs, on the stage that drafts with a head of the model (the last): k after `last` (the head's and the context's most), as
    /// `spec_round` drafts them on one GPU; or `given`, after the head's own first pass all the same (`spec_round_given`'s way to every count kept).
    pub fn stage_drafts(&mut self, last: u32, given: Option<&[u32]>) -> Result<Vec<u32>> {
        let Some(dr) = self.draft.take() else { return err("drafts asked of a stage that does not draft") };
        let most = dr.k.min(VERIFY_ROWS - 1).min(self.ctx.saturating_sub(self.n + 1));
        let r = match (&dr.head, given) {
            (None, _) => err("drafts across GPUs are a draft head's of the model (a draft model's are not split)"),
            _ if self.n == 0 => err("a drafting round before a prompt"),
            (Some(_), None) => self.head_drafts(&dr, last, most),
            (Some(_), Some(d)) if !d.is_empty() && d.len() <= most => self.head_drafts(&dr, last, 1).map(|_| d.to_vec()),
            (Some(_), Some(d)) => err(format!("{} drafts given after {} tokens: 1 to {most}", d.len(), self.n)),
        };
        self.draft = Some(dr);
        r
    }

    /// The stage that drafts after a round's verify pass at `n0` and every stage's `rewind` to the `kept` drafts: the head again on the kept rows (`head_again`).
    pub fn stage_head_again(&mut self, n0: usize, kept: usize, next: &[u32]) -> Result<()> {
        let Some(dr) = self.draft.take() else { return err("the head asked of a stage that does not draft") };
        let r = if dr.head.is_some() { self.head_again(&dr, n0, kept, next) } else { err("a draft model's round across GPUs") };
        self.draft = Some(dr);
        r
    }

    /// A drafting round with `drafts` in place of the head's (its waiting row given `last` all the same, as a round takes it): a check's way to every count of drafts
    /// kept, which a random head's own drafts seldom reach.
    #[cfg(feature = "dev")]
    pub fn spec_round_given(&mut self, last: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        let Some(dr) = self.draft.take() else { return err("no draft model") };
        let r = match &dr.head {
            None => err("drafts given to a round of a draft model: a draft head's round takes them"),
            Some(_) if self.n == 0 || drafts.len() > dr.k.min(VERIFY_ROWS - 1) => {
                err(format!("{} drafts given after {} tokens: 1 to {} after a prompt", drafts.len(), self.n, dr.k.min(VERIFY_ROWS - 1)))
            }
            Some(_) => self.head_drafts(&dr, last, 1).and_then(|_| self.verify_drafts(&dr, last, drafts, None)),
        };
        self.draft = Some(dr);
        r
    }
}

/// `glyd-engine draftcheck CASE [--level L] [--k K] [--tokens N] [--sim] [--split CUT,.. --devices GPU,..]`: a family's draft head on the executor (`lower::draft_head`),
/// on its random case with its MTP layer: CASE `qwen4x` (`logits::qwen4x_mtp_random`: the `qwen4x` case's model to the bit and its head of the model's
/// streams), `glm-dsa` and `glm-dsa-f8` (`logits::glm_dsa_mtp_random`: those cases' models to the bit, bf16 and as released in FP8, and their joined head: a
/// latent attention with an indexer of its own and the experts), `deepseek-v4` and `deepseek-v41` (`logits::deepseek_v4_dspark_random`, `deepseek_v41_random`:
/// those cases' models to the bit and their DSpark drafters, a block head over the compressed attention: its verify pass and rewind, the windows' and the
/// compressors' across their blocks' boundaries; `--sim`: deepseek-v4 with the release's simulated quantizations, its head's figures said, not judged, as
/// logitscheck's third part). For a short prompt and one past two chunks: (1) the drafted greedy generation against the model alone, N tokens:
/// every token and every one of its rows of logits the same bits (the round's verify rows are the steps'), the drafts and how many were kept said (the
/// acceptance of a random head's own drafts: no quality); (2) the head held to the CPU reference of the kinds (`RefMtp::step_streams`, `RefMtp::step`,
/// `RefDSpark::draft_taking`) on the states and tokens it read: each of its passes' last row (a block's every draft) against the reference's on the head's
/// own choices (its experts, its indexer's picks, a block's Markov tokens), at the bound the family is held to (the case's fixture's,
/// tests/logits/CASE.json.gz), the figures on the reference's own choices printed beside them. With `--split` (the `qwen4x` case), (1) again with the model's
/// layers in stages (src/dynamic/split.rs; GPU 0 for each where `--devices` is not given), its head on the last: the tokens and rows the model's alone, and
/// each round's tokens the one-GPU drafted engine's (the head drafts there as on one GPU). `CASE+q4`: the case's model with its tables on the host stored
/// at 4 bits (the reference reads the values they decode to). An error where any fails.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    // (CASE+q4: its tables on the host stored at 4 bits, `logits::tables_at_4_bits`)
    let (case, q4) = crate::logits::q4_case(a.need(1, "CASE")?);
    let make: fn(&std::path::Path) -> Result<()> = match case {
        "qwen4x" => crate::logits::qwen4x_mtp_random,
        "glm-dsa" => |d| crate::logits::glm_dsa_mtp_random(d, false),
        "glm-dsa-f8" => |d| crate::logits::glm_dsa_mtp_random(d, true),
        "deepseek-v4" => crate::logits::deepseek_v4_dspark_random,
        "deepseek-v41" => crate::logits::deepseek_v41_random,
        _ => {
            return err(format!(
                "draftcheck {case}: the draft heads' cases are qwen4x, glm-dsa, glm-dsa-f8, deepseek-v4 and deepseek-v41 (each family's random model with its MTP layer or DSpark drafter)"
            ))
        }
    };
    let level = a.flag("level").unwrap_or("penguin");
    let (k, tokens): (usize, usize) = (a.num("k", 3usize)?, a.num("tokens", 24usize)?);
    let split = super::split::split_args(a)?;
    let sim = a.has("sim");
    if sim && case != "deepseek-v4" {
        return err(format!("draftcheck {case} --sim: the release's simulated quantizations are deepseek-v4's (V4.1's blocks of 32 are not simulated)"));
    }
    let dir = std::env::temp_dir().join(format!("glyd-draftcheck-{}", std::process::id()));
    make(&dir)?;
    if q4 {
        crate::logits::tables_at_4_bits(&dir, &|_| true)?;
    }
    let r = check_on(case, &dir, level, k, tokens, &split, sim);
    std::fs::remove_dir_all(&dir).ok();
    let (bits_ok, held_ok) = r?;
    if !(bits_ok && held_ok) {
        return err(format!(
            "draftcheck {case} at {level}: {}{}",
            if bits_ok { "" } else { "the drafted tokens are not the model's alone; " },
            if held_ok { "" } else { "the head is past its bound" }
        ));
    }
    println!("PASS draftcheck {case}{} at {level}", if q4 { ", its tables on the host at 4 bits" } else { "" });
    Ok(())
}

/// A drafted engine as `draftcheck` drives it: the model on one GPU (`Dyn`), or its layers in stages with the head on the last (`split::Split`).
#[cfg(feature = "dev")]
trait Drafted {
    /// A new sequence's prompt: the token after it and that row's logits (bf16 words).
    fn begin(&mut self, prompt: &[u32]) -> Result<(u32, Vec<u16>)>;
    /// A round from `last`, of the head's drafts or of those `given`: its tokens and their rows' logits.
    fn round(&mut self, last: u32, given: Option<&[u32]>) -> Result<(Vec<u32>, Vec<u16>)>;
}

#[cfg(feature = "dev")]
impl Drafted for Dyn {
    fn begin(&mut self, prompt: &[u32]) -> Result<(u32, Vec<u16>)> {
        self.reset()?;
        let s = self.prefill(prompt)?;
        Ok((s.token, self.logit_words()?))
    }
    fn round(&mut self, last: u32, given: Option<&[u32]>) -> Result<(Vec<u32>, Vec<u16>)> {
        let out = match given {
            Some(d) => self.spec_round_given(last, d)?,
            None => self.spec_round(last, None)?,
        };
        let words = self.head_out(out.len(), true)?.1;
        Ok((out, words))
    }
}

#[cfg(feature = "dev")]
impl Drafted for super::split::Split {
    fn begin(&mut self, prompt: &[u32]) -> Result<(u32, Vec<u16>)> {
        self.reset()?;
        let s = self.prefill(prompt)?;
        // (the last stage's row, widened from its bf16 words: their bits again)
        Ok((s.token, self.logits()?.iter().map(|x| (x.to_bits() >> 16) as u16).collect()))
    }
    fn round(&mut self, last: u32, given: Option<&[u32]>) -> Result<(Vec<u32>, Vec<u16>)> {
        let out = self.spec_round_with(last, given, None)?;
        let words = self.head_out(out.len(), true)?.1;
        Ok((out, words))
    }
}

/// `tokens` greedy tokens of `e` drafting after `prompt` (round r at token `at` of `given(r, at)`'s drafts where it gives some, else the head's): the tokens, their
/// rows of logits and each round's tokens.
#[cfg(feature = "dev")]
#[allow(clippy::type_complexity)]
fn drafted_run(
    e: &mut dyn Drafted,
    prompt: &[u32],
    tokens: usize,
    given: &dyn Fn(usize, usize) -> Option<Vec<u32>>,
) -> Result<(Vec<u32>, Vec<Vec<u16>>, Vec<Vec<u32>>)> {
    let (t, row) = e.begin(prompt)?;
    let (mut got, mut rows, mut rounds) = (vec![t], vec![row], Vec::new());
    while got.len() < tokens {
        let (out, words) = e.round(*got.last().unwrap_or(&0), given(rounds.len(), got.len()).as_deref())?;
        rows.extend(words.chunks(words.len() / out.len().max(1)).map(<[u16]>::to_vec));
        got.extend(&out);
        rounds.push(out);
    }
    got.truncate(tokens);
    rows.truncate(tokens);
    Ok((got, rows, rounds))
}

/// Drafted replies with a sampler (each draft kept by its draw: `Sampler::check_draft`, the round's last token drawn): with the drafts' checks and draws made of each verify
/// row on the GPU (`sched::check_row`: the row's candidates and its weights and tokens by range, no row read) against the same reply with every row read on the host
/// (GLYD_ENGINE_HOST_SAMPLING's), seed for seed, at a top-p with penalties and at a min-p with a bias; and across the stages, the one GPU's. Whether every one holds, and no
/// row was read on the host but the host's runs'.
#[cfg(feature = "dev")]
fn sampled_drafts(drafted: &mut Dyn, staged: Option<&mut super::split::Split>, prompts: &[&[u32]], tokens: usize) -> Result<bool> {
    use crate::conv::Gen;
    use crate::sampler::{Params, DRAWS};
    use std::sync::atomic::Ordering::Relaxed;
    let mut ok_all = true;
    let mut staged = staged;
    for (i, prompt) in prompts.iter().enumerate() {
        for (what, params) in [
            ("top-p 0.95, repetition 1.1", Params { temperature: 0.9, top_p: 0.95, repetition_penalty: 1.1, seed: 11 + i as u64, ..Params::default() }),
            (
                "min-p 0.05, presence 0.4, a bias",
                Params { temperature: 1.1, min_p: 0.05, presence_penalty: 0.4, logit_bias: vec![(3, 2.0)], seed: 21 + i as u64, ..Params::default() },
            ),
        ] {
            let reply = crate::sched::Reply { params, max_new: tokens, ignore_eos: true, logprobs: None, stop_ids: Vec::new() };
            let read = || DRAWS[0].load(Relaxed);
            let r0 = read();
            drafted.host_sampling = false;
            let (gpu, _) = drafted.reply(prompt, &reply, &mut |_| true)?;
            let gpu_reads = read() - r0;
            drafted.host_sampling = true;
            let r1 = read();
            let host = drafted.reply(prompt, &reply, &mut |_| true);
            drafted.host_sampling = false;
            let (host, _) = host?;
            let host_reads = read() - r1;
            let (staged_toks, staged_reads) = match staged.as_mut() {
                Some(st) => {
                    let r2 = read();
                    let (t, _) = st.reply(prompt, &reply, &mut |_| true)?;
                    (Some(t), read() - r2)
                }
                None => (None, 0),
            };
            let ok = gpu == host && gpu_reads == 0 && host_reads > 0 && staged_toks.as_ref().is_none_or(|t| *t == gpu) && staged_reads == 0;
            ok_all &= ok;
            println!(
                "{} drafted with a sampler ({what}), a prompt of {}: {} tokens, {} the host's draws from the rows, seed for seed; rows read on the host {gpu_reads} (the host's run {host_reads}){}",
                if ok { "PASS" } else { "FAIL" },
                prompt.len(),
                gpu.len(),
                if gpu == host { "every token" } else { "NOT" },
                staged_toks.map_or(String::new(), |t| format!("; across the stages {} the one GPU's, {staged_reads} rows read", if t == gpu { "every token" } else { "NOT" }))
            );
        }
    }
    Ok(ok_all)
}

/// The model of `case` in `dir` alone, and drafting with its head `k` tokens a round, each by its family's loader (`sim`: the compressed attention's model
/// with the release's simulated quantizations, else without).
#[cfg(feature = "dev")]
fn engines(case: &str, dir: &std::path::Path, o: &super::Opts, k: usize, sim: bool) -> Result<(Dyn, Dyn)> {
    Ok(match case {
        "qwen4x" => {
            let experts = super::qwen4x::experts_at(o.layout);
            (super::qwen4x::load_level(dir, o, experts)?, super::qwen4x::load_drafting(dir, o, experts, k)?)
        }
        "deepseek-v4" | "deepseek-v41" => (super::deepseek_v4::load_sim(dir, o, sim)?, super::deepseek_v4::load_drafting(dir, o, k, Some(sim))?),
        _ => (super::glm::load_plain(dir, o)?, super::glm::load_plain_drafting(dir, o, k)?),
    })
}

/// `check`'s parts on the model of `case` in `dir`: whether the drafted generations are the model's alone, bit for bit (and across the stages of `split`
/// where it cuts any: the one GPU's rounds too), and whether the head is within its bound.
#[cfg(feature = "dev")]
fn check_on(case: &str, dir: &std::path::Path, level: &str, k: usize, tokens: usize, split: &(Vec<usize>, Vec<i32>), sim: bool) -> Result<(bool, bool)> {
    use crate::logits::{compare, level_layout, Bound, Fixture, HeadRef, HeadRow, Row};
    let layout = level_layout(level)?;
    let o = super::Opts { ctx: 1024, chunk: 64, graph: true, device: 0, layout };
    let fx = Fixture::read(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/logits/{case}.json.gz")))?;
    let bound = fx.bounds.first().map(|b| b.bound.clone()).ok_or_else(|| Error(format!("the {case} fixture has no bound")))?;
    let (mut plain, mut drafted) = engines(case, dir, &o, k, sim)?;
    // (a block head drafts at most its block a round)
    let own_k = match drafted.draft.as_ref().map(|x| &x.p.d) {
        Some(h) => block_of(h).map_or(k, |b| k.min(b.0)),
        None => k,
    };
    let vocab = plain.big.d.vocab;
    let mut staged = match split {
        (cuts, _) if cuts.is_empty() => None,
        (cuts, devices) if case == "qwen4x" => match super::qwen4x::open_split(dir, &o, super::qwen4x::experts_at(o.layout), cuts, devices, Some(k))? {
            super::split::Run::Split(s) if s.drafting() => Some(s),
            _ => return err(format!("draftcheck --split {cuts:?}: the stages do not draft")),
        },
        _ => return err(format!("draftcheck {case} --split: the check across stages takes the qwen4x case")),
    };
    let (short, _, long) = crate::logits::standard_tokens(vocab);
    // (the compressed attention's compressors' ratios past 1: their windows' boundaries)
    let ratios: std::collections::BTreeSet<usize> =
        plain.big.d.layers.iter().flatten().filter_map(|op| if let Op::Compress { ratio, .. } = *op { Some(ratio) } else { None }).filter(|&r| r > 1).collect();
    let (mut bits_ok, mut held_ok) = (true, true);
    drafted.trace_head(true);
    for (what, prompt) in [("a prompt of 24", &short), ("a prompt of 150", &long)] {
        // the model alone: its greedy tokens and each one's row of logits
        plain.reset()?;
        let mut s = plain.prefill(prompt)?;
        let (mut want, mut want_rows) = (vec![s.token], vec![plain.logit_words()?]);
        // (k more than asked: the drafts given below)
        while want.len() < tokens + k + 1 {
            s = plain.step(s.token)?;
            want.push(s.token);
            want_rows.push(plain.logit_words()?);
        }
        // (the drafts given: the model's own next tokens, round r's (r mod (k + 1))-th changed (none where that is k), so that every count of drafts kept, 0 to
        // k, is gone back to: a random head's own drafts are seldom the model's)
        let given = |r: usize, at: usize| -> Option<Vec<u32>> {
            let mut d: Vec<u32> = want[at..at + k].to_vec();
            if let Some(x) = d.get_mut(r % (k + 1)) {
                *x = (*x + 1) % vocab as u32;
            }
            Some(d)
        };
        // (a run against the model alone: every token and row of logits; the rows the same and the first token apart)
        let judge = |got: &[u32], rows: &[Vec<u16>]| {
            let same = rows.iter().zip(&want_rows).filter(|(a, b)| a == b).count();
            (got == &want[..tokens] && same == tokens, same, got.iter().zip(&want).position(|(a, b)| a != b))
        };
        let tokens_said = |got: &[u32]| if got == &want[..tokens] { "every token" } else { "NOT" };
        // drafted: the prompt's row, then each round's tokens and its verify rows' logits
        let (got, rows, own) = drafted_run(&mut drafted, prompt, tokens, &|_, _| None)?;
        let ((ok, same, first), kept) = (judge(&got, &rows), own.iter().map(|r| r.len() - 1).sum::<usize>());
        bits_ok &= ok;
        if !ok {
            // (where a round's verify rows part from the steps': the first operation whose rows do, layer by layer)
            match first_apart(&mut plain, &mut drafted, prompt) {
                Ok(Some(at)) => println!("     the first round's verify pass parts from the steps at {at}"),
                Ok(None) => println!("     the first round's verify pass is the steps' at every operation of every layer"),
                Err(e) => println!("     the first round's operations not compared: {}", e.msg),
            }
            drafted.trace_head(true);
        }
        // (the acceptance of the head's own drafts: of the random model, no quality)
        let made = own.len() * own_k;
        println!(
            "{} drafted {what}, {tokens} tokens: {} the model's alone, rows of logits the same bits {same} of {tokens}{}; {} rounds of {own_k} drafts, {kept} drafts kept ({:.2} tokens a round; acceptance {kept} of {made}, {:.1}%)",
            if ok { "PASS" } else { "FAIL" },
            tokens_said(&got),
            first.map_or(String::new(), |i| format!(" (the first token apart: {i}, {} against {})", got[i], want[i])),
            own.len(),
            tokens as f64 / own.len().max(1) as f64,
            100.0 * kept as f64 / made.max(1) as f64
        );
        let kept_by = |rounds: &[Vec<u32>]| {
            let mut by = vec![0usize; k + 1];
            rounds.iter().for_each(|r| by[(r.len() - 1).min(k)] += 1);
            by
        };
        let (got, rows, rounds) = drafted_run(&mut drafted, prompt, tokens, &given)?;
        let ((ok, same, _), by) = (judge(&got, &rows), kept_by(&rounds));
        // (a compressed attention's: the rounds whose refused rows completed a window of one of its compressors (its ratios past 1), which `rewind` went back
        // across; a round at n0 checks k drafts, keeps the tokens it gives but the last, and refuses the rows after them)
        let (mut n0, mut crossed) = (prompt.len(), 0);
        for r in &rounds {
            crossed += (n0 + r.len()..=n0 + k).any(|q| ratios.iter().any(|&x| (q + 1) % x == 0)) as usize;
            n0 += r.len();
        }
        let ok = ok && by.iter().all(|&c| c > 0) && (ratios.is_empty() || crossed > 0);
        bits_ok &= ok;
        println!(
            "{} drafts given, {what}, {tokens} tokens: {} the model's alone, rows of logits the same bits {same} of {tokens}; {} rounds, of each count of drafts kept 0 to {k}: {by:?}{}",
            if ok { "PASS" } else { "FAIL" },
            tokens_said(&got),
            rounds.len(),
            if ratios.is_empty() { String::new() } else { format!("; {crossed} rounds went back across a compressed block's boundary (ratios {ratios:?})") }
        );
        // across the stages: the head's own drafts (each round the one GPU's), then the drafts given
        if let Some(st) = staged.as_mut() {
            let (got, rows, rounds) = drafted_run(st, prompt, tokens, &|_, _| None)?;
            let (ok, same, first) = judge(&got, &rows);
            let ok = ok && rounds == own;
            bits_ok &= ok;
            println!(
                "{} drafted across the stages {:?} on GPUs {:?}, {what}: {} the model's alone, rows of logits the same bits {same} of {tokens}{}; {} rounds, {} the one GPU's round for round",
                if ok { "PASS" } else { "FAIL" },
                split.0,
                split.1,
                tokens_said(&got),
                first.map_or(String::new(), |i| format!(" (the first token apart: {i}, {} against {})", got[i], want[i])),
                rounds.len(),
                if rounds == own { "every one" } else { "NOT" }
            );
            let (got, rows, rounds) = drafted_run(st, prompt, tokens, &given)?;
            let ((ok, same, _), by) = (judge(&got, &rows), kept_by(&rounds));
            let ok = ok && by.iter().all(|&c| c > 0);
            bits_ok &= ok;
            println!(
                "{} drafts given across the stages, {what}: {} the model's alone, rows of logits the same bits {same} of {tokens}; {} rounds, of each count of drafts kept 0 to {k}: {by:?}",
                if ok { "PASS" } else { "FAIL" },
                tokens_said(&got),
                rounds.len()
            );
        }
    }
    // the head against the reference, a row of each pass on the state and the token it read, its experts and its indexer's picks the head's own
    let blocks = drafted.trace_block();
    let passes = drafted.trace_head(false);
    bits_ok &= sampled_drafts(&mut drafted, staged.as_mut(), &[&short, &long], tokens)?;
    if !blocks.is_empty() {
        held_ok &= blocks_held(dir, &blocks, &fx, &bound, sim)?;
        return Ok((bits_ok, held_ok));
    }
    let w = passes.first().map_or(0, |p| p.streams.len() / p.tokens.len().max(1));
    let widen = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect() };
    let mut rows = Vec::new();
    let mut last_of = Vec::new();
    for p in &passes {
        for (i, &t) in p.tokens.iter().enumerate() {
            let take = (p.k > 0).then(|| p.routes[i * p.k..(i + 1) * p.k].iter().map(|&e| e as usize).collect());
            let picks = p.picks.first().and_then(|x| x.picks.get(i).cloned());
            rows.push(HeadRow { pos: (p.pos + i) as u32, token: t, streams: widen(&p.streams[i * w..(i + 1) * w]), take, picks });
        }
        last_of.push(rows.len() - 1);
    }
    let engine: Vec<Vec<f32>> = passes.iter().map(|p| widen(&p.logits)).collect();
    // (a prompt run after the other begins a new sequence: the reference's head too)
    let mut on_choices = Vec::new();
    let mut own = Vec::new();
    let (mut apart, mut picks_apart) = (0, 0);
    for part in split_sequences(&rows, &last_of) {
        let HeadRef { logits: r, routes, .. } = crate::logits::ref_head(dir, &part.rows, |_| {})?;
        let free: Vec<HeadRow> = part.rows.iter().map(|x| HeadRow { take: None, picks: None, ..x.clone() }).collect();
        let HeadRef { logits: f, routes: froutes, picks: fpicks, .. } = crate::logits::ref_head(dir, &free, |_| {})?;
        let set = |v: &[usize]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v
        };
        apart += routes.iter().zip(&froutes).filter(|(a, b)| set(&a.1) != set(&b.0)).count();
        picks_apart += part.rows.iter().zip(&fpicks).filter(|(x, p)| x.picks.as_ref().is_some_and(|e| *e != p.0)).count();
        on_choices.extend(part.last.iter().map(|&i| r[i].clone()));
        own.extend(part.last.iter().map(|&i| f[i].clone()));
    }
    let held = |refs: &[Vec<f32>]| -> Result<crate::logits::Verdict> {
        let f = Fixture { rows: refs.iter().map(|l| Row::from_logits(l)).collect::<Result<_>>()?, ..fx.clone() };
        Ok(compare(&f, &bound, &engine))
    };
    let (v, raw) = (held(&on_choices)?, held(&own)?);
    held_ok &= v.ok;
    let bound_s = match &bound {
        Bound::Kl { mean, max, top1 } => format!("KL mean {mean:.1e}, max {max:.1e}, top-1 {top1}"),
        Bound::Bits => "bits".into(),
    };
    println!(
            "{} the draft head against the CPU reference on its own choices, {} passes ({} rows): KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}; bound {bound_s}{}",
            if v.ok { "PASS" } else { "FAIL" },
            passes.len(),
            rows.len(),
            v.mean_kl,
            v.max_kl,
            v.top1.0,
            v.top1.1,
            v.max_dlogit,
            if v.notes.is_empty() { String::new() } else { format!(" ({})", v.notes.join("; ")) }
        );
    println!(
        "     on the reference's own choices: KL mean {:.2e} max {:.2e}, best token {}/{}; its experts apart from the head's at {apart} of {} rows, its indexer's picks at {picks_apart}",
        raw.mean_kl,
        raw.max_kl,
        raw.top1.0,
        raw.top1.1,
        rows.len()
    );
    Ok((bits_ok, held_ok))
}

/// Where the first drafting round's verify pass after `prompt` parts from the model's steps over the same tokens: for each layer in turn, each operation's
/// slots (`TRACED`) the step wrote, row j of the pass against step j's row; the first (layer, operation, slot, row) apart. None: none.
#[cfg(feature = "dev")]
fn first_apart(plain: &mut Dyn, drafted: &mut Dyn, prompt: &[u32]) -> Result<Option<String>> {
    for l in 0..plain.big.d.layers.len() {
        drafted.reset()?;
        let t0 = drafted.prefill(prompt)?.token;
        drafted.trace_head(true);
        drafted.trace_ops(Some(l));
        drafted.spec_round(t0, None)?;
        let pass = drafted.trace_ops(None);
        // (the round's tokens: the last one, then the head's drafts in the order it made them; a block head's block's)
        let block: Vec<u32> =
            drafted.trace_block().into_iter().find_map(|b| if let BlockTrace::Block { drafts, .. } = b { Some(drafts) } else { None }).unwrap_or_default();
        let heads = drafted.trace_head(false);
        let k = pass.first().map_or(0, |(_, sl)| sl[0].len() / plain.big.d.width(Slot::H).max(1));
        let drafts: Vec<u32> = if block.is_empty() { heads.iter().map(|h| h.draft).collect() } else { block };
        let seq: Vec<u32> = std::iter::once(t0).chain(drafts).take(k).collect();
        plain.reset()?;
        plain.prefill(prompt)?;
        plain.trace_ops(Some(l));
        for &t in &seq {
            plain.step(t)?;
        }
        let steps = plain.trace_ops(None);
        let per = pass.len();
        if steps.len() != per * seq.len() {
            return err(format!("layer {l}: {} operations in the steps, {per} in the verify pass of {} rows", steps.len(), seq.len()));
        }
        for i in 0..per {
            for (j, step) in steps.chunks(per).enumerate() {
                for (si, slot) in super::TRACED.iter().enumerate() {
                    let w = plain.big.d.width(*slot);
                    let now = &step[i].1[si];
                    let wrote = i == 0 || step[i - 1].1[si] != *now;
                    let got = pass[i].1[si].get(j * w..(j + 1) * w);
                    if w > 0 && wrote && got != now.get(..w) {
                        return Ok(Some(format!("layer {l} operation {i} ({}), slot {slot:?}, row {j}", pass[i].0)));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// A block head's passes (`trace_block`: its contexts and its blocks) held to the CPU reference of the kinds (`RefDSpark::draft_taking`): each block on the
/// context of the taps it read (each position's latest context row: a row the model kept), every draft's logits against the reference's on the head's own
/// choices (each of its blocks of experts' choices, the Markov head's tokens before each draft), at the case's `bound` (judged where the release's quantizations
/// are not simulated: `sim` says the figures alone, as logitscheck's third part); the figures on the reference's own choices said beside them, and the model's
/// taps against the reference model's own of each sequence's tokens (`logits::ref_taps`: where the taps are) with the head on those.
#[cfg(feature = "dev")]
fn blocks_held(dir: &std::path::Path, blocks: &[BlockTrace], fx: &crate::logits::Fixture, bound: &crate::logits::Bound, sim: bool) -> Result<bool> {
    use crate::logits::{compare, Bound, Fixture, Row};
    let (embed, head, mut ds) = crate::logits::ref_dspark(dir)?;
    let widen = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect() };
    let block = ds.head.dspark.as_ref().map_or(1, |x| x.block).max(1);
    let width = ds.head.dspark.as_ref().map_or(0, |x| x.targets.len()) * ds.hidden;
    let moe: Vec<usize> = (0..ds.blocks.len()).filter(|&b| matches!(ds.blocks[b].block.body, crate::runtime::kinds::Body::Ffn(_))).collect();
    // (each sequence: its tokens and the model's taps by position, and its blocks (the context they read, their trace))
    struct SeqOf<'a> {
        tokens: Vec<u32>,
        taps: Vec<Vec<f32>>,
        blocks: Vec<(Vec<Vec<f32>>, &'a BlockTrace)>,
    }
    let mut seqs: Vec<SeqOf> = Vec::new();
    for t in blocks {
        match t {
            BlockTrace::Context { pos, tokens, taps } => {
                if *pos == 0 || seqs.is_empty() {
                    seqs.push(SeqOf { tokens: Vec::new(), taps: Vec::new(), blocks: Vec::new() });
                }
                let s = seqs.last_mut().expect("a sequence");
                // (a context row takes its position's place: the rows the model kept)
                if *pos > s.taps.len() || width == 0 || taps.len() != tokens.len() * width {
                    return err(format!("a block head's context at {pos}, where {} positions were held before it (taps {width} wide)", s.taps.len()));
                }
                (s.taps.truncate(*pos), s.tokens.truncate(*pos));
                s.taps.extend(taps.chunks(width).map(widen));
                s.tokens.extend(tokens);
            }
            BlockTrace::Block { pos, .. } => {
                let s = seqs.last_mut().ok_or_else(|| Error("a block before any context".into()))?;
                if *pos > s.taps.len() {
                    return err(format!("a block at {pos} on a context of {}", s.taps.len()));
                }
                s.blocks.push((s.taps[..*pos].to_vec(), t));
            }
        }
    }
    // (the head on a context: each draft's logits on the head's own choices, on the reference's own, and the counts apart; `take`: on the head's choices)
    let mut draft_on = |c: &[Vec<f32>], t: &BlockTrace, take: bool| -> Result<(Vec<Vec<f32>>, usize, usize, Vec<u32>)> {
        let BlockTrace::Block { anchor, drafts, logits, routes, .. } = t else { unreachable!() };
        if routes.len() != moe.len() {
            return err(format!("{} blocks of experts traced of the head's {}", routes.len(), moe.len()));
        }
        for (&b, (_, ids)) in moe.iter().zip(routes) {
            let k = (ids.len() / block).max(1);
            ds.blocks[b].take = if take { ids.chunks(k).map(|r| r.iter().map(|&e| e as usize).collect()).collect() } else { Default::default() };
            ds.blocks[b].routed = Some(Vec::new());
        }
        let r = ds.draft_taking(c, *anchor, &embed, &head, if take { drafts } else { &[] })?;
        let (mut rows, mut apart) = (0, 0);
        for &b in &moe {
            let rt = ds.blocks[b].routed.take().unwrap_or_default();
            rows += rt.len();
            apart += rt.iter().filter(|x| x.own.iter().collect::<std::collections::BTreeSet<_>>() != x.ids.iter().collect()).count();
            ds.blocks[b].take.clear();
        }
        let m = logits.len().min(r.logits.len());
        Ok((r.logits[..m].to_vec(), rows, apart, r.tokens))
    };
    let (mut engine, mut on_choices, mut own, mut on_ref) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut passes, mut routes_apart, mut routed, mut tokens_apart, mut taps_rel) = (0, 0, 0, 0, Vec::new());
    for s in &seqs {
        // (the reference model's own taps of the sequence's tokens: each row's difference from the model's relative to its size, L2)
        let want = crate::logits::ref_taps(dir, &s.tokens)?;
        for (a, b) in s.taps.iter().zip(&want) {
            let (d, n) = a.iter().zip(b).fold((0f64, 0f64), |(d, n), (x, y)| (d + ((x - y) as f64).powi(2), n + (*y as f64).powi(2)));
            taps_rel.push((d / n.max(1e-30)).sqrt());
        }
        for (c, t) in &s.blocks {
            let BlockTrace::Block { pos, drafts, logits, .. } = t else { unreachable!() };
            passes += 1;
            let (r, rows, apart, _) = draft_on(c, t, true)?;
            (routed, routes_apart) = (routed + rows, routes_apart + apart);
            engine.extend(logits[..r.len()].iter().map(|l| widen(l)));
            let (f, _, _, toks) = draft_on(c, t, false)?;
            tokens_apart += toks.iter().zip(drafts).filter(|(a, b)| a != b).count();
            own.extend(f[..r.len()].iter().cloned());
            on_ref.extend(draft_on(&want[..*pos], t, false)?.0[..r.len()].iter().cloned());
            on_choices.extend(r);
        }
    }
    let held = |refs: &[Vec<f32>]| -> Result<crate::logits::Verdict> {
        let f = Fixture { rows: refs.iter().map(|l| Row::from_logits(l)).collect::<Result<_>>()?, ..fx.clone() };
        Ok(compare(&f, bound, &engine))
    };
    let (v, raw, taps) = (held(&on_choices)?, held(&own)?, held(&on_ref)?);
    let bound_s = match bound {
        Bound::Kl { mean, max, top1 } => format!("KL mean {mean:.1e}, max {max:.1e}, top-1 {top1}"),
        Bound::Bits => "bits".into(),
    };
    println!(
        "{} the DSpark head against the CPU reference on its own choices, {passes} blocks ({} drafts): KL mean {:.2e} max {:.2e}, best token {}/{}, largest logit difference {:.4}; bound {bound_s}{}{}",
        if sim { "SAID" } else if v.ok { "PASS" } else { "FAIL" },
        engine.len(),
        v.mean_kl,
        v.max_kl,
        v.top1.0,
        v.top1.1,
        v.max_dlogit,
        if sim { " (the release's quantizations simulated: not judged)" } else { "" },
        if v.notes.is_empty() { String::new() } else { format!(" ({})", v.notes.join("; ")) }
    );
    println!(
        "     on the reference's own choices: KL mean {:.2e} max {:.2e}, best token {}/{}; its experts apart from the head's at {routes_apart} of {routed} rows, its drafts at {tokens_apart} of {}",
        raw.mean_kl,
        raw.max_kl,
        raw.top1.0,
        raw.top1.1,
        engine.len()
    );
    taps_rel.sort_by(f64::total_cmp);
    let at = |q: f64| taps_rel.get(((taps_rel.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0.0);
    println!(
        "     the model's taps against the reference model's own on its own choices ({} sequences, {} rows): a row's difference relative to its size: median {:.2e}, 90% {:.2e}, largest {:.2e}; the head on those taps and its own choices: KL mean {:.2e} max {:.2e}, best token {}/{}",
        seqs.len(),
        taps_rel.len(),
        at(0.5),
        at(0.9),
        at(1.0),
        taps.mean_kl,
        taps.max_kl,
        taps.top1.0,
        taps.top1.1
    );
    Ok(sim || v.ok)
}

/// The head's rows of each sequence (a row at position 0 begins one), and where each pass's last row is among them.
#[cfg(feature = "dev")]
struct Seq {
    rows: Vec<crate::logits::HeadRow>,
    last: Vec<usize>,
}

#[cfg(feature = "dev")]
fn split_sequences(rows: &[crate::logits::HeadRow], last_of: &[usize]) -> Vec<Seq> {
    let mut out: Vec<Seq> = Vec::new();
    let mut from = 0;
    for (i, r) in rows.iter().enumerate() {
        if r.pos == 0 || out.is_empty() {
            out.push(Seq { rows: Vec::new(), last: Vec::new() });
            from = i;
        }
        let s = out.last_mut().expect("a sequence");
        s.rows.push(r.clone());
        if last_of.contains(&i) {
            s.last.push(i - from);
        }
    }
    out
}
