//! A draft head of the model itself on the executor (`kinds::MtpInput::Streams`, Qwen3.8-Flash-Next's MTP layer; `lower::qwen4x_head`): a draft part whose
//! input is the model's last streams of a row and the token after it (`Op::MtpIn`), its blocks the model's kinds, its keys and values in rings after the model's,
//! the model's own LM head. A head's row at position p reads the model's streams at p and the token at p + 1, and drafts the token at p + 2; its own streams
//! after its blocks are the next draft's input.
//!
//! The head runs a row behind the model: after each prompt chunk, on the chunk's rows whose next token is known (the row the last chunk left waiting, then this
//! chunk's but its last, which waits in the head's Hid for the token the model picks); a drafting round (`spec_round_head`) gives the waiting row that token
//! (or takes the draft the head made after the last round), drafts on its own streams, has the model check the drafts in one verify pass (each row as its
//! step: `Dyn::verify_rows`, the states gone back to after the rows kept: `Dyn::rewind`), and runs the head again on the kept rows' streams, its rows at those
//! positions in place of its drafts' (what its caches held past them is written again before a row reads it).
//!
//! A model split across GPUs (src/dynamic/split.rs) drafts with the head on its last stage, which holds the model's last streams, its LM head and the embedding
//! the head reads: the same steps, a round's in its parts (`stage_chunk`, `stage_drafts`, every stage's verify pass and rewind, `stage_head_again`).

use super::{copy_d2d_async, copy_d2h_async, Draft, Dyn, HeadLast, IdxKeys, IdxPick, ModelDesc, Op, Slot, VERIFY_ROWS};
use crate::ffi::{err, Error, Result};

/// One pass of the head as `Dyn::trace_head` keeps it: the position of its first row, each row's token (the one after the model's row it continues), the
/// streams each row read (bf16, [rows, streams hidden]), its experts' choices ([rows, k], the router's order; empty where the block leaves none), the last
/// row's logits (bf16 words) and its draft.
#[derive(Clone, Debug)]
pub struct HeadPass {
    pub pos: usize,
    pub tokens: Vec<u32>,
    pub streams: Vec<u16>,
    pub routes: Vec<i64>,
    pub k: usize,
    pub logits: Vec<u16>,
    pub draft: u32,
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

/// The rings `d`'s operations name moved `by` places (a draft head's after the model's): its attention's, its indexer's keys and pools. An operation of another
/// ring kind is refused.
pub(super) fn shift_rings(d: &mut ModelDesc, by: usize) -> Result<()> {
    let family = d.family;
    for op in std::iter::once(&mut d.first).chain(d.layers.iter_mut().flatten()) {
        match op {
            Op::Attn(a) => a.ring += by,
            Op::IdxMean { keys, pools, .. } => (*keys, *pools) = (*keys + by, *pools + by),
            Op::IdxPick(IdxPick { keys: IdxKeys::Pools(r), .. }) => *r += by,
            op if !op.rings().is_empty() => return err(format!("{family}: a draft head's {} names rings of a kind not moved", super::rows::op_name(op))),
            _ => {}
        }
    }
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
        let (s, w) = (self.dev.stream, dr.p.d.width(Slot::Hid) * 2);
        let pend = hs.pending.take();
        if pend.is_some_and(|p| p + 1 != start) {
            return err(format!("a draft head's row at {pend:?} waits, where the chunk begins at {start}"));
        }
        let first = pend.is_some() as usize;
        let next: Vec<u32> = pend.map(|_| tokens[0]).into_iter().chain(tokens[1..].iter().copied()).collect();
        if !next.is_empty() {
            copy_d2d_async(dr.p.slot(Slot::Hid) + (first * w) as u64, self.big.slot(Slot::S), (n - 1) * w, s)?;
            self.head_pass(dr, pend.unwrap_or(start), &next)?;
        }
        copy_d2d_async(dr.p.slot(Slot::Hid), self.big.slot(Slot::S) + ((n - 1) * w) as u64, w, s)?;
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
        self.forward(&dr.p, m, (pp, pp), 1)?;
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
                t.push(HeadPass { pos, tokens: next.to_vec(), streams, routes, k, logits, draft });
            }
        }
        Ok(draft)
    }

    /// The draft head's passes run between `trace_head(true)` and `trace_head(false)`, which returns them (`HeadPass`); none where the engine has no draft head.
    pub fn trace_head(&self, on: bool) -> Vec<HeadPass> {
        let Some(hs) = self.draft.as_ref().and_then(|d| d.head.as_ref()) else { return Vec::new() };
        std::mem::replace(&mut *hs.trace.borrow_mut(), on.then(Vec::new)).unwrap_or_default()
    }

    /// Whether the engine drafts with a head of the model itself (`lower::qwen4x_head`).
    pub fn drafts_with_head(&self) -> bool {
        self.draft.as_ref().is_some_and(|d| d.head.is_some())
    }

    /// A draft head's state forgotten (a new sequence: `reset`).
    pub(super) fn head_reset(&self) {
        if let Some(hs) = self.draft.as_ref().and_then(|d| d.head.as_ref()) {
            (hs.pending.set(None), hs.last.set(None));
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
            // SAFETY: the head's last pass left its rows' streams in its S; its Hid takes the last of them.
            unsafe {
                copy_d2d_async(dr.p.slot(Slot::Hid), dr.p.slot(Slot::S) + (at.row * w) as u64, w, s)?;
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
        let (rows, words) = self.head_out(n, sampler.is_some())?;
        let big: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let (out, kept) = super::accept(&seq, &big, &words, self.big.d.vocab, sampler);
        self.rewind(kept + 1)?;
        let next: Vec<u32> = seq[1..=kept].iter().copied().chain(out.last().copied()).collect();
        self.head_again(dr, n0, kept, &next)?;
        self.prof_pass(n, 0)?;
        Ok(out)
    }

    /// After a round's verify pass at `n0` and `rewind` to its `kept` drafts and the row before them: the head again on those rows' streams, each with the token
    /// after it (`next`), its rows there in place of its drafts', its last row's draft the next round's first.
    fn head_again(&self, dr: &Draft, n0: usize, kept: usize, next: &[u32]) -> Result<()> {
        let (s, w) = (self.dev.stream, dr.p.d.width(Slot::Hid) * 2);
        // SAFETY: the model's S holds the verify pass's rows (`rewind` writes other slots); the copy is queued before the head's pass.
        unsafe {
            copy_d2d_async(dr.p.slot(Slot::Hid), self.big.slot(Slot::S), (kept + 1) * w, s)?;
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

/// `glyd-engine draftcheck CASE [--level L] [--k K] [--tokens N] [--split CUT,.. --devices GPU,..]`: the family's draft head on the executor, on the random case
/// with its MTP layer (`logits::qwen4x_mtp_random`, CASE `qwen4x`: the `qwen4x` case's model to the bit and its head), for a short prompt and one past two
/// chunks: (1) the drafted greedy generation against the model alone, N tokens: every token and every one of its rows of logits the same bits (the round's
/// verify rows are the steps'), the drafts and how many were kept said; (2) the head held to the CPU reference of the kinds (`RefMtp::step_streams`) on
/// the streams and tokens it read: each of its passes' last row against the reference's on the head's own expert choices, at the bound the family is
/// held to (`tests/logits/qwen4x.json.gz`'s), the figures on the reference's own choices printed beside them. With `--split`, (1) again with the model's
/// layers in stages (src/dynamic/split.rs; GPU 0 for each where `--devices` is not given), its head on the last: the tokens and rows the model's alone,
/// and each round's tokens the one-GPU drafted engine's (the head drafts there as on one GPU). An error where any fails.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    let case = a.need(1, "CASE")?;
    if case != "qwen4x" {
        return err(format!("draftcheck {case}: the draft head's case is qwen4x (the family's random model with its MTP layer)"));
    }
    let level = a.flag("level").unwrap_or("penguin");
    let (k, tokens): (usize, usize) = (a.num("k", 3usize)?, a.num("tokens", 24usize)?);
    let split = super::split::split_args(a)?;
    let dir = std::env::temp_dir().join(format!("glyd-draftcheck-{}", std::process::id()));
    crate::logits::qwen4x_mtp_random(&dir)?;
    let r = check_on(&dir, level, k, tokens, &split);
    std::fs::remove_dir_all(&dir).ok();
    let (bits_ok, held_ok) = r?;
    if !(bits_ok && held_ok) {
        return err(format!(
            "draftcheck {case} at {level}: {}{}",
            if bits_ok { "" } else { "the drafted tokens are not the model's alone; " },
            if held_ok { "" } else { "the head is past its bound" }
        ));
    }
    println!("PASS draftcheck {case} at {level}");
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

/// `check`'s parts on the model in `dir`: whether the drafted generations are the model's alone, bit for bit (and across the stages of `split` where it cuts
/// any: the one GPU's rounds too), and whether the head is within its bound.
#[cfg(feature = "dev")]
fn check_on(dir: &std::path::Path, level: &str, k: usize, tokens: usize, split: &(Vec<usize>, Vec<i32>)) -> Result<(bool, bool)> {
    use crate::logits::{compare, level_layout, Bound, Fixture, HeadRow, Row};
    let layout = level_layout(level)?;
    let experts = super::qwen4x::experts_at(layout);
    let o = super::Opts { ctx: 1024, chunk: 64, graph: true, device: 0, layout };
    let fx = Fixture::read(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/logits/qwen4x.json.gz"))?;
    let bound = fx.bounds.first().map(|b| b.bound.clone()).ok_or_else(|| Error("the qwen4x fixture has no bound".into()))?;
    let (vocab, mut plain, mut drafted) = (2048, super::qwen4x::load_level(dir, &o, experts)?, super::qwen4x::load_drafting(dir, &o, experts, k)?);
    let mut staged = match split {
        (cuts, _) if cuts.is_empty() => None,
        (cuts, devices) => match super::qwen4x::open_split(dir, &o, experts, cuts, devices, Some(k))? {
            super::split::Run::Split(s) if s.drafting() => Some(s),
            _ => return err(format!("draftcheck --split {cuts:?}: the stages do not draft")),
        },
    };
    let (short, _, long) = crate::logits::standard_tokens(vocab);
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
        println!(
            "{} drafted {what}, {tokens} tokens: {} the model's alone, rows of logits the same bits {same} of {tokens}{}; {} rounds of {k} drafts, {kept} drafts kept ({:.2} tokens a round)",
            if ok { "PASS" } else { "FAIL" },
            tokens_said(&got),
            first.map_or(String::new(), |i| format!(" (the first token apart: {i}, {} against {})", got[i], want[i])),
            own.len(),
            tokens as f64 / own.len().max(1) as f64
        );
        let kept_by = |rounds: &[Vec<u32>]| {
            let mut by = vec![0usize; k + 1];
            rounds.iter().for_each(|r| by[(r.len() - 1).min(k)] += 1);
            by
        };
        let (got, rows, rounds) = drafted_run(&mut drafted, prompt, tokens, &given)?;
        let ((ok, same, _), by) = (judge(&got, &rows), kept_by(&rounds));
        let ok = ok && by.iter().all(|&c| c > 0);
        bits_ok &= ok;
        println!(
            "{} drafts given, {what}, {tokens} tokens: {} the model's alone, rows of logits the same bits {same} of {tokens}; {} rounds, of each count of drafts kept 0 to {k}: {by:?}",
            if ok { "PASS" } else { "FAIL" },
            tokens_said(&got),
            rounds.len()
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
    // the head against the reference, a row of each pass on the streams and the token it read
    let passes = drafted.trace_head(false);
    let w = passes.first().map_or(0, |p| p.streams.len() / p.tokens.len().max(1));
    let widen = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect() };
    let mut rows = Vec::new();
    let mut last_of = Vec::new();
    for p in &passes {
        for (i, &t) in p.tokens.iter().enumerate() {
            let take = (p.k > 0).then(|| p.routes[i * p.k..(i + 1) * p.k].iter().map(|&e| e as usize).collect());
            rows.push(HeadRow { pos: (p.pos + i) as u32, token: t, streams: widen(&p.streams[i * w..(i + 1) * w]), take });
        }
        last_of.push(rows.len() - 1);
    }
    let engine: Vec<Vec<f32>> = passes.iter().map(|p| widen(&p.logits)).collect();
    // (a prompt run after the other begins a new sequence: the reference's head too)
    let mut on_choices = Vec::new();
    let mut own = Vec::new();
    let mut apart = 0;
    for part in split_sequences(&rows, &last_of) {
        let (r, _, routes) = crate::logits::ref_head(dir, &part.rows, |_| {})?;
        let free: Vec<HeadRow> = part.rows.iter().map(|x| HeadRow { take: None, ..x.clone() }).collect();
        let (f, _, froutes) = crate::logits::ref_head(dir, &free, |_| {})?;
        let set = |v: &[usize]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v
        };
        apart += routes.iter().zip(&froutes).filter(|(a, b)| set(&a.1) != set(&b.0)).count();
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
        "     on the reference's own choices: KL mean {:.2e} max {:.2e}, best token {}/{}; its experts apart from the head's at {apart} of {} rows",
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
        // (the round's tokens: the last one, then the head's drafts in the order it made them)
        let heads = drafted.trace_head(false);
        let k = pass.first().map_or(0, |(_, sl)| sl[0].len() / plain.big.d.width(Slot::H).max(1));
        let seq: Vec<u32> = std::iter::once(t0).chain(heads.iter().map(|h| h.draft)).take(k).collect();
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
