//! Several sequences at once on the paged cache: sequences wait, take pages (a prompt that begins as a finished one did finds its pages in the cache), run their prompt a chunk at a time, and decode together, one
//! step for all of them. Where the pages run out, the newest sequence gives its pages back and goes to the head of the queue with what it has made; it is run again from its tokens (recomputed) and its stream goes on.
//! A slot step that can swap (`ModelStep::swap_out`, the description executor's) copies such a sequence's blocks and recurrent states to the host instead, where that costs less than its recompute
//! (`Sched::swap_pays`), and copies them back when there is room again: it goes on from where it was, with the same cache bit for bit (R4d, docs/design/runtime.md section 5, "Preemption").
//!
//! The cache's extra memory is shared and finite. After every step the backend says which blocks the step could not store exactly (they are named lost) and which it kept in the scarce second extra memory (heavy). A sequence
//! that holds a lost block has the step's output taken back, never used or sent: it gives its pages back and waits until the sequences that were running with it have finished, then runs again from its tokens; one that is lost
//! alone twice, or six times, ends with an error in words. Lost and heavy blocks are never kept in the prefix cache.
//!
//! The scheduler is the same for every model: it drives a `ModelStep` (docs/design/runtime.md, section 2). A step keeps its cache one of two ways: pages shared by every sequence (with the prefix cache and
//! preemption above: the scheduler's own tests' steps, since the dense Qwen3 engine went onto the executor), or a room of the whole context for each of a few sequences (`ModelStep::slots`: the description executor, `src/dynamic`), where a sequence holds one slot from
//! its admission to its end, nothing is shared and nothing is preempted for room.

use crate::ffi::{Error, Kind, Result};
use crate::kv::{block_hash, block_names, block_root, Lost, Pager, ALONE_MESSAGE, PAGE};
use crate::runtime::context::Context;
use crate::sampler::{logprobs_of, Lp, Sampler};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// Rows a step on pages takes at most, and the steps counted by their rows (`Stats::by_rows`: 0 to 7, and 8 or more in the last).
pub const MAX_ROWS: usize = 8;

/// The most rows of a product the library's own route takes whatever the GPU (`decoded_way`: past it a product may take the decoded matrix's way): a prompt chunk of at most this many
/// tokens is run with others in one pass (a longer one is its own: it is compute-bound, and the weights are read once for it).
pub const FUSED_ROWS: usize = 64;

/// What a request is sent when its sequence runs: each token as it is chosen, and last how the sequence ended. Where the request asks for log-probabilities, an `Lp` of a token comes before the token
/// (the end-of-sequence token's, which is no token of the reply, before the end).
pub enum Ev {
    Token(u32),
    Lp(Lp),
    End(std::result::Result<Finish, crate::ffi::Error>),
}

/// A sequence to run: its prompt, what the reply is asked to be, where its tokens go, and a flag that ends it (the request has all it wants, or has gone).
pub struct Submit {
    pub prompt: Vec<u32>,
    pub reply: Reply,
    pub out: Sender<Ev>,
    pub abort: Arc<AtomicBool>,
    /// Counts the times the sequence is sent back to wait and run again (for room, or for a block the cache lost); the caller may read it.
    pub resumed: Arc<AtomicU32>,
    /// The prompt tokens found in the cache when the sequence was first let in (a kept sequence's, or pages of an earlier prompt), which were not run again: set before
    /// the sequence's first event; the caller may read it (a request's usage).
    pub cached: Arc<AtomicU32>,
    /// A request with a cache marker (`pin`): the prompt tokens it left findable in the cache that it did not find there (Anthropic's `cache_creation_input_tokens`), set
    /// once its prompt has run; 0 for any other request, and on a step that does not find blocks by name.
    pub created: Arc<AtomicU32>,
    /// The request's cache scope (its `cache_salt`'s digest; 0: none): it finds in the cache only what requests of its scope left, and what it leaves is found by them alone.
    /// A server several callers share gives each its own, so that no caller learns from what is found, or how soon its first token comes, how another's prompt begins.
    pub salt: u64,
    /// Seconds what this request leaves in a slot is held before other kept sequences when a slot or room is wanted (Anthropic's `cache_control`: 300, or 3,600 with
    /// `ttl: "1h"`; 0: no longer than any other). A preference among kept sequences, never a refusal: a request that needs the slot has it.
    pub pin: u32,
    /// The prompt's images in order (their places in it their tokens), for an engine that reads images (src/vision.rs); none for the rest.
    pub images: Vec<Arc<crate::vision::Image>>,
}

/// One sequence's prompt chunk in a batch of them: `tokens` at positions `start` on, into `pages` (a slot step: `[slot]`); where `last`, the token after the chunk is wanted.
pub struct PrefillJob<'a> {
    pub pages: &'a [i32],
    pub start: usize,
    pub tokens: &'a [u32],
    pub last: bool,
}

/// One sequence's row of a decode step: the token whose keys and values are written at `pos` (the sequence's next position) and the pages of the sequence (at least pos / 16 + 1; a slot step: `[slot]`).
pub struct Row<'a> {
    pub token: u32,
    pub pos: usize,
    pub pages: &'a [i32],
    /// Some(r): the token is the one the step launched before this one chooses in its head row r (its largest logit), which the host does not have yet: the step
    /// reads it on the GPU (`ModelStep::launch`, a step that `overlaps`), and `token` is not used.
    pub after: Option<usize>,
}

impl<'a> Row<'a> {
    /// A row whose token the host has.
    pub fn new(token: u32, pos: usize, pages: &'a [i32]) -> Row<'a> {
        Row { token, pos, pages, after: None }
    }
}

/// The samples of a step: each decode row's, then each prompt chunk's that is `last` (None for the others).
pub type Samples = (Vec<Sample>, Vec<Option<Sample>>);

/// A step `ModelStep::launch` queued: its samples where it ran to its end at its launch, else the step's own note of where they will be (`finish` reads them).
pub struct Launched {
    pub done: Option<Samples>,
    pub tag: usize,
}

/// A sampled token with its logit and, where asked, its logprob.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub token: u32,
    pub logit: f32,
    pub logprob: Option<f32>,
}

/// What a reply is asked to be: sampling, length, and whether the end-of-sequence token ends it.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    pub params: crate::sampler::Params,
    pub max_new: usize,
    pub ignore_eos: bool,
    /// Log-probabilities asked for: the number of likeliest tokens with each (0: the chosen token's alone).
    pub logprobs: Option<usize>,
    /// Tokens besides the end-of-sequence tokens that end the reply (vLLM's `stop_token_ids`); not part of its text.
    pub stop_ids: Vec<u32>,
}

/// How a reply ended: the model ended it, `max_new` or the context was reached, or the caller stopped it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Stopped,
}

/// What a scheduler runs on: the model's steps on its cache (docs/design/runtime.md: `ModelStep`).
pub trait ModelStep {
    /// The longest sequence, in tokens.
    fn window(&self) -> usize;
    /// The most prompt tokens a chunk takes.
    fn chunk(&self) -> usize;
    /// Pages in the cache (a slot step: unused).
    fn pages(&self) -> usize;
    /// Some(k) where the cache is a room of `window()` tokens for each of k sequences, not shared pages: a row's and a chunk's `pages` is then `[slot]`, and `release(&[slot])` makes the slot empty
    /// (position 0, recurrent states zeroed). None: shared pages.
    fn slots(&self) -> Option<usize> {
        None
    }
    /// A slot step whose rooms take memory as they fill (pages behind them): room for the sequence in `slot` to hold its first `upto` tokens, taken now; false where the memory has none
    /// (nothing taken). The scheduler then gives the newest sequence's room back and asks again. A room of the whole window from the start: always true.
    fn grow(&mut self, _slot: i32, _upto: usize) -> Result<bool> {
        Ok(true)
    }
    /// The bytes `swap_out` would move of the sequence in `slot` (its blocks and recurrent states; the same again back), and the link's rate in bytes a second (measured on the swaps so far). None: the
    /// step cannot swap it (no swap area, nothing held), and a sequence sent back for room is run again from its tokens.
    fn swap_cost(&self, _slot: i32) -> Option<(usize, f64)> {
        None
    }
    /// The sequence in `slot` copied to the host's swap area on a copy stream and its slot emptied as `release` empties it (the steps after wait for the copy before they write what it read): a ticket
    /// for `swap_in`. None where the area has no room for it (nothing changes).
    fn swap_out(&mut self, _slot: i32) -> Result<Option<u64>> {
        Ok(None)
    }
    /// The sequence of `ticket` copied back into the empty slot `slot` on the copy stream, beside the steps that run meanwhile, with room for its first `upto` tokens: the tokens it holds. None where the
    /// memory has no room for it now (nothing taken, the ticket kept). It steps once `swap_ready` says its copy is done.
    fn swap_in(&mut self, _slot: i32, _ticket: u64, _upto: usize) -> Result<Option<usize>> {
        Ok(None)
    }
    /// Whether the copy back into `slot` is done; with `wait`, once it is.
    fn swap_ready(&mut self, _slot: i32, _wait: bool) -> Result<bool> {
        Ok(true)
    }
    /// The host's copy of `ticket` let go: its request ended while it was swapped, or it runs again from its tokens.
    fn swap_drop(&mut self, _ticket: u64) {}
    /// A slot step that can keep a finished sequence in its slot for a request that begins with its tokens (`resume`): a conversation's next turn, which sends the turns before it again.
    /// False: a finished sequence's slot is emptied at once.
    fn keeps(&self) -> bool {
        false
    }
    /// The finished sequence in `slot` cut back to go on within its first `upto` tokens, as if it had run no further: the tokens it still holds. That is `upto` where it can be cut
    /// anywhere; a model with a recurrent state (the state after the sequence's last token and no other) goes on from its end or from its mark (`mark`), the nearest before `upto`;
    /// 0, the slot as it was, where it cannot go on from any place (no mark within, or blocks a cut needs were given back).
    fn resume(&mut self, _slot: i32, _upto: usize) -> Result<usize> {
        Ok(0)
    }
    /// The first `upto` tokens of the sequence in slot `from` (running or kept) made the start of the empty slot `to` as well, without running them again: the tokens `to`
    /// then holds (0 where nothing can be shared; fewer than `upto` where it is shared by whole blocks, or from a recurrent model's mark). The two go on apart: neither
    /// writes what the other reads.
    fn share(&mut self, _from: i32, _to: i32, _upto: usize) -> Result<usize> {
        Ok(0)
    }
    /// Whether the step finds a sequence's full blocks by their names (`name`, `adopt`), so that a start of any length that it holds is had without running it, whoever ran
    /// it, and a request is worth holding back while a running sequence still runs the start the two have in common. False: nothing is found by name; a start is shared
    /// from a sequence in a slot (`share`), if at all.
    fn shares_starts(&self) -> bool {
        false
    }
    /// A step that finds blocks by name and keeps a sequence's recurrent states only at boundaries, every so many tokens: that spacing (a multiple of PAGE), at which a
    /// prompt's pieces end, so that the states kept there are the states after that token. 0: no boundaries.
    fn snap_every(&self) -> usize {
        0
    }
    /// The sequence in `slot` holds the full blocks `first`, `first + 1`, ... (PAGE tokens each), whose names are `keys` (`kv::block_hash`: of a block's tokens and all
    /// before them, in the request's scope). A step that finds blocks by name finds these by them from now on, while it holds them: a sequence need not.
    fn name(&mut self, _slot: i32, _first: usize, _keys: &[u128]) -> Result<()> {
        Ok(())
    }
    /// The empty slot `slot` given the longest start of a sequence whose full blocks are named `keys`, in order, that the step holds by name and can go on from: the
    /// tokens the slot then holds (whole blocks; 0: none).
    fn adopt(&mut self, _slot: i32, _keys: &[u128]) -> Result<usize> {
        Ok(0)
    }
    /// The tokens `adopt` would find for `keys` now.
    fn held_by_name(&self, _keys: &[u128]) -> usize {
        0
    }
    /// The names of blocks the step no longer finds by them, since the last call (each taken for another use): the cache's removed blocks (src/kv_events.rs).
    fn take_removed(&mut self) -> Vec<u128> {
        Vec::new()
    }
    /// The sequence in `slot` holds `n` tokens and is at a place a later request may go on from (its prompt's end: a reply asked for again, or a next turn whose template leaves out
    /// part of this reply, begins with these tokens and no more). A step that cannot cut a sequence back anywhere keeps what `resume` needs to go on from here.
    fn mark(&mut self, _slot: i32, _n: usize) -> Result<()> {
        Ok(())
    }
    fn eos(&self) -> Vec<i64>;
    /// `tokens` at positions `start` on into `pages`; with `last`, the sample after the final token.
    fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>>;
    /// The most sequences whose prompt chunks `prefill_many` runs in one pass (1: one at a time, as `prefill`).
    fn prefill_rows(&self) -> usize {
        1
    }
    /// The most tokens a chunk has to run in a pass with others (each of the others' too; a longer one is run alone).
    fn prefill_together(&self) -> usize {
        usize::MAX
    }
    /// The chunks of several sequences' prompts in one pass: the sample after each that is `last`, in order (the logits and candidates of the j-th of them are row j, as a decode step's).
    fn prefill_many(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
        jobs.iter().map(|j| self.prefill(j.pages, j.start, j.tokens, j.last)).collect()
    }
    /// Whether `round` runs a round's decode rows and prompt chunks in one pass, each weight read once for both (docs/design/runtime.md section 5, "A round is one step"). False: a round is a
    /// prompt pass and then a decode step (`prefill_many`, `decode`).
    fn one_step(&self) -> bool {
        false
    }
    /// The most prompt tokens a step takes beside `rows` decode rows (`one_step`): what the step's workspace holds after them, or fewer where the rows are not to wait long for the prompts.
    fn prompt_budget(&self, rows: usize) -> usize {
        self.chunk().saturating_sub(rows)
    }
    /// A round as one step over a flat batch (`one_step`): the decode rows, then the prompt chunks; the sample after each row, and after each chunk that is `last`. The logits and candidates of the
    /// decode rows are rows 0 on, then those of the chunks that are `last`, in order (the masks given before it are in that order too).
    fn round(&mut self, _rows: &[Row], _jobs: &[PrefillJob]) -> Result<Samples> {
        Err(crate::ffi::Error("this model's rounds are a prompt pass and a decode step, not one step".into()))
    }
    /// Whether `launch` returns with the step queued on the GPU and not run, so that the host's work between steps (the scheduler's, the sampler's, the server's) runs while the
    /// GPU runs the step, and the next step may be launched before this one's `finish`, its rows reading this step's greedy tokens on the GPU (`Row::after`). SGLang's overlap
    /// loop and vLLM's async scheduling (KB f:overlap:serving:3). A sequence's blocks are named (`name`) and its prompt's end marked (`mark`) when the next step is launched
    /// (before its work), for the step in flight's tokens, never after a later step was launched over it. False: `launch` runs `round` to its end.
    fn overlaps(&self) -> bool {
        false
    }
    /// A step queued: `round`'s rows and chunks (a step that is not `one_step`: rows alone, a decode step, or chunks alone, a prompt pass), whose samples `finish` gives.
    /// Between the two the scheduler calls nothing that reads this step's rows (`top`, `logits`, `gather`) once a later step is launched, and nothing that changes a sequence
    /// of the step but `grow` (room for its next tokens), `release` (it ended) and `name` (its blocks named for the cache). By default the step runs to its end here.
    fn launch(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<Launched> {
        let done = match (self.one_step(), rows.is_empty(), jobs.is_empty()) {
            (true, _, _) => self.round(rows, jobs)?,
            (false, _, true) => (self.decode(rows)?, Vec::new()),
            (false, true, false) => (Vec::new(), self.prefill_many(jobs)?),
            (false, false, false) => return Err(crate::ffi::Error("a step of rows and prompt chunks, on a model whose rounds are two passes".into())),
        };
        Ok(Launched { done: Some(done), tag: 0 })
    }
    /// Whether a row of the next step `launch` queues draws its token from its candidates or reads its logits (a sequence that is not greedy, or asks for
    /// log-probabilities): a step that makes the candidates on the GPU (`top`) then queues them with the step's work and their copy with its samples, and `top` waits
    /// for nothing.
    fn draws(&mut self, _on: bool) {}
    /// The samples of a step `launch` queued, once its work has run (the wait is for that step, not for the GPU's whole queue).
    fn finish(&mut self, l: Launched) -> Result<Samples> {
        l.done.ok_or_else(|| crate::ffi::Error("a step launched to run at once has no samples".into()))
    }
    /// The steps launched and not finished given up (the scheduler ends every sequence: `Sched::fail`): their samples are not asked for.
    fn abandon(&mut self) {}
    /// A drafting step's decode of the one sequence that decodes (a draft model proposes tokens and the model checks them in one pass: src/dynamic/mod.rs `spec_round`): the
    /// tokens after the row's, 1 to the draft's length and one more, as steps one at a time would choose them (drawn by `sampler` where given, else the largest logits); the
    /// cache then holds the row's token and each of them but the last. None: the step does not draft (its decode is one token a row).
    fn draft(&mut self, _row: &Row, _sampler: Option<&mut Sampler>) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
    /// The grammar masks of the rows the next `decode`, `prefill_many` or `round` draws samples from (one for each, in order; None: no grammar), for a backend that makes the sampler's candidates on the GPU to put on the row before its top-k.
    fn masks(&mut self, _masks: Vec<Option<crate::constrain::MaskHandle>>) {}
    /// One step of the rows: the sample after each.
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>>;
    /// What the last step's mixture-of-experts layers did with the expert cache, layer by layer (src/offload.rs `LayerStats`: hits, copies, the host's pairs, bytes, waits); none where no layer's experts
    /// are in a cache.
    fn expert_stats(&self) -> Vec<crate::offload::LayerStats> {
        Vec::new()
    }
    /// The logits behind the last sample of row `row`.
    fn logits(&mut self, row: usize) -> Result<Vec<f32>>;
    /// Pages no sequence holds and nothing names, cleared.
    fn release(&mut self, pages: &[i32]) -> Result<()>;
    /// Full pages about to be given back, checked against what was written (where the backend does that).
    fn verify(&mut self, _pages: &[i32]) -> Result<()> {
        Ok(())
    }
    /// The blocks the steps since the last call could not store exactly (lost) or kept in the scarce second extra memory (heavy).
    fn lost(&mut self) -> Lost {
        Lost::default()
    }
    /// The sampler's candidates for row `row` of the last step: the largest logits of its row, largest first (None: the backend has none, the logits are read).
    fn top(&mut self, _row: usize) -> Option<Vec<(u32, f32)>> {
        None
    }
    /// Row `row`'s logits at `tokens` (those in the vocabulary, with them): what a draw from the candidates needs where penalties, a bias or suppressed tokens
    /// change those tokens' logits (`Sampler::sample_adjusted`). By default from the whole row; a GPU backend reads those tokens alone.
    fn gather(&mut self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
        let l = self.logits(row)?;
        Ok(tokens.iter().filter_map(|&t| l.get(t as usize).map(|&x| (t, x))).collect())
    }
}

/// What the scheduler has done (read by the server's /stats): prompt tokens run or found in the cache, sequences sent back to wait for pages, steps and the rows in them.
#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub prompt: u64,
    pub hit: u64,
    pub preempted: u64,
    pub steps: u64,
    pub rows: u64,
    /// Time (ns) in the model's decode steps, in its prompt chunks, and in all the scheduler's rounds.
    pub decode_ns: u64,
    pub prefill_ns: u64,
    pub tick_ns: u64,
    /// Time (ns) clearing the pages of sequences that ended.
    pub release_ns: u64,
    /// Time (ns) in taking sequences in, in the work after each decode step on each row, and in the pages before it.
    pub admit_ns: u64,
    pub after_ns: u64,
    pub before_ns: u64,
    /// Blocks named lost (not stored exactly) and blocks kept in the second extra memory, sequences whose step was taken back, and sequences ended with an error for it.
    pub lost_blocks: u64,
    pub heavy_blocks: u64,
    pub taken_back: u64,
    pub ended_lost: u64,
    /// Of those ended, the ones lost alone with nothing kept for later prompts to give up, and the ones that had been taken back MAX_TRIES times.
    pub ended_alone: u64,
    pub ended_tries: u64,
    /// Steps by the number of rows they had (index 0 unused: 1 to 8), and the rounds that ran a prompt chunk.
    /// Decode steps by their rows: 0 to 7, and 8 or more in the last.
    pub by_rows: [u64; MAX_ROWS + 1],
    pub prefill_rounds: u64,
    /// Steps that held both rows that decode and a prompt's chunk (a round that is one step, `ModelStep::one_step`).
    pub both: u64,
    /// Of the sequences sent back for room, those swapped to the host (`ModelStep::swap_out`) and not run again from their tokens.
    pub swapped: u64,
    /// The prompt tokens the passes that ran prompts ran, and their time (ns): the rate a recompute is priced at (`Sched::swap_pays`).
    pub pass_tokens: u64,
    pub pass_ns: u64,
}

/// How a sequence sent back for room gives its cache up (docs/design/runtime.md section 5, "Preemption").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapMode {
    /// Its blocks are let go and its tokens run again (recompute).
    Never,
    /// Swapped to the host where its bytes over the link, out and back, take less time than its tokens' recompute (the default).
    ByCost,
    /// Swapped wherever the step can (a check of the swap itself).
    Always,
}

/// The scheduler's figures as the last round left them (one scheduler runs in a process), for the server's /stats.
/// How many figures a scheduler leaves each round (`Sched::gauges`): the length of a context's `sched`.
pub const FIGURES: usize = 30;

/// The figures as JSON: prompt tokens run or found in the cache, those found, sequences sent back to wait for pages, steps, rows in them, sequences running and waiting.
pub fn stats_json() -> String {
    stats_json_of(Context::process())
}

/// `stats_json` of the scheduler of the context `rt`.
pub fn stats_json_of(rt: &Context) -> String {
    let v: Vec<u64> = rt.sched.iter().map(|a| a.load(Relaxed)).collect();
    // (what was found in the cache is told only where the server tells its requests: `Context::tell_cached`. A caller who could read it here before and after a request of
    // its own would learn what that request found, which its usage does not say; null otherwise)
    let (hit, rate) = match rt.tell_cached.load(Relaxed) {
        true => (v[1].to_string(), format!("{:.2}", if v[0] > 0 { 100.0 * v[1] as f64 / v[0] as f64 } else { 0.0 })),
        false => ("null".to_string(), "null".to_string()),
    };
    // (the expert cache's layers since the process started, where a model has experts in one: src/offload.rs `stats_field`)
    let experts = crate::offload::stats_field();
    // a development build's figures are all of them; a release build's are the counts a user reads
    if cfg!(feature = "dev") {
        format!(
            "{{\"prompt_tokens\":{},\"prefix_hit_tokens\":{hit},\"prefix_hit_rate_percent\":{rate},\"preemptions\":{},\"swapped\":{},\"steps\":{},\"rows\":{},\"running\":{},\"waiting\":{},\"decode_ms\":{},\"prefill_ms\":{},\"rounds_ms\":{},\"release_ms\":{},\"admit_ms\":{},\"after_ms\":{},\"before_ms\":{},\"lost_blocks\":{},\"heavy_blocks\":{},\"taken_back\":{},\"ended_for_loss\":{},\"steps_by_rows\":[{}],\"prefill_rounds\":{}{experts}}}",
            v[0], v[2], v[29], v[3], v[4], v[5], v[6], v[7] / 1_000_000, v[8] / 1_000_000, v[9] / 1_000_000, v[10] / 1_000_000, v[11] / 1_000_000, v[12] / 1_000_000, v[13] / 1_000_000, v[14], v[15], v[16], v[17], v[18..27].iter().map(u64::to_string).collect::<Vec<_>>().join(","), v[27]
        )
    } else {
        format!(
            "{{\"prompt_tokens\":{},\"prefix_hit_tokens\":{hit},\"prefix_hit_rate_percent\":{rate},\"preemptions\":{},\"swapped\":{},\"steps\":{},\"rows\":{},\"running\":{},\"waiting\":{}{experts}}}",
            v[0], v[2], v[29], v[3], v[4], v[5], v[6]
        )
    }
}

/// A step behind a reference (an engine handed over as `conv::Gen::as_step` gives it).
impl ModelStep for &mut dyn ModelStep {
    fn window(&self) -> usize {
        (**self).window()
    }
    fn chunk(&self) -> usize {
        (**self).chunk()
    }
    fn pages(&self) -> usize {
        (**self).pages()
    }
    fn slots(&self) -> Option<usize> {
        (**self).slots()
    }
    fn grow(&mut self, slot: i32, upto: usize) -> Result<bool> {
        (**self).grow(slot, upto)
    }
    fn swap_cost(&self, slot: i32) -> Option<(usize, f64)> {
        (**self).swap_cost(slot)
    }
    fn swap_out(&mut self, slot: i32) -> Result<Option<u64>> {
        (**self).swap_out(slot)
    }
    fn swap_in(&mut self, slot: i32, ticket: u64, upto: usize) -> Result<Option<usize>> {
        (**self).swap_in(slot, ticket, upto)
    }
    fn swap_ready(&mut self, slot: i32, wait: bool) -> Result<bool> {
        (**self).swap_ready(slot, wait)
    }
    fn swap_drop(&mut self, ticket: u64) {
        (**self).swap_drop(ticket)
    }
    fn keeps(&self) -> bool {
        (**self).keeps()
    }
    fn resume(&mut self, slot: i32, upto: usize) -> Result<usize> {
        (**self).resume(slot, upto)
    }
    fn mark(&mut self, slot: i32, n: usize) -> Result<()> {
        (**self).mark(slot, n)
    }
    fn share(&mut self, from: i32, to: i32, upto: usize) -> Result<usize> {
        (**self).share(from, to, upto)
    }
    fn shares_starts(&self) -> bool {
        (**self).shares_starts()
    }
    fn snap_every(&self) -> usize {
        (**self).snap_every()
    }
    fn name(&mut self, slot: i32, first: usize, keys: &[u128]) -> Result<()> {
        (**self).name(slot, first, keys)
    }
    fn adopt(&mut self, slot: i32, keys: &[u128]) -> Result<usize> {
        (**self).adopt(slot, keys)
    }
    fn held_by_name(&self, keys: &[u128]) -> usize {
        (**self).held_by_name(keys)
    }
    fn take_removed(&mut self) -> Vec<u128> {
        (**self).take_removed()
    }
    fn eos(&self) -> Vec<i64> {
        (**self).eos()
    }
    fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        (**self).prefill(pages, start, tokens, last)
    }
    fn prefill_rows(&self) -> usize {
        (**self).prefill_rows()
    }
    fn prefill_together(&self) -> usize {
        (**self).prefill_together()
    }
    fn prefill_many(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
        (**self).prefill_many(jobs)
    }
    fn one_step(&self) -> bool {
        (**self).one_step()
    }
    fn prompt_budget(&self, rows: usize) -> usize {
        (**self).prompt_budget(rows)
    }
    fn round(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<Samples> {
        (**self).round(rows, jobs)
    }
    fn overlaps(&self) -> bool {
        (**self).overlaps()
    }
    fn launch(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<Launched> {
        (**self).launch(rows, jobs)
    }
    fn finish(&mut self, l: Launched) -> Result<Samples> {
        (**self).finish(l)
    }
    fn abandon(&mut self) {
        (**self).abandon()
    }
    fn draws(&mut self, on: bool) {
        (**self).draws(on)
    }
    fn draft(&mut self, row: &Row, sampler: Option<&mut Sampler>) -> Result<Option<Vec<u32>>> {
        (**self).draft(row, sampler)
    }
    fn masks(&mut self, masks: Vec<Option<crate::constrain::MaskHandle>>) {
        (**self).masks(masks)
    }
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        (**self).decode(rows)
    }
    fn expert_stats(&self) -> Vec<crate::offload::LayerStats> {
        (**self).expert_stats()
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        (**self).logits(row)
    }
    fn release(&mut self, pages: &[i32]) -> Result<()> {
        (**self).release(pages)
    }
    fn verify(&mut self, pages: &[i32]) -> Result<()> {
        (**self).verify(pages)
    }
    fn lost(&mut self) -> Lost {
        (**self).lost()
    }
    fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        (**self).top(row)
    }
    fn gather(&mut self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
        (**self).gather(row, tokens)
    }
}

struct Seq {
    id: u64,
    out: Sender<Ev>,
    abort: Arc<AtomicBool>,
    resumed: Arc<AtomicU32>,
    cached: Arc<AtomicU32>,
    created: Arc<AtomicU32>,
    salt: u64,
    pin: u32,
    reply: Reply,
    greedy: bool,
    sampler: Sampler,
    /// The prompt and what has been chosen so far. While decoding, the last token is the one the next step takes (its keys and values are not in the cache yet).
    tokens: Vec<u32>,
    generated: usize,
    pages: Vec<i32>,
    /// Tokens whose keys and values are in the pages.
    done: usize,
    /// Full pages given to the cache (a slot step that finds blocks by name: full blocks told to it), and the name of the last (`kv::block_hash`; the scope's root with none).
    published: usize,
    chain: u128,
    /// The names of the full blocks its tokens but the last begin with, once a slot step that finds blocks by name has asked for them (worked out once while it waits).
    head_names: Option<Vec<u128>>,
    prefilling: bool,
    /// Times a step of this sequence was taken back for a lost block, and the sequences that must have ended before it runs again (they were running when it lost one).
    tries: u32,
    after: Vec<u64>,
    /// Hit twice while others ran: it runs alone next (nothing else is let in until it ends).
    solo: bool,
    /// Sent back for room with its cache copied to the host (`ModelStep::swap_out`): the ticket it is copied back by; its `done`, `published` and `prefilling` are as they were.
    swapped: Option<u64>,
    /// When it was sent to the scheduler and when its last token was written (the metrics' first-token and between-token times).
    began: std::time::Instant,
    last_token: Option<std::time::Instant>,
    /// The sequence's clock for its trace (queue, prompt pass, decode).
    trace: crate::otel::Clock,
}

/// A finished sequence kept in its slot (a slot step that `keeps`): the tokens whose keys and values, and states, the slot holds.
struct Kept {
    slot: i32,
    tokens: Vec<u32>,
    /// The scope of the request that left it (`Submit::salt`): a request of another scope does not go on in it.
    salt: u64,
    /// Held before the others until then (`Submit::pin`).
    pinned: Option<std::time::Instant>,
}

/// The fewest tokens of a kept sequence a request must begin with to go on in its slot. It must also begin with half of what the slot holds: a kept conversation is not cut down for
/// the few tokens every prompt of a template begins with, nor for a system prompt that another conversation shares, unless that is most of it.
const KEEP_LEAST: usize = 32;

/// The tokens `a` and `b` both begin with.
fn common(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl Seq {
    /// `n` of its prompt's tokens were in the cache as it is let in: told to the request, the first time (one sent back to wait finds its own tokens again, which it ran).
    fn found(&self, n: usize) {
        if self.resumed.load(Relaxed) == 0 {
            self.cached.store(n as u32, Relaxed);
        }
    }
}

/// The most sequences a slot step runs at once (the rows of a pass's head).
pub const MAX_USERS: usize = 64;

/// Times a sequence's step may be taken back for a lost block before it ends with an error (the extra memory is shared: one that is still lost after this many waits does not fit it).
pub const MAX_TRIES: u32 = 6;

/// The name of how a sequence ended, as the metrics count it.
fn finish_name(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::Length => "length",
        Finish::Stopped => "stopped",
    }
}

/// A step launched and not landed yet (`ModelStep::launch`): the sequences it runs, in its order, for `land` to take its samples to. A sequence that leaves the running ones
/// before then (it ended, or went back to wait) has its id here made 0, so that its row's output is not taken.
struct Flight {
    l: Launched,
    /// The decode rows' sequences (their samples are the head's rows 0 on).
    rows: Vec<u64>,
    /// The prompt chunks: sequence, tokens, and whether the chunk ends the prompt (its sample the head's next row after the rows' and the earlier last chunks').
    picks: Vec<(u64, usize, bool)>,
    used: usize,
    t: std::time::Instant,
    t0: f64,
    /// Whether every sequence that samples in it takes the step's own largest logit (greedy, no log-probabilities): the next step can then read the tokens on the GPU
    /// and be launched before this one lands.
    on_device: bool,
}

impl Flight {
    /// What sequence `id` has in the step: its tokens there, and the head row of its sample (a decode row's, or its prompt's last chunk's).
    fn of(&self, id: u64) -> (usize, Option<usize>) {
        if let Some(r) = self.rows.iter().position(|&x| x == id) {
            return (1, Some(r));
        }
        let mut row = self.rows.len();
        for &(x, n, last) in &self.picks {
            if x == id {
                return (n, last.then_some(row));
            }
            row += last as usize;
        }
        (0, None)
    }

    /// Sequence `id`'s output not to be taken.
    fn strike(&mut self, id: u64) {
        self.rows.iter_mut().chain(self.picks.iter_mut().map(|p| &mut p.0)).filter(|x| **x == id).for_each(|x| *x = 0);
    }
}

/// A round's step as `plan` chose it: the rows that decode (each sequence, and the head row of the step in flight whose sample is its token, where it is not had yet) and the prompts' chunks.
struct Plan {
    rows: Vec<(u64, Option<usize>)>,
    picks: Vec<(u64, usize)>,
    used: usize,
}

/// Whether a step that can be launched before the one before it lands (`ModelStep::overlaps`) is: GLYD_ENGINE_OVERLAP=0 lands each step before the next is launched (the round as it was).
pub(crate) fn overlap_asked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_OVERLAP").is_none_or(|v| v != "0"))
}

/// The most tokens a prompt pass has for the round's decode step to wait on the prompts that are coming in, and the most rounds in a row it does.
const SHORT_PASS: usize = 256;
const MAX_SKIPS: u32 = 4;
const MAX_LONG_SKIPS: u32 = 2;

pub struct Sched<B: ModelStep> {
    be: B,
    /// The runtime context its figures and its sequences' counts go to (the process's own unless `in_context` gave another).
    rt: Arc<Context>,
    /// The blocks named in this step, with their tokens, for the cache's events (`Context::kv_events`; empty where they are off).
    kv_out: Vec<crate::kv_events::Event>,
    pager: Pager,
    users: usize,
    waiting: VecDeque<Seq>,
    active: Vec<Seq>,
    /// Sequences being copied back from the host into their slots (`ModelStep::swap_in`): they join `active` when the copy is done, the steps going on meanwhile.
    loading: Vec<Seq>,
    /// How a sequence sent back for room gives its cache up.
    swap: SwapMode,
    next_id: u64,
    eos: Vec<i64>,
    pub stats: Stats,
    /// Events for the requests, sent in batches by a thread of their own.
    pending: Vec<(Sender<Ev>, Ev)>,
    emit: Sender<Vec<(Sender<Ev>, Ev)>>,
    /// GLYD_ENGINE_TRACE_SCHED: a line to stderr for every step taken back and every request ended for it (to see why), and for every prompt pass.
    #[cfg_attr(not(feature = "dev"), allow(dead_code))]
    trace: bool,
    /// The tokens of the last prompt pass, and the rounds in a row that gave their decode step up for the prompts that came in (see `tick`).
    pass_tokens: usize,
    skips: u32,
    /// A slot step's free slots, the lowest last (None: the cache is pages).
    free: Option<Vec<i32>>,
    /// A slot step's finished sequences kept in their slots, the oldest first: neither free nor running. A request that begins with one's tokens goes on in its slot; a request that
    /// finds no free slot, or no room, has the oldest emptied.
    kept: VecDeque<Kept>,
    /// Whether finished sequences are kept (a slot step that `keeps`, unless `keeping(false)`).
    keep: bool,
    /// The step launched and not landed (a step that `overlaps`), and whether the next may be launched before it lands; whether it landed while the next step was planned
    /// (a sequence of it swapped for room: `ensure_pages`), which then plans again.
    flight: Option<Flight>,
    overlap: bool,
    relanded: bool,
}

impl<B: ModelStep> Sched<B> {
    /// A scheduler for up to `users` sequences at a step: 1 to MAX_ROWS on pages, at most a slot step's slots and MAX_USERS on slots (their rooms are what the
    /// memory holds).
    pub fn new(be: B, users: usize) -> Sched<B> {
        let slots = be.slots();
        let pager = Pager::new(if slots.is_some() { 0 } else { be.pages() });
        let users = match slots {
            Some(k) => users.min(k).clamp(1, MAX_USERS),
            None => users.clamp(1, MAX_ROWS),
        };
        let eos = be.eos();
        let keep = slots.is_some() && be.keeps();
        let overlap = be.one_step() && be.overlaps() && overlap_asked();
        let (emit, batches) = std::sync::mpsc::channel::<Vec<(Sender<Ev>, Ev)>>();
        std::thread::spawn(move || {
            for batch in batches {
                for (to, ev) in batch {
                    let _ = to.send(ev);
                }
            }
        });
        Sched {
            pending: Vec::new(),
            emit,
            be,
            rt: Context::current(),
            kv_out: Vec::new(),
            pager,
            users,
            waiting: VecDeque::new(),
            active: Vec::new(),
            loading: Vec::new(),
            // (a development build's GLYD_ENGINE_SWAP: 0 recompute only, as before; 1 swap wherever the step can)
            swap: match crate::tune!("GLYD_ENGINE_SWAP").as_deref() {
                Some("0") => SwapMode::Never,
                Some("1") => SwapMode::Always,
                _ => SwapMode::ByCost,
            },
            next_id: 0,
            eos,
            stats: Stats::default(),
            trace: crate::tune!("GLYD_ENGINE_TRACE_SCHED").is_some(),
            pass_tokens: 0,
            skips: 0,
            free: slots.map(|k| (0..k as i32).rev().collect()),
            kept: VecDeque::new(),
            keep,
            flight: None,
            overlap,
            relanded: false,
        }
    }

    /// The scheduler with a step launched before the one before it lands where the step can (`ModelStep::overlaps`; on by default), or not.
    pub fn overlapping(mut self, on: bool) -> Sched<B> {
        self.overlap = on && self.be.one_step() && self.be.overlaps();
        self
    }

    /// The scheduler with finished sequences kept in their slots or not (kept by default where the step keeps): off for a check of what runs when every prompt begins at its
    /// first token.
    pub fn keeping(mut self, on: bool) -> Sched<B> {
        self.keep = on && self.free.is_some() && self.be.keeps();
        self
    }

    /// The scheduler with sequences sent back for room swapped as `mode` says (by cost, unless a development build's GLYD_ENGINE_SWAP says otherwise).
    pub fn swapping(mut self, mode: SwapMode) -> Sched<B> {
        self.swap = mode;
        self
    }

    /// `swapping` of a scheduler at work (a check that runs the same requests each way).
    pub fn set_swapping(&mut self, mode: SwapMode) {
        self.swap = mode;
    }

    /// The scheduler with `rt` as its runtime context: a model of a process that serves more than one, or a test's.
    pub fn in_context(mut self, rt: Arc<Context>) -> Sched<B> {
        self.rt = rt;
        self
    }

    /// Its runtime context.
    pub fn context(&self) -> &Arc<Context> {
        &self.rt
    }

    /// `in_context` of a scheduler that is already at work for a server (`server::work_with` gives it the server's).
    pub fn set_context(&mut self, rt: Arc<Context>) {
        self.rt = rt;
    }

    /// The sequences that run at once at most.
    pub fn users(&self) -> usize {
        self.users
    }

    pub fn idle(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty() && self.loading.is_empty() && self.flight.is_none()
    }

    /// Running sequence `i` taken out of the running ones: its output in the step in flight is not taken.
    fn unrun(&mut self, i: usize) -> Seq {
        let s = self.active.remove(i);
        if let Some(f) = &mut self.flight {
            f.strike(s.id);
        }
        s
    }

    /// Whether sequence `id` has tokens in the step in flight.
    fn flying(&self, id: u64) -> bool {
        self.flight.as_ref().is_some_and(|f| f.of(id).0 > 0)
    }

    /// Pages that can be had now (free or only cached); a slot step's slots that are free or only kept.
    pub fn available(&self) -> usize {
        match &self.free {
            Some(f) => f.len() + self.kept.len(),
            None => self.pager.available(),
        }
    }

    /// The finished sequences kept in their slots (a slot step that keeps them).
    pub fn kept(&self) -> usize {
        self.kept.len()
    }

    /// A slot step that keeps: sequence `i`'s prompt is all in its slot, a place a later request may go on from.
    fn mark_prompt(&mut self, i: usize) -> Result<()> {
        // (not where its next tokens are in a step launched since: it was marked as that step was launched, `launch`)
        if self.keep && !self.flying(self.active[i].id) {
            let s = &self.active[i];
            self.be.mark(s.pages[0], s.done)?;
        }
        // (a request with a cache marker: the prompt tokens it left findable by name that it did not find, the first time it runs)
        let s = &self.active[i];
        if s.pin > 0 && self.names() && s.resumed.load(Relaxed) == 0 {
            let keys = block_names(block_root(s.salt), &s.tokens[..s.tokens.len().saturating_sub(1)]);
            let findable = self.be.held_by_name(&keys);
            s.created.store(findable.saturating_sub(s.cached.load(Relaxed) as usize) as u32, Relaxed);
        }
        Ok(())
    }

    /// The oldest kept sequence's slot emptied and free again; false where none is kept.
    fn drop_kept(&mut self) -> Result<bool> {
        self.drop_kept_but(None)
    }

    /// `drop_kept`, sparing the kept sequence in slot `spare` (one a request is about to share the start of). The oldest that no request asked to be held (`Submit::pin`),
    /// or whose time is over; with none such, the oldest.
    fn drop_kept_but(&mut self, spare: Option<i32>) -> Result<bool> {
        let now = std::time::Instant::now();
        let free = |k: &Kept| Some(k.slot) != spare && k.pinned.is_none_or(|t| t <= now);
        let Some(i) = self.kept.iter().position(free).or_else(|| self.kept.iter().position(|k| Some(k.slot) != spare)) else { return Ok(false) };
        let k = self.kept.remove(i).expect("a kept sequence");
        self.be.release(&[k.slot])?;
        let free = self.free.as_mut().expect("a slot step");
        free.push(k.slot);
        free.sort_unstable_by(|a, b| b.cmp(a));
        Ok(true)
    }

    /// A slot for a sequence of `tokens` in the scope `salt`, and the tokens of it the slot already holds.
    ///
    /// The kept sequence of its scope it begins with the most of is gone on in, cut back to those tokens, where they are at least KEEP_LEAST and half of what is kept (and
    /// never the prompt's last token, which is run for its logits): a conversation's next turn. Otherwise it has a slot of its own (a free one, else the oldest kept one's,
    /// emptied), and what the step holds of its start is had without running it. A step that finds blocks by name (`ModelStep::adopt`) is asked for the full blocks the
    /// prompt begins with, in its scope: a system prompt another conversation ran, whether that one still runs, is kept, has grown far past it or has ended. Another step
    /// shares from a sequence in a slot (`ModelStep::share`), where the prompt begins with at least KEEP_LEAST tokens of one of its scope, kept or running. None: every slot
    /// is running.
    fn take_slot(&mut self, tokens: &[u32], salt: u64, keys: &[u128]) -> Result<Option<(i32, usize)>> {
        let head = &tokens[..tokens.len().saturating_sub(1)];
        let best = self.kept.iter().enumerate().filter(|(_, k)| k.salt == salt).map(|(i, k)| (common(&k.tokens, head), i)).max();
        // where the step finds blocks by name (`keys`: the names of the full blocks `head` begins with): the tokens held by them. A kept sequence is gone on in only where
        // that is no fewer tokens (its last block, not full, is its own alone): a conversation's next turn; not a prompt whose start is held longer by name than any kept
        // one has it.
        let names = self.names();
        let by_name = self.be.held_by_name(keys);
        if let Some((n, i)) = best.filter(|&(n, i)| n >= KEEP_LEAST && 2 * n >= self.kept[i].tokens.len() && n >= by_name) {
            let slot = self.kept[i].slot;
            let held = self.be.resume(slot, n)?;
            if held > 0 {
                self.kept.remove(i);
                return Ok(Some((slot, held)));
            }
            // (it cannot be gone on with from there, and for the same cause its start is not shared: it stays as it is, for its own next turn, while a slot is free, and is
            // the first to give its slot when none is, before another conversation's)
            if self.free.as_ref().is_some_and(Vec::is_empty) {
                self.kept.remove(i);
                self.be.release(&[slot])?;
                self.free.as_mut().expect("a slot step").push(slot);
            }
        }
        // where the step finds blocks by name the start is looked for so (below), whoever holds it. Else: the sequence whose start it shares the most of, (tokens, its slot)
        let running =
            self.active.iter().filter(|a| a.salt == salt && a.pages.len() == 1).map(|a| (common(&a.tokens[..a.done.min(a.tokens.len())], head), a.pages[0]));
        let source = self
            .kept
            .iter()
            .filter(|k| k.salt == salt)
            .map(|k| (common(&k.tokens, head), k.slot))
            .chain(running)
            .max()
            .filter(|&(n, _)| n >= KEEP_LEAST && self.keep && (!names || (self.be.snap_every() > 0 && n > by_name)));
        let slot = match self.free.as_mut().and_then(Vec::pop) {
            Some(slot) => slot,
            None if self.drop_kept_but(source.map(|x| x.1))? => self.free.as_mut().and_then(Vec::pop).expect("the slot just emptied"),
            None => {
                // (no slot but the kept source's own: it is gone on in after all, cut back to what the two share)
                let Some(i) = source.and_then(|(_, from)| self.kept.iter().position(|k| k.slot == from)) else { return Ok(None) };
                let k = self.kept.remove(i).expect("a kept sequence");
                let held = self.be.resume(k.slot, source.expect("a source").0)?;
                if held == 0 {
                    // (nothing of it to go on from: the slot, emptied)
                    self.be.release(&[k.slot])?;
                }
                return Ok(Some((k.slot, held)));
            }
        };
        let held = match source {
            // (a step that keeps states at boundaries shares another's mark where that is more than a name finds; where it is not, the slot is emptied and the name taken)
            Some((n, from)) => match self.be.share(from, slot, n)? {
                h if names && h < by_name => {
                    if h > 0 {
                        self.be.release(&[slot])?;
                    }
                    self.be.adopt(slot, keys)?
                }
                h => h,
            },
            None if names => self.be.adopt(slot, keys)?,
            None => 0,
        };
        Ok(Some((slot, held)))
    }

    /// Every cached page that no sequence holds given up and cleared (the extra memory they hold goes back): of a scheduler that is idle, nothing is held.
    pub fn clear_cache(&mut self) -> Result<()> {
        while self.drop_kept()? {}
        let clear = self.pager.drop_cache();
        if clear.is_empty() {
            return Ok(());
        }
        self.be.release(&clear)
    }

    pub fn submit(&mut self, s: Submit) {
        self.next_id += 1;
        let greedy = s.reply.params.greedy();
        let mut sampler = Sampler::new(s.reply.params.clone(), &s.prompt);
        if s.reply.params.min_tokens > 0 {
            // (the end tokens, unless the reply goes on past them, and the request's stop tokens: not before the least length)
            let ends = if s.reply.ignore_eos { Vec::new() } else { self.eos.iter().map(|&t| t as u32).collect() };
            sampler.set_suppress(ends.into_iter().chain(s.reply.stop_ids.iter().copied()).collect());
        }
        if self.trace {
            eprintln!("sched {:.4} submit {} ({} tokens)", crate::monitor::clock(), self.next_id, s.prompt.len());
        }
        let prompt_len = s.prompt.len();
        self.waiting.push_back(Seq {
            id: self.next_id,
            out: s.out,
            abort: s.abort,
            resumed: s.resumed,
            cached: s.cached,
            created: s.created,
            salt: s.salt,
            pin: s.pin,
            reply: s.reply,
            greedy,
            sampler,
            tokens: s.prompt,
            generated: 0,
            pages: Vec::new(),
            done: 0,
            published: 0,
            chain: 0,
            head_names: None,
            prefilling: true,
            tries: 0,
            after: Vec::new(),
            solo: false,
            swapped: None,
            began: std::time::Instant::now(),
            last_token: None,
            trace: crate::otel::Clock::new(prompt_len),
        });
        self.gauges();
    }

    fn gauges(&self) {
        let st = &self.stats;
        // (a slot step's cache is its rooms: their count, and those in use)
        let pages = self.be.slots().unwrap_or_else(|| self.pager.pages());
        self.rt.figures.kv_pages.store(pages as u64, Relaxed);
        self.rt.figures.kv_pages_held.store(pages.saturating_sub(self.available()) as u64, Relaxed);
        for (a, v) in self.rt.sched.iter().zip([
            st.prompt,
            st.hit,
            st.preempted,
            st.steps,
            st.rows,
            (self.active.len() + self.loading.len()) as u64,
            self.waiting.len() as u64,
            st.decode_ns,
            st.prefill_ns,
            st.tick_ns,
            st.release_ns,
            st.admit_ns,
            st.after_ns,
            st.before_ns,
            st.lost_blocks,
            st.heavy_blocks,
            st.taken_back,
            st.ended_lost,
            st.by_rows[0],
            st.by_rows[1],
            st.by_rows[2],
            st.by_rows[3],
            st.by_rows[4],
            st.by_rows[5],
            st.by_rows[6],
            st.by_rows[7],
            st.by_rows[8],
            st.prefill_rounds,
            st.both,
            st.swapped,
        ]) {
            a.store(v, Relaxed);
        }
    }

    /// One round: the sequences that are asked to end end, waiting ones are taken in while there is room, one prompt chunk, one decode step. On an error nothing is sure of the pages: the caller ends every sequence (`fail`).
    pub fn tick(&mut self) -> Result<()> {
        let t = std::time::Instant::now();
        let ta = std::time::Instant::now();
        self.reap()?;
        self.join_loaded(false)?;
        let came = self.stats.prompt;
        self.admit()?;
        if self.active.is_empty() && !self.loading.is_empty() {
            // (nothing else to step: the first copy back is waited for)
            self.join_loaded(true)?;
        }
        let admitted = self.stats.prompt != came;
        self.stats.admit_ns += ta.elapsed().as_nanos() as u64;
        if self.be.one_step() {
            // (a round is one step: a token for every sequence that is decoding and, beside them, the prompts' chunks; nothing waits for the other's pass)
            self.round_one_step()?;
            self.flush();
            self.kv_flush();
            self.stats.tick_ns += t.elapsed().as_nanos() as u64;
            self.gauges();
            return Ok(());
        }
        let passes = self.stats.prefill_rounds;
        self.pass_tokens = 0;
        self.prefill_step()?;
        // (the first token of a prompt that ended goes to its request now, not after the round's decode step)
        self.flush();
        // Requests that come in together are taken in over a few milliseconds: after a short pass for prompts that were taken in this round, the next round takes in the rest before the decode step
        // (which would be 17 ms of the earlier prompts' decoding between the passes), at most MAX_SKIPS rounds in a row so that a sequence that is decoding is never left for long.
        // After a pass of a long prompt (a first token is 350 ms of it) with more prompts still to run, the decode step likewise waits for up to MAX_LONG_SKIPS passes: a decode step between each two of 8 prompts of
        // 2,000 tokens put 7 x 17 ms into their mean first token (3.5% of it), and the sequences that are decoding wait two passes longer, not seven steps' worth.
        let more = self.active.iter().any(|s| s.prefilling) || !self.waiting.is_empty();
        let ran = self.stats.prefill_rounds != passes;
        if ran
            && ((admitted && self.pass_tokens <= SHORT_PASS && self.skips < MAX_SKIPS)
                || (more && self.pass_tokens > SHORT_PASS && self.skips < MAX_LONG_SKIPS))
        {
            self.skips += 1;
            self.kv_flush();
            self.stats.tick_ns += t.elapsed().as_nanos() as u64;
            self.gauges();
            return Ok(());
        }
        self.skips = 0;
        self.decode_step()?;
        self.flush();
        self.kv_flush();
        self.stats.tick_ns += t.elapsed().as_nanos() as u64;
        self.gauges();
        Ok(())
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            let _ = self.emit.send(std::mem::take(&mut self.pending));
        }
    }

    /// Every sequence ended with `msg`, the pages all free again.
    pub fn fail(&mut self, err: &crate::ffi::Error) {
        // (a step in flight: its samples are not taken)
        if self.flight.take().is_some() {
            self.be.abandon();
        }
        let all: Vec<Seq> = self.active.drain(..).chain(self.waiting.drain(..)).chain(self.loading.drain(..)).collect();
        for s in all {
            if let Some(t) = s.swapped {
                self.be.swap_drop(t);
            }
            self.rt.figures.finished("error", s.began, &s.trace, s.generated);
            self.pending.push((s.out.clone(), Ev::End(Err(err.clone()))));
        }
        let _ = self.be.lost();
        if let Some(k) = self.be.slots() {
            // (every slot emptied)
            let all: Vec<i32> = (0..k as i32).collect();
            let _ = self.be.release(&all);
            self.kept.clear();
            self.free = Some(all.into_iter().rev().collect());
            self.flush();
            self.gauges();
            return;
        }
        self.pager = Pager::new(self.be.pages());
        // (nothing is sure of the pages: every one is cleared, so that the extra memory they hold goes back)
        let all: Vec<i32> = (0..self.be.pages() as i32).collect();
        let _ = self.be.release(&all);
        self.flush();
        self.gauges();
    }

    fn reap(&mut self) -> Result<()> {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].abort.load(Relaxed) {
                self.finish(i, Finish::Stopped)?;
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.loading.len() {
            if self.loading[i].abort.load(Relaxed) {
                let mut s = self.loading.remove(i);
                self.give_back(&mut s)?;
                self.rt.figures.finished("stopped", s.began, &s.trace, s.generated);
                self.pending.push((s.out.clone(), Ev::End(Ok(Finish::Stopped))));
            } else {
                i += 1;
            }
        }
        let gone: Vec<usize> = self.waiting.iter().enumerate().filter(|(_, s)| s.abort.load(Relaxed)).map(|(i, _)| i).collect();
        for i in gone.into_iter().rev() {
            if let Some(s) = self.waiting.remove(i) {
                // (one swapped to the host lets its copy go)
                if let Some(t) = s.swapped {
                    self.be.swap_drop(t);
                }
                self.rt.figures.finished("stopped", s.began, &s.trace, s.generated);
                self.pending.push((s.out.clone(), Ev::End(Ok(Finish::Stopped))));
            }
        }
        Ok(())
    }

    /// The sequences whose copy back from the host is done (`ModelStep::swap_ready`) join the running ones, in the order they came back; with `wait`, the first is waited for.
    fn join_loaded(&mut self, wait: bool) -> Result<()> {
        let mut i = 0;
        while i < self.loading.len() {
            if self.be.swap_ready(self.loading[i].pages[0], wait && i == 0)? {
                let s = self.loading.remove(i);
                self.active.push(s);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Whether the sequence with this id is still there (running, waiting, or being copied back).
    fn alive(&self, id: u64) -> bool {
        self.active.iter().chain(&self.loading).any(|s| s.id == id) || self.waiting.iter().any(|s| s.id == id)
    }

    /// A waiting sequence that lost a block waits until the sequences that were running with it are gone. And one whose start a running sequence of its scope is still
    /// running as its own prompt waits for that, to share it (`take_slot`) and not run the same tokens beside it, where the step shares starts of any length: while at
    /// least a chunk of what the two have in common is not run yet. (A round runs a chunk of the other's prompt, so waiting for less than a chunk costs more than
    /// running those tokens does; the last of the common start, less than a chunk, the two run side by side.) It waits no longer than its own prompt would take.
    fn held(&self, s: &Seq) -> bool {
        if s.after.iter().any(|&id| self.alive(id)) {
            return true;
        }
        // (a swapped one has its own cache to copy back: it shares no start)
        if s.swapped.is_some() || !self.keep || !self.be.shares_starts() {
            return false;
        }
        let chunk = self.be.chunk().max(KEEP_LEAST);
        let head = &s.tokens[..s.tokens.len().saturating_sub(1)];
        self.active.iter().any(|a| {
            let upto = a.done + chunk;
            a.salt == s.salt && a.prefilling && upto <= a.tokens.len().min(head.len()) && a.tokens[..upto] == head[..upto]
        })
    }

    /// Waiting sequences taken in, the oldest first (those that are not held back), while there are places and pages for the whole prompt and a page for each running sequence to grow into.
    fn admit(&mut self) -> Result<()> {
        while self.active.len() + self.loading.len() < self.users {
            // a sequence that must run alone runs when nothing else does, and nothing else is let in while it waits or runs
            if self.active.iter().any(|s| s.solo) {
                break;
            }
            let k = match self.waiting.iter().position(|s| s.solo) {
                Some(k) if self.active.is_empty() => k,
                Some(_) => break,
                None => match (0..self.waiting.len()).find(|&k| !self.held(&self.waiting[k])) {
                    Some(k) => k,
                    None => break,
                },
            };
            if self.free.is_some() {
                if !self.admit_slot(k)? {
                    break;
                }
                continue;
            }
            let need = self.waiting[k].tokens.len().div_ceil(PAGE) + 1;
            if self.pager.available() < need + self.active.len() {
                if self.active.is_empty() {
                    // nothing will free a page: this one never fits
                    let s = self.waiting.remove(k).expect("a waiting sequence");
                    self.rt.figures.finished("error", s.began, &s.trace, s.generated);
                    self.pending.push((
                        s.out.clone(),
                        Ev::End(Err(Error::of(Kind::Request, format!("{} tokens, where the cache holds {} blocks", s.tokens.len(), self.pager.pages())))),
                    ));
                    continue;
                }
                break;
            }
            let mut s = self.waiting.remove(k).expect("a waiting sequence");
            s.trace.admit();
            // the pages of an earlier sequence with the same first tokens (at least the last token is run, for the logits)
            let root = block_root(s.salt);
            let hit = self.pager.lookup_in(root, &s.tokens, s.tokens.len().saturating_sub(1) / PAGE);
            s.chain = block_names(root, &s.tokens[..hit.len() * PAGE]).last().copied().unwrap_or(root);
            s.published = hit.len();
            s.done = hit.len() * PAGE;
            s.found(s.done);
            s.pages = hit;
            s.prefilling = true;
            s.after.clear();
            self.stats.prompt += s.tokens.len() as u64;
            self.stats.hit += s.done as u64;
            self.active.push(s);
        }
        Ok(())
    }

    /// Waiting sequence `k` given a slot of its own (a slot step); false where none is free. A prompt the slot cannot hold with a token after it ends at once with an error in words.
    fn admit_slot(&mut self, k: usize) -> Result<bool> {
        if let Some(ticket) = self.waiting[k].swapped {
            return self.admit_swapped(k, ticket);
        }
        let window = self.be.window();
        if self.waiting[k].tokens.len() >= window {
            let s = self.waiting.remove(k).expect("a waiting sequence");
            self.pending.push((s.out.clone(), Ev::End(Err(Error::of(Kind::Request, format!("{} tokens, where the context holds {window}", s.tokens.len()))))));
            return Ok(true);
        }
        if self.names() && self.waiting[k].head_names.is_none() {
            let s = &self.waiting[k];
            let v = block_names(block_root(s.salt), &s.tokens[..s.tokens.len().saturating_sub(1)]);
            self.waiting[k].head_names = Some(v);
        }
        let keys = self.waiting[k].head_names.clone().unwrap_or_default();
        let Some((slot, held)) = self.take_slot(&self.waiting[k].tokens.clone(), self.waiting[k].salt, &keys)? else { return Ok(false) };
        // (room for the whole prompt and a token after it now: an admitted prompt runs; where there is none the kept sequences give theirs up, the oldest first, then it waits for
        // the running ones, and with nothing running it never fits)
        let need = self.waiting[k].tokens.len() + 1;
        let mut room = self.be.grow(slot, need)?;
        while !room && self.drop_kept()? {
            room = self.be.grow(slot, need)?;
        }
        if !room {
            // (what the slot held of a kept sequence goes with it)
            if held > 0 {
                self.be.release(&[slot])?;
            }
            self.free.as_mut().expect("a slot step").push(slot);
            if self.active.is_empty() && self.loading.is_empty() {
                let s = self.waiting.remove(k).expect("a waiting sequence");
                self.rt.figures.finished("error", s.began, &s.trace, s.generated);
                self.pending.push((
                    s.out.clone(),
                    Ev::End(Err(Error::of(Kind::Request, format!("{} tokens, where the cache holds fewer with nothing else running", s.tokens.len())))),
                ));
                return Ok(true);
            }
            return Ok(false);
        }
        let mut s = self.waiting.remove(k).expect("a waiting sequence");
        s.trace.admit();
        s.pages = vec![slot];
        // (what it holds in full blocks has its names already: they are the names of these tokens)
        s.published = if self.names() { held / PAGE } else { 0 };
        s.chain = s.published.checked_sub(1).and_then(|b| keys.get(b)).copied().unwrap_or_else(|| block_root(s.salt));
        s.head_names = None;
        (s.done, s.prefilling) = (held, true);
        s.found(held);
        s.after.clear();
        self.stats.prompt += s.tokens.len() as u64;
        self.stats.hit += held as u64;
        self.active.push(s);
        Ok(true)
    }

    /// Waiting sequence `k`, swapped to the host (`ticket`), copied back into an empty slot (a free one, else a kept one's) with room for its next step (its next token, or its prompt's next chunk):
    /// it waits in `loading` until the copy is done, the steps going on meanwhile. False where there is no slot or no room now. Where nothing runs or comes back that would give memory back, it is
    /// run again from its tokens instead (its copy let go), as one sent back is without a swap.
    fn admit_swapped(&mut self, k: usize, ticket: u64) -> Result<bool> {
        let slot = match self.free.as_mut().and_then(Vec::pop) {
            Some(slot) => slot,
            None if self.drop_kept()? => self.free.as_mut().and_then(Vec::pop).expect("the slot just emptied"),
            None => return Ok(false),
        };
        let s = &self.waiting[k];
        let upto = if s.prefilling { s.done + (s.tokens.len() - s.done).min(self.be.chunk()) } else { s.done + 1 };
        let held = loop {
            if let Some(n) = self.be.swap_in(slot, ticket, upto)? {
                break n;
            }
            if self.drop_kept()? {
                continue;
            }
            let free = self.free.as_mut().expect("a slot step");
            free.push(slot);
            free.sort_unstable_by(|a, b| b.cmp(a));
            if !self.active.is_empty() || !self.loading.is_empty() {
                return Ok(false);
            }
            self.be.swap_drop(ticket);
            let s = &mut self.waiting[k];
            (s.swapped, s.done, s.published, s.chain, s.head_names, s.prefilling) = (None, 0, 0, block_root(s.salt), None, true);
            return Ok(true);
        };
        let mut s = self.waiting.remove(k).expect("a waiting sequence");
        if held != s.done {
            return Err(Error(format!("a sequence of {} tokens came back from the host with {held}", s.done)));
        }
        s.trace.admit();
        s.pages = vec![slot];
        s.swapped = None;
        s.after.clear();
        self.loading.push(s);
        Ok(true)
    }

    /// Pages for the sequence `id`'s first `upto` tokens, the newest sequences giving theirs back where there are none to take. False where `id` itself was sent back to wait. (A slot step's room grows
    /// the same way, `ModelStep::grow`.)
    fn ensure_pages(&mut self, id: u64, upto: usize) -> Result<bool> {
        if self.free.is_some() {
            loop {
                let Some(i) = self.active.iter().position(|s| s.id == id) else { return Ok(false) };
                if self.be.grow(self.active[i].pages[0], upto)? {
                    return Ok(true);
                }
                // (a kept sequence's room before a running one's)
                if self.drop_kept()? {
                    continue;
                }
                if self.active.len() == 1 && self.loading.is_empty() {
                    // alone, nothing kept or coming back: no room will come. Sent back, it would be let in again to the same end, without end: it ends with the reason
                    let mut s = self.unrun(i);
                    self.give_back(&mut s)?;
                    self.rt.figures.finished("error", s.began, &s.trace, s.generated);
                    let msg = format!("{upto} tokens, where the cache holds fewer with nothing else running");
                    self.pending.push((s.out.clone(), Ev::End(Err(Error::of(Kind::Request, msg)))));
                    return Ok(false);
                }
                let victim = self.active.iter().enumerate().max_by_key(|(_, s)| s.id).map(|(v, _)| v).expect("an active sequence");
                // (one with tokens in the step in flight that would be swapped: that step lands first, so that its cache is what its tokens say when it is copied; then the room is
                // asked for again, the step's own having ended or grown)
                if self.flying(self.active[victim].id) && self.swap_pays(&self.active[victim]) {
                    let f = self.flight.take().expect("a step in flight");
                    self.land(f)?;
                    self.relanded = true;
                    continue;
                }
                self.preempt(victim, true)?;
            }
        }
        loop {
            let Some(i) = self.active.iter().position(|s| s.id == id) else { return Ok(false) };
            if self.active[i].pages.len() >= upto.div_ceil(PAGE) {
                return Ok(true);
            }
            match self.pager.alloc() {
                Some(p) => self.active[i].pages.push(p),
                None => {
                    let victim = self.active.iter().enumerate().max_by_key(|(_, s)| s.id).map(|(v, _)| v).expect("an active sequence");
                    self.preempt(victim, true)?;
                }
            }
        }
    }

    /// After a step: what the cache could not store exactly in it (lost blocks) and what it kept in the second extra memory (heavy blocks). Both are taken out of the prefix cache and kept out (an idle one is cleared at once, a
    /// held one when its holders go); where the status's rings overflowed any block may be lost, and the prefix cache is dropped. The ids of the sequences whose step must be taken back: those that hold a lost block (every
    /// running sequence where the rings overflowed).
    fn after_step(&mut self) -> Result<Vec<u64>> {
        let l = self.be.lost();
        if l.is_empty() {
            return Ok(Vec::new());
        }
        self.stats.lost_blocks += l.void.len() as u64;
        self.stats.heavy_blocks += l.raw.len() as u64;
        let mut clear = Vec::new();
        for &b in l.raw.iter().chain(&l.void) {
            if self.pager.evict(b, true) {
                clear.push(b);
            }
        }
        if l.overflow {
            clear.extend(self.pager.drop_cache());
        }
        if !clear.is_empty() {
            self.be.release(&clear)?;
        }
        let void: std::collections::HashSet<i32> = l.void.iter().copied().collect();
        // (a sequence being copied back is hit too where a block it is copied into could not be stored exactly)
        Ok(self.active.iter().chain(&self.loading).filter(|s| l.overflow || s.pages.iter().any(|p| void.contains(p))).map(|s| s.id).collect())
    }

    /// The sequences `hit` (the step's output of each is not to be used) given back their pages and sent to wait, to run again from their tokens when the sequences still running that were not hit have ended (and the ones hit
    /// before them): they have the cache's extra memory to themselves then. A sequence hit twice while others ran is sent to run alone, with nothing else let in. One hit while it runs alone has the cache of finished prompts (which
    /// holds extra memory too) given up and runs again from nothing; hit alone with that cache empty, it does not fit the cache's extra memory, and ends with an error in words (as does one that has been hit MAX_TRIES times, whatever
    /// ran with it).
    fn take_back(&mut self, hit: &[u64]) -> Result<()> {
        if hit.is_empty() {
            return Ok(());
        }
        let mut hit = hit.to_vec();
        hit.sort_unstable();
        // (one hit while it was copied back has its copy finished and is taken back as a running one is: its cache is not what it was)
        let mut i = 0;
        while i < self.loading.len() {
            if hit.contains(&self.loading[i].id) {
                self.be.swap_ready(self.loading[i].pages[0], true)?;
                let s = self.loading.remove(i);
                self.active.push(s);
            } else {
                i += 1;
            }
        }
        let others: Vec<u64> = self.active.iter().map(|s| s.id).filter(|id| !hit.contains(id)).collect();
        let alone = others.is_empty() && hit.len() == 1;
        let mut before: Vec<u64> = Vec::new();
        for &id in &hit {
            let Some(i) = self.active.iter().position(|s| s.id == id) else { continue };
            self.stats.taken_back += 1;
            self.active[i].tries += 1;
            let mut end = self.active[i].tries > MAX_TRIES;
            if end {
                self.stats.ended_tries += 1;
            }
            #[cfg(feature = "dev")]
            if self.trace {
                let s = &self.active[i];
                eprintln!(
                    "sched: step taken back: sequence {} ({} tokens, {} done, {} pages held, tries {}, solo {}), {} others running, {} hit together, alone {alone}, idle cached pages {}, pages free or idle {}, {} waiting",
                    s.id, s.tokens.len(), s.done, s.pages.len(), s.tries, s.solo, others.len(), hit.len(), self.pager.idle_pages(), self.pager.available(), self.waiting.len()
                );
            }
            if alone && !end {
                // nothing else runs: what else holds the extra memory is pages kept for later prompts: given up, and the sequence's own pages are not kept (it runs again from nothing); with none to give up it is alone with the cache
                self.active[i].done = 0;
                let clear = self.pager.drop_cache();
                if clear.is_empty() {
                    end = true;
                    self.stats.ended_alone += 1;
                } else {
                    self.be.release(&clear)?;
                }
            }
            if end {
                // it does not fit the cache's extra memory: ended, with the reason
                #[cfg(feature = "dev")]
                if self.trace {
                    eprintln!("sched: sequence {} ended for the cache's extra memory (tries {}, alone {alone})", id, self.active[i].tries);
                }
                self.stats.ended_lost += 1;
                let mut s = self.unrun(i);
                self.give_back(&mut s)?;
                self.rt.figures.finished("error", s.began, &s.trace, s.generated);
                self.pending.push((s.out.clone(), Ev::End(Err(Error::of(Kind::Request, ALONE_MESSAGE)))));
                continue;
            }
            let solo = !alone && self.active[i].tries >= 2;
            self.preempt(i, false)?;
            let s = self.waiting.front_mut().expect("it was just sent back to wait");
            s.solo |= solo;
            s.after = if s.solo { Vec::new() } else { others.iter().chain(before.iter()).copied().collect() };
            before.push(id);
        }
        Ok(())
    }

    /// A sequence sent back to wait, first in the queue. With `swap` (sent back for room, its cache whole), where `swap_pays`: its cache copied to the host (`ModelStep::swap_out`; its full blocks
    /// named first, as `give_back` names them), to be copied back and go on from where it is. Otherwise, or where the host's area has no room: its full pages go to the cache, the rest are cleared,
    /// and it is run again from its tokens.
    fn preempt(&mut self, i: usize, swap: bool) -> Result<()> {
        // (not one with tokens in the step in flight, whose cache is not what its tokens say until that step lands: `ensure_pages` lands it first where a swap would pay)
        if swap && !self.flying(self.active[i].id) && self.swap_pays(&self.active[i]) {
            let slot = self.active[i].pages[0];
            if self.names() {
                Self::name_blocks(&mut self.be, &mut self.active[i], self.rt.kv_events.as_ref().map(|_| &mut self.kv_out))?;
            }
            let t = std::time::Instant::now();
            let out = self.be.swap_out(slot)?;
            self.stats.release_ns += t.elapsed().as_nanos() as u64;
            if let Some(ticket) = out {
                let mut s = self.unrun(i);
                s.pages.clear();
                let free = self.free.as_mut().expect("a slot step");
                free.push(slot);
                free.sort_unstable_by(|a, b| b.cmp(a));
                s.swapped = Some(ticket);
                s.resumed.fetch_add(1, Relaxed);
                self.stats.preempted += 1;
                self.stats.swapped += 1;
                self.waiting.push_front(s);
                return Ok(());
            }
        }
        let mut s = self.unrun(i);
        self.give_back(&mut s)?;
        s.done = 0;
        s.published = 0;
        s.chain = block_root(s.salt);
        // (its tokens are more than they were when it came)
        s.head_names = None;
        s.prefilling = true;
        s.resumed.fetch_add(1, Relaxed);
        self.stats.preempted += 1;
        self.waiting.push_front(s);
        Ok(())
    }

    /// Whether sequence `s`, sent back for room, is swapped to the host rather than run again from its tokens (`SwapMode`): a slot step's sequence that holds tokens and that the step can swap
    /// (`ModelStep::swap_cost`), and by cost where its bytes over the link, out and back, take less time than its tokens' recompute at the rate the prompt passes have run at so far (every token it
    /// holds: a recurrent model's states are had only by running all of them; no pass run yet: swapped).
    fn swap_pays(&self, s: &Seq) -> bool {
        if self.swap == SwapMode::Never || self.free.is_none() || s.pages.len() != 1 || s.done == 0 {
            return false;
        }
        let Some((bytes, rate)) = self.be.swap_cost(s.pages[0]) else { return false };
        if self.swap == SwapMode::Always {
            return true;
        }
        let st = &self.stats;
        let again = if st.pass_tokens == 0 { f64::INFINITY } else { s.done as f64 * st.pass_ns as f64 / st.pass_tokens as f64 * 1e-9 };
        2.0 * bytes as f64 / rate.max(1.0) < again
    }

    /// The sequence's full pages given to the cache and all its pages released; those nothing names are cleared on the GPU. A slot step's slot is emptied and free again.
    fn give_back(&mut self, s: &mut Seq) -> Result<()> {
        if self.names() {
            // (its full blocks stay findable by name when the slot is emptied)
            Self::name_blocks(&mut self.be, s, self.rt.kv_events.as_ref().map(|_| &mut self.kv_out))?;
        }
        if let Some(free) = &mut self.free {
            let slots = std::mem::take(&mut s.pages);
            free.extend(slots.iter().copied());
            free.sort_unstable_by(|a, b| b.cmp(a));
            let t = std::time::Instant::now();
            let r = self.be.release(&slots);
            self.stats.release_ns += t.elapsed().as_nanos() as u64;
            return r;
        }
        while (s.published + 1) * PAGE <= s.done {
            let k = s.published;
            s.chain = self.pager.publish(s.pages[k], s.chain, &s.tokens[k * PAGE..(k + 1) * PAGE]);
            s.published += 1;
        }
        self.be.verify(&s.pages[..(s.done / PAGE).min(s.pages.len())])?;
        let freed = self.pager.release(&s.pages);
        s.pages.clear();
        let t = std::time::Instant::now();
        let r = self.be.release(&freed);
        self.stats.release_ns += t.elapsed().as_nanos() as u64;
        r
    }

    fn finish(&mut self, i: usize, f: Finish) -> Result<()> {
        let mut s = self.unrun(i);
        if self.keep && s.done >= KEEP_LEAST && s.pages.len() == 1 {
            // (kept in its slot: the conversation's next turn begins with these tokens)
            if self.names() {
                Self::name_blocks(&mut self.be, &mut s, self.rt.kv_events.as_ref().map(|_| &mut self.kv_out))?;
            }
            let slot = s.pages.pop().expect("its slot");
            let pinned = (s.pin > 0).then(|| std::time::Instant::now() + std::time::Duration::from_secs(s.pin as u64));
            self.kept.push_back(Kept { slot, tokens: s.tokens[..s.done].to_vec(), salt: s.salt, pinned });
        } else {
            self.give_back(&mut s)?;
        }
        self.rt.figures.finished(finish_name(f), s.began, &s.trace, s.generated);
        self.pending.push((s.out.clone(), Ev::End(Ok(f))));
        Ok(())
    }

    /// A prompt piece of `n` tokens from `done`, cut at the next boundary the step keeps a sequence's states at (`ModelStep::snap_every`): the states it keeps there are
    /// then the states after that token.
    fn snap_cut(&self, done: usize, n: usize) -> usize {
        let every = if self.names() { self.be.snap_every() } else { 0 };
        if every == 0 {
            return n;
        }
        let next = (done / every + 1) * every;
        if done + n > next {
            next - done
        } else {
            n
        }
    }

    /// Whether the full blocks of the sequences are told to the step by name and a prompt's start is looked for by name: a slot step that finds blocks so, keeping on.
    fn names(&self) -> bool {
        self.keep && self.free.is_some() && self.be.shares_starts()
    }

    /// The full blocks of a slot step's sequence that the step has not been told the names of yet, told (`ModelStep::name`).
    /// With `out` (the cache's events are kept): the blocks told, their parent's name and their tokens, as a stored event.
    fn name_blocks(be: &mut B, s: &mut Seq, out: Option<&mut Vec<crate::kv_events::Event>>) -> Result<()> {
        let done = s.done;
        Self::name_blocks_to(be, s, done, out)
    }

    /// `name_blocks` of the blocks of its first `upto` tokens (more than it has done where the step in flight writes the rest: `launch`).
    fn name_blocks_to(be: &mut B, s: &mut Seq, upto: usize, out: Option<&mut Vec<crate::kv_events::Event>>) -> Result<()> {
        let first = s.published;
        let parent = (first > 0).then_some(s.chain);
        let mut keys = Vec::new();
        while (s.published + 1) * PAGE <= upto {
            s.chain = block_hash(s.chain, &s.tokens[s.published * PAGE..(s.published + 1) * PAGE]);
            keys.push(s.chain);
            s.published += 1;
        }
        match (keys.is_empty(), &s.pages[..]) {
            (false, &[slot]) => {
                be.name(slot, first, &keys)?;
                if let Some(out) = out {
                    out.push(crate::kv_events::Event::Stored { blocks: keys, parent, tokens: s.tokens[first * PAGE..s.published * PAGE].to_vec() });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The step's events as one batch, where the cache's events are kept: the blocks no longer found by their names first (taken while sequences were let in, before
    /// the step ran), then those named.
    fn kv_flush(&mut self) {
        let removed = self.be.take_removed();
        if let Some(log) = &self.rt.kv_events {
            let mut ev = Vec::with_capacity(self.kv_out.len() + 1);
            if !removed.is_empty() {
                ev.push(crate::kv_events::Event::Removed { blocks: removed });
            }
            ev.append(&mut self.kv_out);
            log.push(ev);
        }
    }

    /// Full pages of the prompt as far as it has run, offered to the cache at once (a request with the same first tokens that comes now finds them).
    fn publish(&mut self, i: usize) -> Result<()> {
        if self.free.is_some() {
            return if self.names() {
                Self::name_blocks(&mut self.be, &mut self.active[i], self.rt.kv_events.as_ref().map(|_| &mut self.kv_out))
            } else {
                Ok(())
            };
        }
        let s = &mut self.active[i];
        while (s.published + 1) * PAGE <= s.done {
            let k = s.published;
            s.chain = self.pager.publish(s.pages[k], s.chain, &s.tokens[k * PAGE..(k + 1) * PAGE]);
            s.published += 1;
        }
        Ok(())
    }

    /// The token the sample after row `row` of the step stands for: the largest logit, or a draw from the logits as the request's sampling says; with it, its log-probabilities where the request asks for them
    /// (from the whole row: the draw may have looked at the largest 64 alone, the log-sum-exp is of all).
    fn pick(&mut self, i: usize, sample: Sample, row: usize) -> Result<(u32, Option<Lp>)> {
        let want = self.active[i].reply.logprobs;
        if self.active[i].greedy && want.is_none() {
            return Ok((sample.token, None));
        }
        // the row's largest logits come with the step: where they are all the sampler looks at (a top-k within them, no penalties, no bias) the logits are not copied to the host, unless log-probabilities are asked for
        let top = self.be.top(row);
        if want.is_none() {
            if let Some(t) = top.as_deref().and_then(|c| self.active[i].sampler.sample_top(c, None)) {
                return Ok((t, None));
            }
            // (penalties, a bias or suppressed tokens: the candidates and those tokens' own logits, changed as the whole row's would be, where that shows the draw)
            if let (Some(c), Some(ts)) = (top.as_deref(), self.active[i].sampler.adjusts()) {
                let raw = self.be.gather(row, &ts)?;
                if let Some(t) = self.active[i].sampler.sample_adjusted(c, &raw) {
                    return Ok((t, None));
                }
            }
        }
        let logits = self.be.logits(row)?;
        let tok = if self.active[i].greedy { sample.token } else { self.active[i].sampler.sample(&logits) };
        Ok((tok, want.map(|k| logprobs_of(&logits, tok, k, top.as_deref()))))
    }

    /// A token chosen for sequence `i`: sent to its request and added; Some(how it ends) where it ends (the end-of-sequence token is not a token of the reply).
    fn accept(&mut self, i: usize, tok: u32, lp: Option<Lp>) -> Option<Finish> {
        let window = self.be.window();
        let s = &mut self.active[i];
        if let Some(lp) = lp {
            self.pending.push((s.out.clone(), Ev::Lp(lp)));
        }
        if (!s.reply.ignore_eos && self.eos.contains(&(tok as i64))) || s.reply.stop_ids.contains(&tok) {
            return Some(Finish::Stop);
        }
        if s.abort.load(Relaxed) {
            return Some(Finish::Stopped);
        }
        self.pending.push((s.out.clone(), Ev::Token(tok)));
        self.rt.figures.token(s.began, &mut s.last_token);
        s.trace.token();
        s.tokens.push(tok);
        s.generated += 1;
        if s.generated >= s.reply.max_new || s.tokens.len() > window {
            return Some(Finish::Length);
        }
        None
    }

    /// One chunk of the oldest sequence that is still on its prompt, and with it the prompts of the next ones that are still on theirs, as many as the backend runs in one pass and a chunk's tokens hold (each of those
    /// a whole rest of its prompt, so that no sequence is cut differently for it); at a prompt's end, its first token.
    fn prefill_step(&mut self) -> Result<()> {
        let at = |a: &Vec<Seq>, id: u64| a.iter().position(|s| s.id == id);
        let cands: Vec<u64> = self.active.iter().filter(|s| s.prefilling).take(self.be.prefill_rows().max(1)).map(|s| s.id).collect();
        let budget = self.be.chunk();
        let together = self.be.prefill_together();
        let mut picks: Vec<(u64, usize)> = Vec::new();
        let mut used = 0;
        for id in cands {
            let Some(i) = at(&self.active, id) else { continue };
            let left = self.active[i].tokens.len() - self.active[i].done;
            let n = self.snap_cut(self.active[i].done, if picks.is_empty() { left.min(budget) } else { left });
            if n == 0 || used + n > budget || (!picks.is_empty() && n > together) {
                break;
            }
            let upto = self.active[i].done + n;
            if !self.ensure_pages(id, upto)? {
                continue;
            }
            picks.push((id, n));
            used += n;
            if n > together {
                break; // (too long to run with others: alone)
            }
        }
        // (the pages of a later one may have been taken from an earlier one, which went back to wait)
        let slots = self.free.is_some();
        picks.retain(|&(id, n)| {
            at(&self.active, id).is_some_and(|i| self.active[i].prefilling && (slots || self.active[i].pages.len() * PAGE >= self.active[i].done + n))
        });
        if picks.is_empty() {
            return Ok(());
        }
        let jobs: Vec<PrefillJob> = picks
            .iter()
            .map(|&(id, n)| {
                let s = &self.active[at(&self.active, id).expect("a picked sequence")];
                PrefillJob { pages: &s.pages, start: s.done, tokens: &s.tokens[s.done..s.done + n], last: s.done + n == s.tokens.len() }
            })
            .collect();
        let t = std::time::Instant::now();
        let t0 = crate::monitor::clock();
        // (the grammar masks of the sequences whose prompt ends in this pass, in the order of their samples)
        let masks: Vec<_> = picks
            .iter()
            .filter_map(|&(id, n)| at(&self.active, id).filter(|&i| self.active[i].done + n == self.active[i].tokens.len()))
            .map(|i| self.active[i].sampler.mask_handle())
            .collect();
        self.be.masks(masks);
        let samples = self.be.launch(&[], &jobs).and_then(|l| self.be.finish(l))?.1;
        if self.trace {
            eprintln!(
                "sched {:.4} pass from {:.4} of {} sequences {:?}, {} tokens, {:.1} ms; {} active, {} waiting",
                crate::monitor::clock(),
                t0,
                jobs.len(),
                picks.iter().map(|p| p.0).collect::<Vec<_>>(),
                used,
                t.elapsed().as_secs_f64() * 1e3,
                self.active.len(),
                self.waiting.len()
            );
        }
        drop(jobs);
        self.stats.prefill_ns += t.elapsed().as_nanos() as u64;
        self.stats.pass_ns += t.elapsed().as_nanos() as u64;
        self.stats.pass_tokens += used as u64;
        self.stats.prefill_rounds += 1;
        self.pass_tokens = used;
        // what the chunks could not store exactly: not used, whatever they were told (the sequence is sent back, with any other that holds such a block)
        let hit = self.after_step()?;
        let mine: Vec<bool> = picks.iter().map(|(id, _)| hit.contains(id)).collect();
        self.take_back(&hit)?;
        let mut row = 0;
        for (k, &(id, n)) in picks.iter().enumerate() {
            let sample = samples.get(k).copied().flatten();
            let my_row = row;
            row += sample.is_some() as usize;
            if mine[k] {
                continue;
            }
            let Some(i) = at(&self.active, id) else { continue };
            self.active[i].done += n;
            self.publish(i)?;
            if let Some(sample) = sample {
                self.active[i].prefilling = false;
                self.mark_prompt(i)?;
                let (tok, lp) = self.pick(i, sample, my_row)?;
                if let Some(f) = self.accept(i, tok, lp) {
                    self.finish(i, f)?;
                }
            }
        }
        Ok(())
    }

    /// A round as one step (`ModelStep::one_step`): a token for every sequence that is decoding and, in the same pass, chunks of the prompts of those still on theirs, the oldest first: as
    /// many tokens as the backend takes beside those rows (`prompt_budget`; with no row decoding, a chunk's). The first prompt's chunk is cut to that; a later prompt joins only with its whole
    /// rest, as in `prefill_step`. A sequence that is decoding gets a token every round whatever prompts come in, and a prompt's first token does not wait for a decode step between its chunks.
    ///
    /// Where the step `overlaps` (docs/design/runtime.md section 5, "Launch and finish"), the round's step is launched before the step before it lands: the rows whose token that step chose
    /// read it on the GPU (`Row::after`), and the host takes that step's samples, sends its tokens and ends its sequences while the GPU runs this one. That is so only where every sequence that
    /// samples in the step before takes its largest logit (greedy, no log-probabilities): otherwise that step lands first, its tokens drawn on the host, and this one is launched after. A
    /// sequence that ends in a step has had its row in the next one run too (the end is seen only when it lands); that row's output is not taken.
    fn round_one_step(&mut self) -> Result<()> {
        if self.flight.as_ref().is_some_and(|f| !(self.overlap && f.on_device)) {
            let f = self.flight.take().expect("a step in flight");
            self.land(f)?;
        }
        let Some(plan) = self.plan()? else {
            if let Some(f) = self.flight.take() {
                self.land(f)?;
            }
            return Ok(());
        };
        let f = self.launch(plan)?;
        if !self.overlap {
            return self.land(f);
        }
        match self.flight.replace(f) {
            Some(before) => self.land(before),
            None => Ok(()),
        }
    }

    /// The round's step (`round_one_step`): room for the next token of each sequence that is decoding (the newest give theirs back where there is none), and the prompts' chunks beside them.
    /// A sequence with tokens in the step in flight goes on after them: a decode row there (or its prompt's last chunk) makes this one's row a token not had yet (the head row it is in).
    fn plan(&mut self) -> Result<Option<Plan>> {
        let at = |a: &Vec<Seq>, id: u64| a.iter().position(|s| s.id == id);
        let tb = std::time::Instant::now();
        let ahead = |f: &Option<Flight>, id: u64| f.as_ref().map_or((0, None), |f| f.of(id));
        // (a sequence is decoding when its prompt has run, or its last chunk is in flight; one whose token in flight is its last by its length or the window's has no row,
        // nor anything else, in this step)
        let window = self.be.window();
        let ends = |f: &Option<Flight>, s: &Seq| ahead(f, s.id).1.is_some() && (s.generated + 1 >= s.reply.max_new || s.tokens.len() + 1 > window);
        let decoding = |f: &Option<Flight>, s: &Seq| (!s.prefilling || ahead(f, s.id).1.is_some()) && !ends(f, s);
        let prompting = |f: &Option<Flight>, s: &Seq| s.prefilling && ahead(f, s.id).1.is_none();
        let ids: Vec<u64> = self.active.iter().filter(|s| decoding(&self.flight, s)).map(|s| s.id).collect();
        for id in ids {
            let Some(s) = self.active.iter().find(|s| s.id == id) else { continue };
            let upto = s.done + ahead(&self.flight, id).0 + 1;
            self.ensure_pages(id, upto)?;
        }
        let rows: Vec<(u64, Option<usize>)> = self.active.iter().filter(|s| decoding(&self.flight, s)).map(|s| (s.id, ahead(&self.flight, s.id).1)).collect();
        let n_rows = rows.len();
        let budget = if n_rows == 0 { self.be.chunk() } else { self.be.prompt_budget(n_rows).min(self.be.chunk().saturating_sub(n_rows)) };
        let together = self.be.prefill_together();
        // (as many prompts as a pass of prompts takes: with the rows that are decoding they are no more than the sequences the scheduler runs at once, which the step's head holds)
        let most = self.be.prefill_rows().max(1);
        let cands: Vec<u64> = self.active.iter().filter(|s| prompting(&self.flight, s)).take(most).map(|s| s.id).collect();
        let mut picks: Vec<(u64, usize)> = Vec::new();
        let mut used = 0;
        for id in cands {
            let Some(i) = at(&self.active, id) else { continue };
            let done = self.active[i].done + ahead(&self.flight, id).0;
            let left = self.active[i].tokens.len() - done;
            let n = self.snap_cut(done, if picks.is_empty() { left.min(budget) } else { left });
            if n == 0 || used + n > budget || (!picks.is_empty() && n > together) {
                break;
            }
            if !self.ensure_pages(id, done + n)? {
                continue;
            }
            picks.push((id, n));
            used += n;
            if n > together {
                break; // (too long to run with other prompts: the only one of the step)
            }
        }
        // (the room of a later one may have been taken from an earlier one, or from one that was decoding: those went back to wait)
        let slots = self.free.is_some();
        picks.retain(|&(id, n)| {
            at(&self.active, id).is_some_and(|i| {
                let s = &self.active[i];
                s.prefilling && (slots || s.pages.len() * PAGE >= s.done + ahead(&self.flight, id).0 + n)
            })
        });
        let rows: Vec<(u64, Option<usize>)> = rows.into_iter().filter(|r| at(&self.active, r.0).is_some()).collect();
        self.stats.before_ns += tb.elapsed().as_nanos() as u64;
        if std::mem::take(&mut self.relanded) {
            // (the step in flight landed on the way: its tokens are the host's now, and what ended in it has gone; the room taken stays taken)
            return self.plan();
        }
        Ok((!rows.is_empty() || !picks.is_empty()).then_some(Plan { rows, picks, used }))
    }

    /// The planned step launched (`ModelStep::launch`), with the grammar masks of the rows that are sampled (the decode rows, then the prompts that end in it).
    ///
    /// Where the step in flight lands after it, what landing would ask of the step for that one's sequences is asked first, so that it is queued before this step's work:
    /// the blocks that step fills named (a model that keeps states at boundaries copies the states there), and each prompt that ends in it marked (a recurrent model's states
    /// at the prompt's end copied aside). Those are the states after that step's tokens, which this step's rows go on from.
    fn launch(&mut self, plan: Plan) -> Result<Flight> {
        let at = |a: &Vec<Seq>, id: u64| a.iter().position(|s| s.id == id);
        if let Some(f) = &self.flight {
            let ahead: Vec<(u64, usize, bool)> = f.rows.iter().map(|&id| (id, 1, false)).chain(f.picks.iter().copied()).collect();
            for (id, n, last) in ahead {
                let Some(i) = at(&self.active, id) else { continue };
                let upto = self.active[i].done + n;
                if self.names() {
                    Self::name_blocks_to(&mut self.be, &mut self.active[i], upto, self.rt.kv_events.as_ref().map(|_| &mut self.kv_out))?;
                }
                if last && self.keep {
                    let slot = self.active[i].pages[0];
                    self.be.mark(slot, upto)?;
                }
            }
        }
        let ahead = |f: &Option<Flight>, id: u64| f.as_ref().map_or(0, |f| f.of(id).0);
        let picks: Vec<(u64, usize, bool)> = plan
            .picks
            .iter()
            .map(|&(id, n)| {
                let s = &self.active[at(&self.active, id).expect("a picked sequence")];
                (id, n, s.done + ahead(&self.flight, id) + n == s.tokens.len())
            })
            .collect();
        let rows: Vec<Row> = plan
            .rows
            .iter()
            .map(|&(id, after)| {
                let s = &self.active[at(&self.active, id).expect("a decoding sequence")];
                let pos = s.done + ahead(&self.flight, id);
                Row { token: if after.is_some() { 0 } else { s.tokens[pos] }, pos, pages: &s.pages, after }
            })
            .collect();
        let jobs: Vec<PrefillJob> = picks
            .iter()
            .map(|&(id, n, last)| {
                let s = &self.active[at(&self.active, id).expect("a picked sequence")];
                let start = s.done + ahead(&self.flight, id);
                PrefillJob { pages: &s.pages, start, tokens: &s.tokens[start..start + n], last }
            })
            .collect();
        let sampled: Vec<usize> =
            plan.rows.iter().map(|r| r.0).chain(picks.iter().filter(|p| p.2).map(|p| p.0)).filter_map(|id| at(&self.active, id)).collect();
        let masks: Vec<_> = sampled.iter().map(|&i| self.active[i].sampler.mask_handle()).collect();
        let on_device = sampled.iter().all(|&i| self.active[i].greedy && self.active[i].reply.logprobs.is_none());
        let (t, t0) = (std::time::Instant::now(), crate::monitor::clock());
        self.be.masks(masks);
        self.be.draws(!on_device);
        let l = self.be.launch(&rows, &jobs)?;
        drop((rows, jobs));
        Ok(Flight { l, rows: plan.rows.iter().map(|r| r.0).collect(), picks, used: plan.used, t, t0, on_device })
    }

    /// A launched step's samples (`ModelStep::finish`) taken: each decode row's token chosen and sent, each prompt chunk's tokens counted (a prompt's first token at its last), and the sequences
    /// that ended given their slot back; a sequence that holds a block the step could not store exactly is sent back instead (`after_step`).
    fn land(&mut self, f: Flight) -> Result<()> {
        let at = |a: &Vec<Seq>, id: u64| a.iter().position(|s| s.id == id);
        let (decoded, prompted) = self.be.finish(f.l)?;
        let n_rows = f.rows.len();
        if self.trace {
            eprintln!(
                "sched {:.4} step from {:.4}: {} rows, {} prompts {:?} of {} tokens, {:.1} ms; {} active, {} waiting",
                crate::monitor::clock(),
                f.t0,
                n_rows,
                f.picks.len(),
                f.picks.iter().map(|p| p.0).collect::<Vec<_>>(),
                f.used,
                f.t.elapsed().as_secs_f64() * 1e3,
                self.active.len(),
                self.waiting.len()
            );
        }
        let took = f.t.elapsed().as_nanos() as u64;
        if n_rows > 0 {
            self.stats.decode_ns += took;
            self.stats.steps += 1;
            self.stats.rows += n_rows as u64;
            self.stats.by_rows[n_rows.min(MAX_ROWS)] += 1;
        } else {
            self.stats.prefill_ns += took;
        }
        if !f.picks.is_empty() {
            self.stats.prefill_rounds += 1;
            self.stats.both += (n_rows > 0) as u64;
            // (the step's whole time, its rows' share too: a recompute is priced no cheaper than it is)
            self.stats.pass_ns += took;
            self.stats.pass_tokens += f.used as u64;
        }
        self.pass_tokens = f.used;
        // what the step could not store exactly: the output of the sequences that hold such a block is not used (they are sent back below)
        let hit = self.after_step()?;
        let tc = std::time::Instant::now();
        let mut ended: Vec<(u64, Finish)> = Vec::new();
        for (row, (&id, sample)) in f.rows.iter().zip(decoded).enumerate() {
            if hit.contains(&id) {
                continue;
            }
            let Some(i) = at(&self.active, id) else { continue };
            self.active[i].done += 1;
            self.publish(i)?;
            let (tok, lp) = self.pick(i, sample, row)?;
            if let Some(fin) = self.accept(i, tok, lp) {
                ended.push((id, fin));
            }
        }
        let mut row = n_rows;
        for (k, &(id, n, _)) in f.picks.iter().enumerate() {
            let sample = prompted.get(k).copied().flatten();
            let my_row = row;
            row += sample.is_some() as usize;
            if hit.contains(&id) {
                continue;
            }
            let Some(i) = at(&self.active, id) else { continue };
            self.active[i].done += n;
            self.publish(i)?;
            if let Some(sample) = sample {
                self.active[i].prefilling = false;
                self.mark_prompt(i)?;
                let (tok, lp) = self.pick(i, sample, my_row)?;
                if let Some(fin) = self.accept(i, tok, lp) {
                    ended.push((id, fin));
                }
            }
        }
        // (every row's token is chosen before a sequence gives its room back: the step's logits are the backend's until then)
        for (id, fin) in ended {
            if let Some(i) = at(&self.active, id) {
                self.finish(i, fin)?;
            }
        }
        self.take_back(&hit)?;
        self.stats.after_ns += tc.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// One step for every sequence that is decoding (each needs a page for its next token first).
    fn decode_step(&mut self) -> Result<()> {
        let tb = std::time::Instant::now();
        let ids: Vec<u64> = self.active.iter().filter(|s| !s.prefilling).map(|s| s.id).collect();
        for id in ids {
            let Some(s) = self.active.iter().find(|s| s.id == id) else { continue };
            let upto = s.done + 1;
            self.ensure_pages(id, upto)?;
        }
        let idx: Vec<usize> = self.active.iter().enumerate().filter(|(_, s)| !s.prefilling).map(|(i, _)| i).collect();
        if idx.is_empty() {
            return Ok(());
        }
        if let [i] = idx[..] {
            if self.draft_step(i, tb)? {
                return Ok(());
            }
        }
        let rows: Vec<Row> = idx.iter().map(|&i| Row::new(self.active[i].tokens[self.active[i].done], self.active[i].done, &self.active[i].pages)).collect();
        self.stats.before_ns += tb.elapsed().as_nanos() as u64;
        let t = std::time::Instant::now();
        let masks: Vec<_> = idx.iter().map(|&i| self.active[i].sampler.mask_handle()).collect();
        self.be.masks(masks);
        let samples = self.be.launch(&rows, &[]).and_then(|l| self.be.finish(l))?.0;
        self.stats.decode_ns += t.elapsed().as_nanos() as u64;
        self.stats.steps += 1;
        self.stats.rows += rows.len() as u64;
        self.stats.by_rows[rows.len().min(MAX_ROWS)] += 1;
        // (a row whose token could not be stored exactly has the step's output taken back below: its token is not used)
        let hit = self.after_step()?;
        let tc = std::time::Instant::now();
        let mut ended = Vec::new();
        for (row, (&i, sample)) in idx.iter().zip(samples).enumerate() {
            if hit.contains(&self.active[i].id) {
                continue;
            }
            self.active[i].done += 1;
            self.publish(i)?;
            let (tok, lp) = self.pick(i, sample, row)?;
            if let Some(f) = self.accept(i, tok, lp) {
                ended.push((i, f));
            }
        }
        // the indices shift as sequences are taken out: from the last
        for (i, f) in ended.into_iter().rev() {
            self.finish(i, f)?;
        }
        self.take_back(&hit)?;
        self.stats.after_ns += tc.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// The decode step of the one sequence `i` that decodes, where the step drafts (`ModelStep::draft`): the tokens a drafting round gives, each sent and added as a step's
    /// would be, up to the sequence's end. Not for a sequence that asks for log-probabilities or a least length (the round reads no row's whole logits): its step is the
    /// plain one. False: the step does not draft.
    fn draft_step(&mut self, i: usize, tb: std::time::Instant) -> Result<bool> {
        let s = &mut self.active[i];
        if s.reply.logprobs.is_some() || s.reply.params.min_tokens > 0 {
            return Ok(false);
        }
        let row = Row::new(s.tokens[s.done], s.done, &s.pages);
        let t = std::time::Instant::now();
        let Some(toks) = self.be.draft(&row, if s.greedy { None } else { Some(&mut s.sampler) })? else { return Ok(false) };
        self.stats.before_ns += t.duration_since(tb).as_nanos() as u64;
        self.stats.decode_ns += t.elapsed().as_nanos() as u64;
        (self.stats.steps, self.stats.rows) = (self.stats.steps + 1, self.stats.rows + 1);
        self.stats.by_rows[1] += 1;
        // (the cache holds the row's token and every token of the round but its last, which the next round takes)
        self.active[i].done += toks.len();
        self.publish(i)?;
        for tok in toks {
            if let Some(f) = self.accept(i, tok, None) {
                return self.finish(i, f).map(|_| true);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::sampler::Params;
    use std::sync::mpsc::{channel, Receiver};

    /// A model that is a function of the tokens in its pages: the token after a sequence is a hash of all of it, so a wrong page, a lost token or a stale cache entry changes every token after.
    /// It models the cache's extra memory too: a page that holds a token for which `heavy` is true needs one of `slabs` slabs (it is then raw: held exactly, and named heavy); a page that finds no slab is void: what it
    /// holds reads back wrong, and it is named lost (a token the scheduler used from a void page changes every token after it).
    pub struct Fake {
        pub store: Vec<[u32; PAGE]>,
        pub chunk: usize,
        pub released: usize,
        pub steps: usize,
        pub eos: Vec<i64>,
        pub heavy: fn(u32) -> bool,
        pub slabs: usize,
        pub slabs_used: usize,
        /// The first extra memory: a page that holds a heavy token takes a unit of it where there is one (it stays an ordinary page, kept for later prompts like any other), else a slab.
        pub chunks: usize,
        pub chunks_used: usize,
        /// Pages: 0 plain, 1 raw (holds a slab), 2 void, 3 plain and holding a unit of the first extra memory.
        pub state: Vec<u8>,
        pub pending: Lost,
        /// At this step the status's rings are said to have overflowed.
        pub overflow_at: Option<usize>,
        pub voided: usize,
        /// A decode step takes this long (to see when a token reaches its request).
        pub decode_delay: std::time::Duration,
        /// The logits of each row of the last step (`row_of` the token it chose), what `logits` gives; the GPU's largest 64 of them where `gpu_top` (else `top` has none: the draws read the row), and how often a row was read.
        pub rows: Vec<Vec<f32>>,
        pub gpu_top: bool,
        pub logit_reads: usize,
        /// The prompt tokens the model has run (a prompt that finds its blocks in the cache runs fewer).
        pub prefilled: Arc<std::sync::atomic::AtomicUsize>,
        /// A model that says these tokens after any prompt (then the end token 0), for the tests of what is made of a reply's text; `script_from` is the length of the prompt it was begun after.
        pub script: Option<Vec<u32>>,
        pub script_from: usize,
        /// How many sequences' prompt chunks `prefill_many` takes in one pass (1: one at a time), and the passes of more than one it has made.
        pub batch: usize,
        pub batches: usize,
        /// A round is one step (`one_step`): the prompt tokens it takes beside decode rows (None: what a chunk holds after them), and each step's rows and prompt tokens.
        pub one_step: bool,
        pub beside: Option<usize>,
        pub stepped: Vec<(usize, usize)>,
        /// The decode step that fails as a call of the driver does, and with what status (700: an illegal address, after which the context is lost; 1: an invalid value, after which it is not).
        pub fault: Option<(usize, i32)>,
    }

    /// A row of logits (2,048 tokens) in which `token` is the largest and the one after it the second.
    pub fn row_of(token: u32) -> Vec<f32> {
        let mut r = vec![-5.0f32; 2048];
        r[token as usize] = 10.0;
        r[token as usize + 1] = 5.0;
        r
    }

    pub fn next_of(toks: &[u32]) -> u32 {
        let mut h = 17u64;
        for &t in toks {
            h = h.wrapping_mul(1_315_423_911).wrapping_add(t as u64 + 7);
            h ^= h >> 13;
        }
        (h % 1000) as u32 + 1
    }

    impl Fake {
        pub fn new(pages: usize, chunk: usize) -> Fake {
            Fake {
                store: vec![[0; PAGE]; pages],
                chunk,
                released: 0,
                steps: 0,
                eos: vec![0],
                heavy: |_| false,
                slabs: usize::MAX,
                slabs_used: 0,
                chunks: 0,
                chunks_used: 0,
                state: vec![0; pages],
                pending: Lost::default(),
                overflow_at: None,
                voided: 0,
                decode_delay: std::time::Duration::ZERO,
                rows: Vec::new(),
                gpu_top: false,
                logit_reads: 0,
                prefilled: Default::default(),
                script: None,
                script_from: 0,
                batch: 1,
                batches: 0,
                one_step: false,
                beside: None,
                stepped: Vec::new(),
                fault: None,
            }
        }
        /// The token after `toks` (`n` of them): the hash, or the script's word for this place.
        fn said(&self, toks: &[u32], n: usize) -> u32 {
            match &self.script {
                Some(s) => s.get(n - self.script_from).copied().unwrap_or(0),
                None => next_of(toks),
            }
        }
        fn clear(&mut self, page: usize) {
            match self.state[page] {
                1 => self.slabs_used -= 1,
                3 => self.chunks_used -= 1,
                _ => {}
            }
            self.state[page] = 0;
        }
        fn write(&mut self, page: usize, off: usize, tok: u32) {
            if off == 0 {
                self.clear(page);
            }
            if self.state[page] == 2 {
                return;
            }
            self.store[page][off] = tok;
            if (self.heavy)(tok) && self.state[page] == 0 {
                if self.chunks_used < self.chunks {
                    self.chunks_used += 1;
                    self.state[page] = 3;
                } else if self.slabs_used < self.slabs {
                    self.slabs_used += 1;
                    self.state[page] = 1;
                    self.pending.raw.push(page as i32);
                } else {
                    self.state[page] = 2;
                    self.voided += 1;
                    self.pending.void.push(page as i32);
                }
            }
        }
        fn read(&self, pages: &[i32], n: usize) -> Vec<u32> {
            (0..n)
                .map(|p| {
                    let page = pages[p / PAGE] as usize;
                    let t = self.store[page][p % PAGE];
                    if self.state[page] == 2 {
                        t ^ 0xDEAD_0000
                    } else {
                        t
                    }
                })
                .collect()
        }
    }

    impl ModelStep for Fake {
        fn window(&self) -> usize {
            self.store.len() * PAGE
        }
        fn chunk(&self) -> usize {
            self.chunk
        }
        fn pages(&self) -> usize {
            self.store.len()
        }
        fn eos(&self) -> Vec<i64> {
            self.eos.clone()
        }
        fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
            self.prefilled.fetch_add(tokens.len(), Relaxed);
            for (k, &t) in tokens.iter().enumerate() {
                let p = start + k;
                self.write(pages[p / PAGE] as usize, p % PAGE, t);
            }
            if last {
                self.script_from = start + tokens.len();
            }
            let sample = last.then(|| Sample { token: self.said(&self.read(pages, start + tokens.len()), start + tokens.len()), logit: 0.0, logprob: None });
            if let Some(s) = &sample {
                self.rows = vec![row_of(s.token)];
            }
            Ok(sample)
        }
        fn prefill_rows(&self) -> usize {
            self.batch
        }
        fn prefill_many(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
            let (mut out, mut rows) = (Vec::new(), Vec::new());
            for j in jobs {
                let s = self.prefill(j.pages, j.start, j.tokens, j.last)?;
                rows.extend(s.map(|x| row_of(x.token)));
                out.push(s);
            }
            if !rows.is_empty() {
                self.rows = rows;
            }
            self.batches += (jobs.len() > 1) as usize;
            Ok(out)
        }
        fn one_step(&self) -> bool {
            self.one_step
        }
        fn prompt_budget(&self, rows: usize) -> usize {
            self.beside.unwrap_or(self.chunk.saturating_sub(rows))
        }
        fn round(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<(Vec<Sample>, Vec<Option<Sample>>)> {
            assert!(self.one_step, "a step of both, of a model whose rounds are two passes");
            let tokens: usize = jobs.iter().map(|j| j.tokens.len()).sum();
            assert!(rows.len() + tokens <= self.chunk, "{} rows and {tokens} prompt tokens in a step that holds {}", rows.len(), self.chunk);
            self.stepped.push((rows.len(), tokens));
            let decoded = if rows.is_empty() { Vec::new() } else { self.decode(rows)? };
            let mut all = if rows.is_empty() { Vec::new() } else { std::mem::take(&mut self.rows) };
            let mut prompted = Vec::new();
            for j in jobs {
                let s = self.prefill(j.pages, j.start, j.tokens, j.last)?;
                all.extend(s.map(|x| row_of(x.token)));
                prompted.push(s);
            }
            self.rows = all;
            Ok((decoded, prompted))
        }
        fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
            std::thread::sleep(self.decode_delay);
            self.steps += 1;
            if let Some((_, status)) = self.fault.filter(|f| f.0 == self.steps) {
                return Err(crate::ffi::Error::of(
                    crate::ffi::Kind::of_status(status),
                    format!("cuLaunchKernel: the made-up model's step {} failed ({status})", self.steps),
                ));
            }
            if self.overflow_at == Some(self.steps) {
                self.pending.overflow = true;
            }
            for r in rows {
                self.write(r.pages[r.pos / PAGE] as usize, r.pos % PAGE, r.token);
            }
            let out: Vec<Sample> =
                rows.iter().map(|r| Sample { token: self.said(&self.read(r.pages, r.pos + 1), r.pos + 1), logit: 0.0, logprob: None }).collect();
            self.rows = out.iter().map(|s| row_of(s.token)).collect();
            Ok(out)
        }
        fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
            self.logit_reads += 1;
            self.rows.get(row).cloned().ok_or_else(|| crate::ffi::Error("no logits".into()))
        }
        fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
            if !self.gpu_top {
                return None;
            }
            let r = self.rows.get(row)?;
            let mut by: Vec<u32> = (0..r.len() as u32).collect();
            by.sort_by(|&a, &b| r[b as usize].partial_cmp(&r[a as usize]).unwrap().then(a.cmp(&b)));
            Some(by[..64].iter().map(|&t| (t, r[t as usize])).collect())
        }
        // (as a GPU's: those tokens alone, not a read of the row)
        fn gather(&mut self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
            let r = self.rows.get(row).ok_or_else(|| crate::ffi::Error("no logits".into()))?;
            Ok(tokens.iter().filter_map(|&t| r.get(t as usize).map(|&x| (t, x))).collect())
        }
        fn release(&mut self, pages: &[i32]) -> Result<()> {
            self.released += pages.len();
            for &p in pages {
                self.clear(p as usize);
            }
            Ok(())
        }
        fn lost(&mut self) -> Lost {
            std::mem::take(&mut self.pending)
        }
    }

    /// What the model says after `prompt`, `n` tokens (it stops at `eos` unless told to go on).
    pub fn reference(prompt: &[u32], n: usize) -> Vec<u32> {
        let mut t = prompt.to_vec();
        for _ in 0..n {
            let x = next_of(&t);
            t.push(x);
        }
        t[prompt.len()..].to_vec()
    }

    fn reply(max_new: usize) -> Reply {
        Reply { params: Params { temperature: 0.0, ..Params::default() }, max_new, ignore_eos: false, logprobs: None, stop_ids: Vec::new() }
    }

    pub fn add<B: ModelStep>(s: &mut Sched<B>, prompt: Vec<u32>, max_new: usize) -> (Receiver<Ev>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let abort = Arc::new(AtomicBool::new(false));
        s.submit(Submit {
            prompt,
            reply: reply(max_new),
            out: tx,
            abort: abort.clone(),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: Vec::new(),
        });
        (rx, abort)
    }

    /// The tokens a request got and how its sequence ended.
    pub fn collect(rx: &Receiver<Ev>) -> (Vec<u32>, Finish) {
        let mut out = Vec::new();
        loop {
            match rx.recv().expect("the sequence ends") {
                Ev::Token(t) => out.push(t),
                Ev::Lp(_) => {}
                Ev::End(f) => return (out, f.expect("no error")),
            }
        }
    }

    pub fn run<B: ModelStep>(s: &mut Sched<B>) {
        let mut guard = 0;
        while !s.idle() {
            s.tick().unwrap();
            guard += 1;
            assert!(guard < 100_000, "the scheduler does not finish");
        }
    }

    fn prompt(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 1 + (seed * 7919 + i * 104_729) % 4000).collect()
    }

    /// Prompts run several to a pass give every sequence the tokens it has alone, whatever the mix of lengths, a shared start, sampling from the logits' rows, and a pool too small for all of them.
    #[test]
    fn prompts_run_together_give_the_tokens_they_give_alone() {
        for (pages, chunk, batch) in [(64, 32, 8), (64, 200, 8), (64, 200, 3), (14, 64, 8)] {
            let mut be = Fake::new(pages, chunk);
            be.batch = batch;
            be.gpu_top = batch == 3;
            let mut s = Sched::new(be, 8);
            let mut prompts: Vec<Vec<u32>> = (0..8).map(|k| prompt(k, 3 + 11 * k as usize)).collect();
            prompts.push(prompts[3].clone()); // (one that begins as an earlier one does: its pages are found)
            prompts.push(prompt(50, 90)); // (longer than a chunk where the chunk is 32)
            let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 20).0).collect();
            run(&mut s);
            for (p, rx) in prompts.iter().zip(&rxs) {
                let (got, f) = collect(rx);
                assert_eq!(got, reference(p, 20), "{pages} pages, chunk {chunk}, {batch} to a pass");
                assert_eq!(f, Finish::Length);
            }
            assert!(s.be.batches > 0, "no pass took more than one prompt ({pages} pages, chunk {chunk}, {batch} to a pass)");
        }
    }

    /// A round that is one step (`ModelStep::one_step`): every sequence gives the tokens it gives alone, whatever the mix of lengths, a shared start, sampling from the logits' rows and a pool
    /// too small for all of them; some step held both rows that decode and a prompt's chunk, none held more than a chunk's tokens, and none more prompt tokens beside rows than the budget.
    #[test]
    fn a_round_of_one_step_gives_the_tokens_each_sequence_gives_alone() {
        for (pages, chunk, batch, beside) in [(64, 32, 8, None), (64, 200, 8, Some(24)), (64, 200, 3, Some(1)), (14, 64, 8, None)] {
            let mut be = Fake::new(pages, chunk);
            (be.one_step, be.batch, be.beside) = (true, batch, beside);
            be.gpu_top = batch == 3;
            let mut s = Sched::new(be, 8);
            let mut prompts: Vec<Vec<u32>> = (0..8).map(|k| prompt(k, 3 + 11 * k as usize)).collect();
            prompts.push(prompts[3].clone());
            prompts.push(prompt(50, 90));
            let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 20).0).collect();
            run(&mut s);
            for (p, rx) in prompts.iter().zip(&rxs) {
                assert_eq!(collect(rx), (reference(p, 20), Finish::Length), "{pages} pages, chunk {chunk}, {batch} to a pass, {beside:?} beside");
            }
            let st = &s.be.stepped;
            assert!(st.iter().any(|&(r, t)| r > 0 && t > 0), "no step held rows and a prompt's chunk ({pages} pages, chunk {chunk})");
            assert!(st.iter().all(|&(r, t)| r + t <= chunk && r + t > 0));
            if let Some(b) = beside {
                assert!(st.iter().all(|&(r, t)| r == 0 || t <= b), "more than {b} prompt tokens beside rows: {st:?}");
            }
        }
    }

    /// The same on a slot step, the description executor's kind of cache: every reply as alone, no two parts of a step in one slot, every slot empty at the end.
    #[test]
    fn a_round_of_one_step_on_slots_gives_the_tokens_each_sequence_gives_alone() {
        for (slots, users, batch, chunk, beside) in
            [(1, 4, 1, 16, None), (3, 4, 3, 16, Some(5)), (4, 8, 4, 64, None), (8, 8, 8, 7, Some(2)), (24, 64, 24, 512, Some(64))]
        {
            let mut f = SlotFake::new(slots, 256, chunk);
            (f.one_step, f.batch, f.beside) = (true, batch, beside);
            let mut s = Sched::new(f, users);
            let reqs: Vec<(Vec<u32>, usize)> = (0..slots.max(9) + 3).map(|k| (prompt(k as u32 + 1, 3 + 11 * (k % 9)), 5 + 3 * (k % 9))).collect();
            let rx: Vec<_> = reqs.iter().map(|(p, n)| add(&mut s, p.clone(), *n).0).collect();
            run(&mut s);
            for ((p, n), rx) in reqs.iter().zip(&rx) {
                assert_eq!(collect(rx), (reference(p, *n), Finish::Length), "slots {slots}, batch {batch}, chunk {chunk}, {beside:?} beside");
            }
            assert!(s.be.most <= slots && s.be.rooms.iter().all(Vec::is_empty) && s.be.emptied == reqs.len());
            assert_eq!(s.available(), slots);
            let st = &s.be.stepped;
            if slots > 1 {
                assert!(st.iter().any(|&(r, t)| r > 0 && t > 0), "no step held rows and a prompt's chunk (slots {slots}, chunk {chunk})");
            }
            if let Some(b) = beside {
                assert!(st.iter().all(|&(r, t)| r == 0 || t <= b), "more than {b} prompt tokens beside rows: {st:?}");
            }
        }
    }

    /// A request with its own reply's asks.
    fn add_reply<B: ModelStep>(s: &mut Sched<B>, prompt: Vec<u32>, reply: Reply) -> Receiver<Ev> {
        let (tx, rx) = channel();
        s.submit(Submit {
            prompt,
            reply,
            out: tx,
            abort: Arc::new(AtomicBool::new(false)),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: Vec::new(),
        });
        rx
    }

    /// A slot fake whose round is one step and whose steps are launched before the step before them is read (`overlaps`).
    fn launching(slots: usize, chunk: usize, beside: Option<usize>) -> SlotFake {
        let mut f = SlotFake::new(slots, 256, chunk);
        (f.one_step, f.batch, f.beside, f.overlap) = (true, slots, beside, true);
        f
    }

    /// #138, a step's launch and its finish (docs/design/runtime.md section 5, "Launch and finish"): where the step overlaps, each round's step is launched before the step before it
    /// lands, its rows reading the tokens that step chose on the GPU (the fake: from the head rows of its last launch). Every reply is the one it gives alone, whatever comes in and
    /// when; no step's rows are read after a later launch (the fake panics); every slot is empty at the end. With `overlapping(false)` no step is launched before the one before lands.
    #[test]
    fn a_step_launched_before_the_one_before_lands_gives_the_replies_alone() {
        for (slots, chunk, beside, on) in [(1, 16, None, true), (3, 16, Some(5), true), (8, 7, Some(2), true), (4, 64, None, true), (4, 64, None, false)] {
            let mut s = Sched::new(launching(slots, chunk, beside), slots).overlapping(on);
            let reqs: Vec<(Vec<u32>, usize)> = (0..slots + 6).map(|k| (prompt(k as u32 + 1, 3 + 11 * (k % 9)), 1 + 3 * (k % 9))).collect();
            // (half the requests at once, the rest one a round while those run)
            let half = reqs.len() / 2;
            let mut rx: Vec<_> = reqs[..half].iter().map(|(p, n)| add(&mut s, p.clone(), *n).0).collect();
            for (p, n) in &reqs[half..] {
                s.tick().unwrap();
                rx.push(add(&mut s, p.clone(), *n).0);
            }
            run(&mut s);
            for ((p, n), rx) in reqs.iter().zip(&rx) {
                assert_eq!(collect(rx), (reference(p, *n), Finish::Length), "slots {slots}, chunk {chunk}, {beside:?} beside, overlap {on}");
            }
            assert!(s.be.rooms.iter().all(Vec::is_empty) && s.be.emptied == reqs.len() && s.be.out.is_empty() && s.idle());
            assert_eq!(s.be.ahead > 0, on, "steps launched ahead: {} (slots {slots}, overlap {on})", s.be.ahead);
            // (a sequence whose last token by its length is in flight has no row in the next step: no room holds a token past its reply)
            assert_eq!(s.be.ran, reqs.iter().map(|(p, _)| p.len()).sum::<usize>());
        }
    }

    /// A sequence that ends where a launched step chose its end (a stop token) has had its row in the next step run too: that row's output is not taken, and the reply is the
    /// one it gives alone, ended by the stop. A sequence that draws from its logits (not its largest) has its step land before the next is launched: its rows are read on the
    /// host; the replies are those of the same requests with no step launched ahead.
    #[test]
    fn a_stop_or_a_draw_in_a_launched_step() {
        let greedy = |n: usize, stop: Vec<u32>| Reply { stop_ids: stop, ..reply(n) };
        let mut s = Sched::new(launching(4, 32, None), 4);
        let ps: Vec<Vec<u32>> = (0..4).map(|k| prompt(k + 3, 9 + 5 * k as usize)).collect();
        let full: Vec<Vec<u32>> = ps.iter().map(|p| reference(p, 30)).collect();
        // (each stops at its reply's 6th + k token, where that token first comes)
        let stops: Vec<u32> = full.iter().enumerate().map(|(k, r)| r[5 + k]).collect();
        let rx: Vec<_> = ps.iter().zip(&stops).map(|(p, &t)| add_reply(&mut s, p.clone(), greedy(30, vec![t]))).collect();
        run(&mut s);
        for ((r, &t), rx) in full.iter().zip(&stops).zip(&rx) {
            let upto = r.iter().position(|&x| x == t).expect("the stop is in the reply");
            assert_eq!(collect(rx), (r[..upto].to_vec(), Finish::Stop));
        }
        assert!(s.be.ahead > 0 && s.be.rooms.iter().all(Vec::is_empty));
        // drawn: the same with and without launching ahead, and nothing launched ahead while a draw's step was out
        let drawn = |on: bool| {
            let mut s = Sched::new(launching(4, 32, None), 4).overlapping(on);
            let sampled = Reply { params: Params { temperature: 0.9, seed: 7, ..Params::default() }, ..reply(12) };
            let rx: Vec<_> = ps.iter().map(|p| add_reply(&mut s, p.clone(), sampled.clone())).collect();
            run(&mut s);
            (rx.iter().map(collect).collect::<Vec<_>>(), s.be.ahead)
        };
        let (on, ahead) = drawn(true);
        assert_eq!((on, ahead), (drawn(false).0, 0));
    }

    /// A conversation's next turn goes on in the slot its last turn finished in where steps are launched ahead too, with the same replies and the same tokens found; and a
    /// scheduler that fails while a step is out gives that step up (`abandon`), every request ending with the error and every slot empty.
    #[test]
    fn launched_steps_keep_a_conversation_and_are_given_up_on_a_failure() {
        let mut runs = Vec::new();
        for on in [false, true] {
            let mut f = launching(2, 16, None);
            (f.window, f.keeps) = (512, true);
            let mut s = Sched::new(f, 2).overlapping(on);
            let p1 = prompt(1, 40);
            let r1 = turn(&mut s, &p1, 8);
            let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 10)[..]].concat();
            let r2 = turn(&mut s, &p2, 8);
            assert_eq!((r1, r2), (reference(&p1, 8), reference(&p2, 8)));
            runs.push((s.be.resumed.clone(), s.stats.hit, s.be.ahead > 0));
        }
        assert_eq!((runs[0].0.clone(), runs[0].1), (runs[1].0.clone(), runs[1].1));
        assert_eq!((runs[0].2, runs[1].2), (false, true));
        let mut s = Sched::new(launching(3, 16, None), 3);
        let rx: Vec<_> = (0..3).map(|k| add(&mut s, prompt(k + 1, 20), 50).0).collect();
        for _ in 0..6 {
            s.tick().unwrap();
        }
        assert!(s.flight.is_some());
        s.fail(&crate::ffi::Error("a made-up failure".into()));
        assert!(s.idle() && s.be.abandoned == 1 && s.be.out.is_empty());
        for rx in &rx {
            while !matches!(rx.recv().expect("an end"), Ev::End(Err(_))) {}
        }
    }

    /// #138: the models that ran one request at a time through a loop of their own (`server::work_gen`, gone) are the scheduler's steps of one slot. A drafted model's decode
    /// step is a drafting round (`ModelStep::draft`), its tokens each sent and added as a step's would be; a request that asks for log-probabilities or a least length has plain
    /// steps (a round reads no row's whole logits); a request that draws gives the round its sampler. Every reply is the one it gives alone, ended by its length or a stop.
    #[test]
    fn a_drafted_model_is_a_step_of_one_slot_whose_decode_is_a_round() {
        let mut f = SlotFake::new(1, 256, 16);
        f.drafts = 3;
        let mut s = Sched::new(f, 4);
        assert_eq!(s.users(), 1);
        let ps: Vec<Vec<u32>> = (0..4).map(|k| prompt(k + 11, 7 + 9 * k as usize)).collect();
        let full = reference(&ps[1], 30);
        let rx = [
            add(&mut s, ps[0].clone(), 25).0,
            add_reply(&mut s, ps[1].clone(), Reply { stop_ids: vec![full[13]], ..reply(30) }),
            add_reply(&mut s, ps[2].clone(), Reply { logprobs: Some(0), ..reply(9) }),
            add_reply(&mut s, ps[3].clone(), Reply { params: Params { temperature: 0.5, top_k: 1, ..Params::default() }, ..reply(11) }),
        ];
        run(&mut s);
        assert_eq!(collect(&rx[0]), (reference(&ps[0], 25), Finish::Length));
        let stop = full.iter().position(|&t| t == full[13]).expect("the stop");
        assert_eq!(collect(&rx[1]), (full[..stop].to_vec(), Finish::Stop));
        assert_eq!(collect(&rx[2]), (reference(&ps[2], 9), Finish::Length));
        assert_eq!(collect(&rx[3]), (reference(&ps[3], 11), Finish::Length));
        // (the plain steps are the log-probabilities' request's alone: 8 tokens after its first; the rounds gave the others' tokens, more than one a round)
        assert_eq!(s.be.steps.len(), 8);
        assert!(s.be.rounds > 0 && s.be.rounds < 24 + stop + 10, "{} rounds", s.be.rounds);
        assert!(s.be.drawn_rounds > 0 && s.be.rooms[0].is_empty() && s.be.emptied == 4);
    }

    /// What the one step is for: while a long prompt comes in, every sequence that is decoding still gets a token every round (the prompt's chunk rides in their step, as many tokens as the
    /// budget beside them), and the prompt takes the rounds its length over that budget asks, with no decode step between its chunks to wait for.
    #[test]
    fn a_prompt_runs_in_the_steps_of_the_sequences_that_decode() {
        let mut f = SlotFake::new(4, 4096, 64);
        (f.one_step, f.batch, f.beside) = (true, 4, Some(16));
        let mut s = Sched::new(f, 4);
        let rx: Vec<_> = (0..3).map(|k| add(&mut s, prompt(k + 1, 8), 400).0).collect();
        // (the three prompts in one step of prompts alone, then a few steps of their rows)
        for _ in 0..4 {
            s.tick().unwrap();
        }
        assert_eq!(s.be.stepped[0], (0, 24));
        assert!(s.be.stepped[1..].iter().all(|&st| st == (3, 0)), "{:?}", s.be.stepped);
        let from = s.be.stepped.len();
        let late = add(&mut s, prompt(9, 100), 5).0;
        // 100 tokens at 16 a step: 7 steps, each with the three rows; the 8th has its first decode row with theirs
        for _ in 0..8 {
            s.tick().unwrap();
        }
        assert_eq!(s.be.stepped[from..], [(3, 16), (3, 16), (3, 16), (3, 16), (3, 16), (3, 16), (3, 4), (4, 0)]);
        assert_eq!(s.stats.both, 7);
        run(&mut s);
        assert_eq!(collect(&late), (reference(&prompt(9, 100), 5), Finish::Length));
        for (k, r) in rx.iter().enumerate() {
            assert_eq!(collect(r), (reference(&prompt(k as u32 + 1, 8), 400), Finish::Length));
        }
    }

    /// The metrics see what a sequence did: its first token and the gaps after it, its tokens, how it ended, and the cache's pages while it holds some (the counters are the process's, other tests add to them: at least these).
    #[test]
    fn a_sequence_is_counted_in_the_metrics() {
        use crate::ops;
        let fig = ops::figures();
        let (first, gaps, whole) = (fig.ttft.count(), fig.token_gap.count(), fig.request_time.count());
        let mut s = Sched::new(Fake::new(64, 32), 4);
        let (rx, _) = add(&mut s, prompt(2, 20), 6);
        s.tick().unwrap();
        // (the gauges are the process's, and the other tests' schedulers write them too: this one's written again just before it is read, until a read sees its own)
        let seen = (0..1000).any(|_| {
            s.gauges();
            fig.kv_pages.load(Relaxed) >= 64 && fig.kv_pages_held.load(Relaxed) > 0
        });
        assert!(seen, "the sequence holds pages while it runs");
        run(&mut s);
        let (tokens, f) = collect(&rx);
        assert_eq!((tokens.len(), f), (6, Finish::Length));
        assert!(fig.ttft.count() > first, "the first token was not counted");
        assert!(fig.token_gap.count() >= gaps + 5, "the gaps of the five tokens after the first were not counted");
        assert!(fig.request_time.count() > whole, "the end was not counted");
        let text = ops::render(&ops::State { model: "m", ready: true, life: &ops::Life::default(), engine: true });
        assert!(text.contains("glyd_sequences_finished_total{reason=\"length\"} "), "{text}");
        // a request that is asked to end before it runs is counted as stopped
        let (rx, abort) = add(&mut s, prompt(3, 20), 6);
        abort.store(true, Relaxed);
        run(&mut s);
        assert_eq!(collect(&rx).1, Finish::Stopped);
    }

    /// With a collector configured, a sequence run on the scheduler comes out of it as a request span with its queue, prompt pass and decode beneath (the process's exporter is the test's own collector: other
    /// tests' sequences send spans there too, so the sequence is found by its prompt's length).
    #[test]
    fn a_sequence_is_traced_to_a_local_collector() {
        use crate::otel::{self, Config, Exporter};
        let col = otel::testing::collector();
        otel::install(Exporter::start(Config { url: col.url.clone(), headers: vec![], service: "glyd-sched-test".into() }));
        let _ = crate::runtime::context::Context::process().figures.model.set("fake/model".into());
        let mut s = Sched::new(Fake::new(64, 32), 4);
        let (rx, _) = add(&mut s, prompt(5, 57), 6);
        run(&mut s);
        let (tokens, f) = collect(&rx);
        assert_eq!((tokens.len(), f), (6, Finish::Length));
        otel::flush(std::time::Duration::from_secs(5));
        let all: Vec<serde_json::Value> = col
            .bodies
            .lock()
            .unwrap()
            .iter()
            .flat_map(|(b, _)| serde_json::from_str::<serde_json::Value>(b).unwrap()["resourceSpans"][0]["scopeSpans"][0]["spans"].as_array().unwrap().clone())
            .collect();
        let attr = |sp: &serde_json::Value, k: &str| sp["attributes"].as_array().and_then(|a| a.iter().find(|x| x["key"] == k).map(|x| x["value"].clone()));
        let root = all
            .iter()
            .find(|sp| sp["name"] == "glyd.request" && attr(sp, "glyd.prompt_tokens").is_some_and(|v| v["intValue"] == "57"))
            .expect("the sequence's request span");
        assert_eq!(
            (attr(root, "glyd.generated_tokens").unwrap()["intValue"].as_str(), attr(root, "glyd.finish_reason").unwrap()["stringValue"].as_str()),
            (Some("6"), Some("length"))
        );
        let parts: Vec<&str> =
            all.iter().filter(|sp| sp["traceId"] == root["traceId"] && sp["parentSpanId"] == root["spanId"]).map(|sp| sp["name"].as_str().unwrap()).collect();
        assert_eq!(parts, ["glyd.queue", "glyd.prefill", "glyd.decode"], "{parts:?}");
        // the parts end where the next begins, inside the whole
        let t = |sp: &serde_json::Value, k: &str| sp[k].as_str().unwrap().parse::<u64>().unwrap();
        let kids: Vec<_> = all.iter().filter(|sp| sp["parentSpanId"] == root["spanId"]).collect();
        assert!(t(kids[0], "startTimeUnixNano") >= t(root, "startTimeUnixNano") && t(kids[2], "endTimeUnixNano") <= t(root, "endTimeUnixNano"));
        assert_eq!((t(kids[0], "endTimeUnixNano"), t(kids[1], "endTimeUnixNano")), (t(kids[1], "startTimeUnixNano"), t(kids[2], "startTimeUnixNano")));
    }

    #[test]
    fn the_first_token_goes_out_before_the_decode_step_of_its_round() {
        let mut be = Fake::new(64, 32);
        be.decode_delay = std::time::Duration::from_millis(400);
        let mut s = Sched::new(be, 4);
        let (rx, _) = add(&mut s, prompt(1, 20), 3);
        let t0 = std::time::Instant::now();
        let first = std::thread::spawn(move || (rx.recv().expect("a first token"), t0.elapsed(), rx));
        run(&mut s);
        let (ev, at, rx) = first.join().unwrap();
        assert!(matches!(ev, Ev::Token(_)));
        assert!(at < std::time::Duration::from_millis(300), "the first token came {at:?} after the request: it waited for a decode step");
        let (rest, f) = collect(&rx);
        assert_eq!(rest.len(), 2);
        assert_eq!(f, Finish::Length);
    }

    #[test]
    fn sequences_together_as_each_alone() {
        let mut s = Sched::new(Fake::new(64, 32), 8);
        let prompts: Vec<Vec<u32>> = (0..8).map(|k| prompt(k, 5 + 13 * k as usize)).collect();
        let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 40).0).collect();
        run(&mut s);
        for (p, rx) in prompts.iter().zip(&rxs) {
            let (got, f) = collect(rx);
            assert_eq!(got, reference(p, 40));
            assert_eq!(f, Finish::Length);
        }
        // every page is back (the full ones cached, the rest cleared)
        assert_eq!(s.available(), 64);
    }

    #[test]
    fn pages_run_out_and_every_stream_goes_on_whole() {
        // 3 sequences of 20 tokens, each to grow to 50 (4 pages): the 6 pages hold one of them and part of the next
        let mut s = Sched::new(Fake::new(6, 16), 8);
        let prompts: Vec<Vec<u32>> = (0..3).map(|k| prompt(k + 20, 20)).collect();
        let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 30).0).collect();
        run(&mut s);
        assert!(s.stats.preempted > 0, "the pages were enough");
        for (p, rx) in prompts.iter().zip(&rxs) {
            let (got, f) = collect(rx);
            assert_eq!(got.len(), 30, "all of max_tokens");
            assert_eq!(got, reference(p, 30), "no gap and no repeat");
            assert_eq!(f, Finish::Length);
        }
        assert_eq!(s.available(), 6);
    }

    #[test]
    fn a_prompt_that_begins_as_an_earlier_one_finds_its_pages() {
        let mut s = Sched::new(Fake::new(32, 64), 4);
        let a = prompt(3, 100);
        let ra = add(&mut s, a.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&ra).0, reference(&a, 10));
        let mut b = a[..70].to_vec();
        b.extend(prompt(9, 30));
        let hits = s.stats.hit;
        let rb = add(&mut s, b.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&rb).0, reference(&b, 10));
        assert_eq!(s.stats.hit - hits, 64, "four pages of the first 70 tokens");
        // the same prompt again: all but the last page (its last token is run for the logits)
        let hits = s.stats.hit;
        let rc = add(&mut s, a.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&rc).0, reference(&a, 10));
        assert!(s.stats.hit - hits >= 96);
    }

    #[test]
    fn a_request_that_is_done_gives_its_pages_back() {
        let mut s = Sched::new(Fake::new(16, 32), 4);
        let (rx, abort) = add(&mut s, prompt(1, 40), 1000);
        for _ in 0..5 {
            s.tick().unwrap();
        }
        assert!(s.available() < 16);
        abort.store(true, Relaxed);
        run(&mut s);
        assert_eq!(collect(&rx).1, Finish::Stopped);
        assert_eq!(s.available(), 16);
        // a request that is gone before its turn
        let (rx, abort) = add(&mut s, prompt(2, 40), 10);
        abort.store(true, Relaxed);
        run(&mut s);
        assert_eq!(collect(&rx), (vec![], Finish::Stopped));
    }

    #[test]
    fn the_end_token_ends_a_sequence_unless_told_to_go_on() {
        let p = prompt(5, 30);
        let r = reference(&p, 50);
        let at = r.len() / 2;
        let mut s = Sched::new(Fake::new(32, 32), 4);
        s.eos = vec![r[at] as i64];
        let (rx, _) = add(&mut s, p.clone(), 50);
        run(&mut s);
        let (got, f) = collect(&rx);
        let first = r.iter().position(|&t| t == r[at]).unwrap();
        assert_eq!((got, f), (r[..first].to_vec(), Finish::Stop));
        let (tx, rx) = channel();
        s.submit(Submit {
            prompt: p.clone(),
            reply: Reply { ignore_eos: true, ..reply(50) },
            out: tx,
            abort: Arc::new(AtomicBool::new(false)),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: Vec::new(),
        });
        run(&mut s);
        assert_eq!(collect(&rx), (r, Finish::Length));
    }

    #[test]
    fn a_prompt_that_cannot_fit_is_refused_and_the_rest_go_on() {
        let mut s = Sched::new(Fake::new(4, 32), 4);
        let (big, _) = add(&mut s, prompt(1, 100), 5);
        let (ok, _) = add(&mut s, prompt(2, 10), 5);
        run(&mut s);
        // (the request's own doing: a server answers it 400, not 500)
        assert!(matches!(big.recv().unwrap(), Ev::End(Err(e)) if e.kind == Kind::Request));
        assert_eq!(collect(&ok).0, reference(&prompt(2, 10), 5));
    }
    /// The figures of /stats are one JSON object: a development build's name what the cache lost, a release build's are the counts a user reads and no more.
    #[test]
    fn stats_are_json_with_the_losses() {
        let v = glyd_gpu::json::parse(&stats_json()).expect("valid JSON");
        let user = ["prompt_tokens", "prefix_hit_tokens", "prefix_hit_rate_percent", "preemptions", "steps", "rows", "running", "waiting"];
        let dev = ["before_ms", "lost_blocks", "heavy_blocks", "taken_back", "ended_for_loss", "prefill_rounds"];
        for k in user {
            assert!(v.get(k).is_some(), "no {k}");
        }
        for k in dev {
            assert_eq!(v.get(k).is_some(), cfg!(feature = "dev"), "{k}");
        }
    }

    /// The tokens a request got and how its sequence ended, an error included.
    fn outcome(rx: &Receiver<Ev>) -> (Vec<u32>, std::result::Result<Finish, String>) {
        let mut out = Vec::new();
        loop {
            match rx.recv().expect("the sequence ends") {
                Ev::Token(t) => out.push(t),
                Ev::Lp(_) => {}
                Ev::End(f) => return (out, f.map_err(|e| e.msg)),
            }
        }
    }

    fn heavy_quarter(t: u32) -> bool {
        t.is_multiple_of(4)
    }

    fn heavy_all(_: u32) -> bool {
        true
    }

    /// The extra memory is too small for the sequences together: sequences hold blocks it could not store exactly, their steps are taken back, and every stream is what it would be alone, whole, with no token from a lost block.
    #[test]
    fn a_full_extra_memory_takes_steps_back_and_every_stream_goes_on_whole() {
        let mut f = Fake::new(64, 16);
        f.heavy = heavy_quarter;
        f.slabs = 10;
        let mut s = Sched::new(f, 4);
        let prompts: Vec<Vec<u32>> = (0..4).map(|k| prompt(k + 40, 60)).collect();
        let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 20).0).collect();
        run(&mut s);
        assert!(s.be.voided > 0, "the extra memory was enough: the test says nothing");
        assert!(s.stats.taken_back > 0);
        for (p, rx) in prompts.iter().zip(&rxs) {
            let (got, f) = collect(rx);
            assert_eq!(got, reference(p, 20), "a token came from a lost block, or the stream has a gap");
            assert_eq!(f, Finish::Length);
        }
        assert_eq!(s.be.slabs_used, 0, "the extra memory the sequences held is back");
        assert_eq!(s.available(), 64);
    }

    /// The same where the blocks run out while the sequences decode (short prompts, long replies, a new block for every 16 tokens): a row whose token was stored in a lost block has its output taken back.
    #[test]
    fn a_full_extra_memory_during_decode_steps_takes_the_rows_back() {
        let mut f = Fake::new(64, 16);
        f.heavy = heavy_all;
        f.slabs = 10;
        let mut s = Sched::new(f, 4);
        let prompts: Vec<Vec<u32>> = (0..4).map(|k| prompt(k + 80, 10)).collect();
        let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 100).0).collect();
        run(&mut s);
        assert!(s.be.voided > 0, "the extra memory was enough: the test says nothing");
        for (p, rx) in prompts.iter().zip(&rxs) {
            let (got, f) = collect(rx);
            assert_eq!(got, reference(p, 100), "a token came from a lost block, or the stream has a gap");
            assert_eq!(f, Finish::Length);
        }
        assert_eq!(s.be.slabs_used, 0);
        assert_eq!(s.available(), 64);
    }

    /// Requests that fit the extra memory alone are never ended for the others' use of it: however often they are lost while others run, they end up running alone, and are served.
    #[test]
    fn requests_that_fit_alone_are_served_however_the_others_use_the_extra_memory() {
        let mut f = Fake::new(64, 16);
        f.heavy = heavy_all;
        f.slabs = 7;
        let mut s = Sched::new(f, 4);
        // each needs 6 blocks: it fits alone (7), two never fit together
        let prompts: Vec<Vec<u32>> = (0..8).map(|k| prompt(k + 100, 70)).collect();
        let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 20).0).collect();
        run(&mut s);
        assert!(s.stats.taken_back > 3, "taken back {}: the test says nothing", s.stats.taken_back);
        for (p, rx) in prompts.iter().zip(&rxs) {
            assert_eq!(outcome(rx), (reference(p, 20), Ok(Finish::Length)), "a request that fits alone was ended, or has a wrong token");
        }
        assert_eq!(s.stats.ended_lost, 0);
        assert_eq!(s.be.slabs_used, 0);
    }

    /// What finished prompts keep in the cache holds extra memory too: a request that is lost while others run, and again alone because of what they left, runs a third time with the cache given up, and is served.
    #[test]
    fn pages_kept_for_later_prompts_do_not_end_a_request_that_fits_alone() {
        let mut f = Fake::new(64, 16);
        f.heavy = heavy_all;
        f.chunks = 7;
        f.slabs = 0;
        let mut s = Sched::new(f, 2);
        // a big request (6 blocks: the whole of the first extra memory but one) and a small one (3 blocks, kept in the cache when it ends): together they do not fit
        let (pa, pb) = (prompt(120, 90), prompt(121, 40));
        let ra = add(&mut s, pa.clone(), 5).0;
        let rb = add(&mut s, pb.clone(), 5).0;
        run(&mut s);
        assert_eq!(outcome(&rb), (reference(&pb, 5), Ok(Finish::Length)));
        assert_eq!(outcome(&ra), (reference(&pa, 5), Ok(Finish::Length)), "ended for what another request left in the cache");
        assert!(s.stats.taken_back >= 1);
        assert_eq!(s.stats.ended_lost, 0);
    }

    /// Whatever the mix of requests and however the cache's pages and the others' use of the extra memory fall, a request that fits alone is served whole (random requests, the first extra memory the size of the biggest, pages kept for later prompts holding it).
    #[test]
    fn no_request_that_fits_alone_is_ended_whatever_the_mix() {
        for seed in 1..=std::env::var("GLYD_TEST_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(12u64) {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut rnd = move |m: u64| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % m
            };
            let mut f = Fake::new(96, 16);
            f.heavy = heavy_all;
            f.chunks = 8;
            f.slabs = 0;
            // (every other seed with a round that is one step of the rows and the prompts' chunks)
            (f.one_step, f.batch) = (seed % 2 == 0, if seed % 2 == 0 { 4 } else { 1 });
            let users = 2 + (seed % 3) as usize;
            let mut s = Sched::new(f, users);
            // 8 blocks at the most a request (prompt and reply): each fits alone; some begin as an earlier one, so that finished prompts are kept
            let bases: Vec<Vec<u32>> = (0..3).map(|k| prompt(200 + k, 48)).collect();
            let reqs: Vec<(Vec<u32>, usize)> = (0..24)
                .map(|_| {
                    let mut p = if rnd(3) == 0 { bases[rnd(3) as usize].clone() } else { Vec::new() };
                    p.extend(prompt(300 + rnd(50) as u32, 16 + rnd(60) as usize));
                    p.truncate(100);
                    let n = 1 + rnd(120 - p.len() as u64 - 1).min(20) as usize;
                    (p, n)
                })
                .collect();
            let rxs: Vec<_> = reqs.iter().map(|(p, n)| add(&mut s, p.clone(), *n).0).collect();
            run(&mut s);
            for ((p, n), rx) in reqs.iter().zip(&rxs) {
                assert_eq!(outcome(rx), (reference(p, *n), Ok(Finish::Length)), "seed {seed}: a request that fits alone was ended or has a wrong token");
            }
            assert_eq!(s.stats.ended_lost, 0, "seed {seed}");
            assert_eq!(s.be.chunks_used, s.be.state.iter().filter(|&&st| st == 3).count(), "seed {seed}: the first extra memory is held by pages that hold it");
        }
    }

    /// Requests stopped at random moments (waiting, running, held back, sent to run alone) while the extra memory runs out leave nothing behind: every page is back, every unit of the extra memory held is held by a page that holds it, no request
    /// that fits alone is ended with an error, and a stopped request has a first part of its tokens and nothing wrong in it.
    #[test]
    fn requests_stopped_at_random_moments_leave_nothing_behind() {
        for seed in 1..=std::env::var("GLYD_TEST_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(12u64) {
            let mut x = seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) | 1;
            let mut rnd = move |m: u64| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % m
            };
            let mut f = Fake::new(80, 16);
            f.heavy = heavy_all;
            f.chunks = 6;
            f.slabs = 2;
            let mut s = Sched::new(f, 2 + (seed % 4) as usize);
            let reqs: Vec<(Vec<u32>, usize)> = (0..20).map(|_| (prompt(400 + rnd(40) as u32, 16 + rnd(40) as usize), 1 + rnd(25) as usize)).collect();
            let made: Vec<_> = reqs.iter().map(|(p, n)| add(&mut s, p.clone(), *n)).collect();
            let mut ticks = 0;
            while !s.idle() {
                s.tick().unwrap();
                ticks += 1;
                assert!(ticks < 100_000, "seed {seed}: the scheduler does not finish");
                if rnd(7) == 0 {
                    made[rnd(reqs.len() as u64) as usize].1.store(true, Relaxed);
                }
            }
            for ((p, n), (rx, _)) in reqs.iter().zip(&made) {
                let (got, end) = outcome(rx);
                let want = reference(p, *n);
                match end {
                    Ok(Finish::Length) => assert_eq!(got, want, "seed {seed}"),
                    Ok(Finish::Stopped) => assert!(want.starts_with(&got), "seed {seed}: a stopped request has a wrong token"),
                    other => panic!("seed {seed}: ended {other:?}"),
                }
            }
            assert_eq!(s.available(), 80, "seed {seed}");
            s.clear_cache().unwrap();
            assert_eq!((s.be.chunks_used, s.be.slabs_used), (0, 0), "seed {seed}: extra memory is held by nothing");
        }
    }

    /// A request that does not fit the extra memory even alone ends with an error in words; the others are served, and their streams are whole.
    #[test]
    fn a_request_that_cannot_fit_the_extra_memory_alone_ends_with_an_error_in_words() {
        let mut f = Fake::new(64, 16);
        f.heavy = heavy_all;
        f.slabs = 3;
        let mut s = Sched::new(f, 4);
        let (big, _) = add(&mut s, prompt(1, 100), 5);
        let (small, _) = add(&mut s, prompt(2, 30), 5);
        run(&mut s);
        let (got, f) = outcome(&big);
        assert!(got.is_empty(), "no token of a request that never fit");
        assert_eq!(f, Err(ALONE_MESSAGE.to_string()));
        assert!(ALONE_MESSAGE.contains("cannot hold this request"));
        let (got, f) = outcome(&small);
        assert_eq!((got, f), (reference(&prompt(2, 30), 5), Ok(Finish::Length)));
        assert_eq!(s.stats.ended_lost, 1);
        assert_eq!(s.be.slabs_used, 0);
        assert_eq!(s.available(), 64);
    }

    /// A block kept in the second extra memory is not kept for a later prompt (the scarce memory goes back as soon as its sequence ends), and a block that is not is.
    #[test]
    fn heavy_blocks_are_not_kept_for_later_prompts_and_their_memory_goes_back() {
        let mut f = Fake::new(32, 64);
        f.heavy = heavy_all;
        f.slabs = 100;
        let mut s = Sched::new(f, 2);
        let a = prompt(7, 100);
        let ra = add(&mut s, a.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&ra).0, reference(&a, 10));
        assert_eq!(s.be.slabs_used, 0, "the extra memory the ended sequence held is back");
        assert!(s.stats.heavy_blocks > 0);
        let hits = s.stats.hit;
        let rb = add(&mut s, a.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&rb).0, reference(&a, 10));
        assert_eq!(s.stats.hit, hits, "heavy blocks were found in the cache");
        // blocks with nothing heavy in them are kept as before
        s.be.heavy = |_| false;
        let b = prompt(9, 100);
        let rc = add(&mut s, b.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&rc).0, reference(&b, 10));
        let hits = s.stats.hit;
        let rd = add(&mut s, b.clone(), 10).0;
        run(&mut s);
        assert_eq!(collect(&rd).0, reference(&b, 10));
        assert!(s.stats.hit - hits >= 80);
    }

    /// Where the status's rings have overflowed any block may be lost: every running sequence's step is taken back, the cache of finished prompts is dropped, and every stream still goes on whole.
    #[test]
    fn a_ring_that_overflowed_takes_back_every_running_sequence() {
        let mut f = Fake::new(64, 16);
        f.overflow_at = Some(5);
        let mut s = Sched::new(f, 4);
        let prompts: Vec<Vec<u32>> = (0..4).map(|k| prompt(k + 60, 40)).collect();
        let rxs: Vec<_> = prompts.iter().map(|p| add(&mut s, p.clone(), 30).0).collect();
        run(&mut s);
        assert!(s.stats.taken_back >= 4, "taken back: {}", s.stats.taken_back);
        for (p, rx) in prompts.iter().zip(&rxs) {
            assert_eq!(collect(rx), (reference(p, 30), Finish::Length));
        }
        assert_eq!(s.available(), 64);
    }

    /// A logit bias acts where the draws come from the GPU's 64 largest logits and where they come from the row: a token that is last of the row, which no candidate shows, is the choice whatever the sampling
    /// is, and a request with no bias still takes its draw from the candidates (it reads no row).
    #[test]
    fn a_logit_bias_acts_on_the_gpu_path_and_the_host_path_alike() {
        for gpu_top in [true, false] {
            let mut f = Fake::new(64, 32);
            f.gpu_top = gpu_top;
            let mut s = Sched::new(f, 4);
            let p = prompt(3, 40);
            let ask = |s: &mut Sched<Fake>, params: Params| {
                let (tx, rx) = channel();
                s.submit(Submit {
                    prompt: p.clone(),
                    reply: Reply { params, max_new: 12, ignore_eos: false, logprobs: None, stop_ids: Vec::new() },
                    out: tx,
                    abort: Arc::new(AtomicBool::new(false)),
                    resumed: Default::default(),
                    cached: Default::default(),
                    created: Default::default(),
                    salt: 0,
                    pin: 0,
                    images: Vec::new(),
                });
                run(s);
                collect(&rx)
            };
            let base = Params { temperature: 1.0, top_k: 20, seed: 4, ..Params::default() };
            // no bias: drawn from the candidates (the row is not read where there are candidates)
            let reads = s.be.logit_reads;
            let (plain, _) = ask(&mut s, base.clone());
            assert_eq!(plain.len(), 12);
            assert_eq!(s.be.logit_reads - reads > 0, !gpu_top, "gpu_top {gpu_top}: the row is read {} times", s.be.logit_reads - reads);
            // a bias of 100 on token 2000, last of the 2,048: every token of the reply, drawn and greedy
            for t in [1.0f32, 0.0] {
                let (got, finish) = ask(&mut s, Params { temperature: t, logit_bias: vec![(2000, 100.0)], ..base.clone() });
                assert_eq!((got, finish), (vec![2000; 12], Finish::Length), "gpu_top {gpu_top}, temperature {t}");
            }
            // a bias that is not enough leaves the greedy choice (the model's token has 10.0, the biased one -5.0 + 8.0 = 3.0), and one that is enough moves it
            let (got, _) = ask(&mut s, Params { temperature: 0.0, logit_bias: vec![(1500, 8.0)], ..Params::default() });
            assert!(got.iter().all(|&t| t != 1500), "gpu_top {gpu_top}: a bias that does not overtake");
            let (got, _) = ask(&mut s, Params { temperature: 0.0, logit_bias: vec![(1500, 16.0)], ..Params::default() });
            assert_eq!(got, vec![1500; 12], "gpu_top {gpu_top}: a bias that overtakes");
        }
    }

    /// Every token of a request that asks for log-probabilities comes with its own (an `Ev::Lp` before it, the end-of-sequence token's before the end), from the row as the model made it (the same on the
    /// GPU's candidates and on the row), whatever the sampling is, and asking for them changes no token.
    #[test]
    fn log_probabilities_ride_with_the_tokens() {
        let events = |rx: &Receiver<Ev>| {
            let (mut lps, mut toks) = (Vec::new(), Vec::new());
            loop {
                match rx.recv().expect("the sequence ends") {
                    Ev::Lp(l) => lps.push(l),
                    Ev::Token(t) => toks.push(t),
                    Ev::End(f) => return (lps, toks, f.expect("no error")),
                }
            }
        };
        for gpu_top in [true, false] {
            let p = prompt(8, 30);
            let go = |logprobs: Option<usize>, params: Params, eos: Vec<i64>| {
                let mut f = Fake::new(64, 32);
                f.gpu_top = gpu_top;
                f.eos = eos;
                let mut s = Sched::new(f, 4);
                s.eos = s.be.eos.clone();
                let (tx, rx) = channel();
                s.submit(Submit {
                    prompt: p.clone(),
                    reply: Reply { params, max_new: 10, ignore_eos: false, logprobs, stop_ids: Vec::new() },
                    out: tx,
                    abort: Arc::new(AtomicBool::new(false)),
                    resumed: Default::default(),
                    cached: Default::default(),
                    created: Default::default(),
                    salt: 0,
                    pin: 0,
                    images: Vec::new(),
                });
                run(&mut s);
                (events(&rx), s.be.logit_reads)
            };
            let greedy = Params { temperature: 0.0, ..Params::default() };
            let drawn = Params { temperature: 0.9, top_k: 20, seed: 12, ..Params::default() };
            for params in [greedy, drawn] {
                let ((none_lps, none_toks, _), _) = go(None, params.clone(), vec![0]);
                assert!(none_lps.is_empty(), "no log-probabilities were asked for");
                let ((lps, toks, finish), _) = go(Some(3), params.clone(), vec![0]);
                assert_eq!((toks.clone(), finish), (none_toks, Finish::Length), "asking for log-probabilities changed a token (gpu_top {gpu_top}, {params:?})");
                assert_eq!(lps.len(), toks.len());
                for (lp, &t) in lps.iter().zip(&toks) {
                    assert_eq!(lp.token, t);
                    assert_eq!(*lp, crate::sampler::logprobs_of(&row_of(t), t, 3, None), "gpu_top {gpu_top}");
                    // the row has 10.0 for the token the model chose, 5.0 for the next, and -5.0 for 2,046 others
                    let lse = (10f64.exp() + 5f64.exp() + 2046.0 * (-5f64).exp()).ln();
                    let x = row_of(t)[t as usize] as f64;
                    assert!((lp.logprob - (x - lse)).abs() < 1e-9, "{} against {}", lp.logprob, x - lse);
                    assert_eq!(lp.top.len(), 3);
                }
            }
            // the end-of-sequence token is a token with its own log-probability, before the end, and no token event
            let r = reference(&p, 10);
            let (lps, toks, finish) = go(Some(1), Params { temperature: 0.0, ..Params::default() }, vec![r[4] as i64]).0;
            let first = r.iter().position(|&t| t == r[4]).unwrap();
            assert_eq!((toks.clone(), finish), (r[..first].to_vec(), Finish::Stop));
            assert_eq!(lps.len(), toks.len() + 1, "the end token's log-probability comes too");
            assert_eq!(lps.last().unwrap().token, r[4]);
        }
        // with the candidates of the GPU, a request that asks for log-probabilities and draws reads the row (its log-sum-exp is the row's); one that does not, does not
        let mut f = Fake::new(64, 32);
        f.gpu_top = true;
        let mut s = Sched::new(f, 4);
        let (tx, rx) = channel();
        s.submit(Submit {
            prompt: prompt(2, 20),
            reply: Reply {
                params: Params { temperature: 1.0, top_k: 20, seed: 3, ..Params::default() },
                max_new: 6,
                ignore_eos: false,
                logprobs: None,
                stop_ids: Vec::new(),
            },
            out: tx,
            abort: Arc::new(AtomicBool::new(false)),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: Vec::new(),
        });
        run(&mut s);
        let _ = events(&rx);
        assert_eq!(s.be.logit_reads, 0, "a draw from the candidates reads no row");
        // penalties change some logits: the candidates and those tokens' own give the tokens the whole row gives, and no row is read (a top-k of 2: the made-up
        // rows tie most tokens at -5, the last candidate's, so a top-k reaching them would be the row's)
        let penalised = |gpu_top: bool| {
            let mut f = Fake::new(64, 32);
            f.gpu_top = gpu_top;
            let mut s = Sched::new(f, 4);
            let (tx, rx) = channel();
            s.submit(Submit {
                prompt: prompt(2, 20),
                reply: Reply {
                    params: Params { temperature: 1.0, top_k: 2, repetition_penalty: 1.3, frequency_penalty: 0.2, seed: 3, ..Params::default() },
                    max_new: 8,
                    ignore_eos: true,
                    logprobs: None,
                    stop_ids: Vec::new(),
                },
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: Default::default(),
                created: Default::default(),
                salt: 0,
                pin: 0,
                images: Vec::new(),
            });
            run(&mut s);
            let toks = events(&rx).1;
            (toks, s.be.logit_reads)
        };
        let ((by_cands, reads), (by_row, _)) = (penalised(true), penalised(false));
        assert_eq!(by_cands.len(), 8);
        assert_eq!(by_cands, by_row, "the candidates' draw is the row's");
        assert_eq!(reads, 0, "penalties drawn from the candidates read no row");
    }

    /// vLLM's stop_token_ids and min_tokens in the scheduler: a stop token ends the reply as the end token does (counted, not in the reply), and no end or stop token may be chosen before min_tokens tokens: the reply is
    /// at least as long, though the model would have ended it sooner.
    #[test]
    fn stop_tokens_and_a_least_length() {
        let p = prompt(11, 30);
        let r = reference(&p, 12);
        let run_with = |stop_ids: Vec<u32>, min_tokens: usize, eos: Vec<i64>| {
            let mut f = Fake::new(64, 32);
            f.eos = eos;
            f.gpu_top = true;
            // (the model's tokens are hashes: with the row of logits the fake gives, a token that is not the model's has a logit of -5 and is chosen only where the largest is taken out)
            let mut s = Sched::new(f, 4);
            s.eos = s.be.eos.clone();
            let (tx, rx) = channel();
            s.submit(Submit {
                prompt: p.clone(),
                reply: Reply { params: Params { temperature: 0.0, min_tokens, ..Params::default() }, max_new: 12, ignore_eos: false, logprobs: None, stop_ids },
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: Default::default(),
                created: Default::default(),
                salt: 0,
                pin: 0,
                images: Vec::new(),
            });
            run(&mut s);
            collect(&rx)
        };
        // a stop token: the reply ends where the model says it, before it
        let at = 4;
        let (got, f) = run_with(vec![r[at]], 0, vec![0]);
        let first = r.iter().position(|&t| t == r[at]).unwrap();
        assert_eq!((got, f), (r[..first].to_vec(), Finish::Stop));
        // a least length: the end token the model says at `first` is not chosen before the length asked for (the model's next token, the second of its row, is then)
        let (got, f) = run_with(vec![], first + 3, vec![r[first] as i64]);
        assert_eq!(f, Finish::Length, "{got:?}");
        assert!(got.len() >= first + 3 && !got[..first + 3].contains(&r[first]), "{got:?}");
        // with no least length, the end token ends it where the model says
        assert_eq!(run_with(vec![], 0, vec![r[first] as i64]), (r[..first].to_vec(), Finish::Stop));
        // a stop token is held back by the least length as the end token is
        let (got, _) = run_with(vec![r[first]], first + 3, vec![0]);
        assert!(got.len() >= first + 3 && !got[..first + 3].contains(&r[first]), "{got:?}");
    }

    // ---- a slot step: a room of the whole context for each sequence (the description executor's cache), driven by the same scheduler ------------------------------------------------------------------

    /// A model with a cache of `slots` rooms of `window` tokens each: the token after a sequence is `next_of` all of its tokens, read from its own room, so a token written to another sequence's room, at a
    /// wrong place, or left from a sequence that ended changes every token after it. It checks what the scheduler hands it: a chunk or a row goes on exactly where its slot's sequence is.
    pub struct SlotFake {
        pub rooms: Vec<Vec<u32>>,
        pub window: usize,
        pub chunk: usize,
        pub batch: usize,
        /// Slots emptied (`release`), passes of several prompts, decode steps by their rows, and the most slots in use at once.
        pub emptied: usize,
        pub batches: usize,
        pub steps: Vec<usize>,
        pub most: usize,
        /// The first token of each prompt as it begins (at position 0), in order.
        pub begun: Vec<u32>,
        /// Rooms that take memory as they fill: the tokens all the rooms may hold together (in blocks of PAGE), and each room's now. None: rooms of the whole window.
        pub budget: Option<usize>,
        pub held: Vec<usize>,
        pub rows: Vec<Vec<f32>>,
        /// A round is one step (`one_step`): the prompt tokens it takes beside decode rows (None: what a chunk holds after them), and each step's rows and prompt tokens.
        pub one_step: bool,
        pub beside: Option<usize>,
        pub stepped: Vec<(usize, usize)>,
        /// A finished sequence is kept in its slot (`keeps`); one whose state is only what it is after its last token (`whole_only`: a recurrent model) goes on from its end or not at
        /// all. Each `resume`: the slot, the tokens asked for, the tokens kept.
        pub keeps: bool,
        pub whole_only: bool,
        pub resumed: Vec<(i32, usize, usize)>,
        /// A recurrent model that keeps its state at a mark: each slot's mark (0: none).
        pub marking: bool,
        pub marks: Vec<usize>,
        /// Each `share`: from, to, the tokens asked for, the tokens shared (whole pages; a recurrent model's mark).
        pub shared: Vec<(i32, i32, usize, usize)>,
        /// Blocks found by name: each named block's tokens (they stay when a slot is emptied, as a pool's do until it needs them), and each `adopt`: the slot and the
        /// tokens found.
        pub named: std::collections::HashMap<u128, Vec<u32>>,
        pub adopted: Vec<(i32, usize)>,
        /// A recurrent model that keeps its states at boundaries every `snap_every` tokens (0: none) by the name of the block ending there: each kept state (the tokens
        /// it stands for). No prompt piece may run across a boundary.
        pub snap_every: usize,
        pub snaps: std::collections::HashMap<u128, Vec<u32>>,
        /// Names the step says it no longer finds blocks by (`take_removed`), as a test sets them.
        pub removed: Vec<u128>,
        /// Swapping (`swap_out`, `swap_in`): the host's copies by ticket (a room and its mark), the tokens the host's area holds at most (0: the step cannot swap), the bytes a token takes and the
        /// link's bytes a second (`swap_cost`), the polls a copy back takes before `swap_ready` says it is done, each slot's polls left, the tickets given and those let go (`swap_drop`), and the
        /// prompt tokens run (a sequence swapped and copied back runs none again).
        pub host: std::collections::HashMap<u64, (Vec<u32>, usize)>,
        pub host_room: usize,
        pub token_bytes: usize,
        pub link: f64,
        pub load_polls: usize,
        pub loading: Vec<usize>,
        pub tickets: u64,
        pub dropped: usize,
        pub ran: usize,
        /// Steps that ran while a copy back was in flight.
        pub beside_copies: usize,
        /// Rooms of the executor's blocks (`dynamic::pages::Pages`, group 0 full, group 1 sliding with a window of `slide` tokens and a chunk) in place of `budget`, grown as `Dyn::grow`
        /// grows them, or with `jumps` as `Pages::grow` does (to `upto`'s own window: a step whose room for a prompt does not hold its first pass); each group's blocks' tokens, which every
        /// pass is checked to read.
        pub paged: Option<crate::dynamic::pages::Pages>,
        pub slide: usize,
        pub jumps: bool,
        pub blocks: [std::collections::HashMap<u32, [u32; PAGE]>; 2],
        /// A step launched before the step before it is read (`overlaps`, as the executor's on a model with no recurrent state): each step runs at its launch, as a stream runs
        /// what is queued in order, and its samples wait for `finish`; a row's token the step before chose is read from that step's head rows. The steps launched while
        /// another was out, and the steps given up (`abandon`).
        pub overlap: bool,
        pub out: VecDeque<(usize, Samples)>,
        pub heads: Vec<u32>,
        pub launches: usize,
        pub ahead: usize,
        pub abandoned: usize,
        /// A drafting step (`draft`): the most draft tokens a round has (0: it does not draft; a round at position p keeps p mod (drafts + 1) of them), the rounds run, and
        /// those given a sampler.
        pub drafts: usize,
        pub rounds: usize,
        pub drawn_rounds: usize,
    }

    impl SlotFake {
        pub fn new(slots: usize, window: usize, chunk: usize) -> SlotFake {
            SlotFake {
                rooms: vec![Vec::new(); slots],
                window,
                chunk,
                batch: 1,
                emptied: 0,
                batches: 0,
                steps: Vec::new(),
                most: 0,
                rows: Vec::new(),
                begun: Vec::new(),
                budget: None,
                held: vec![0; slots],
                one_step: false,
                beside: None,
                stepped: Vec::new(),
                keeps: false,
                whole_only: false,
                resumed: Vec::new(),
                marking: false,
                marks: vec![0; slots],
                shared: Vec::new(),
                named: Default::default(),
                adopted: Vec::new(),
                snap_every: 0,
                snaps: Default::default(),
                removed: Vec::new(),
                host: Default::default(),
                host_room: 0,
                token_bytes: 1000,
                link: 1e15,
                load_polls: 0,
                loading: vec![0; slots],
                tickets: 0,
                dropped: 0,
                ran: 0,
                beside_copies: 0,
                paged: None,
                slide: 0,
                jumps: false,
                blocks: Default::default(),
                overlap: false,
                out: VecDeque::new(),
                heads: Vec::new(),
                launches: 0,
                ahead: 0,
                abandoned: 0,
                drafts: 0,
                rounds: 0,
                drawn_rounds: 0,
            }
        }
        /// `new` on the executor's blocks: pools as `Dyn::page` sizes them for `slots` sequences of `window` tokens, a full group and a sliding one of `slide` tokens.
        pub fn paged(slots: usize, window: usize, chunk: usize, slide: usize) -> SlotFake {
            let mut f = SlotFake::new(slots, window, chunk);
            let most = |keep: usize| (slots * (keep.div_ceil(PAGE) + 1) + 1) as u32;
            f.paged = Some(crate::dynamic::pages::Pages::new(&[most(window), most(slide + chunk)], &[None, Some(slide + chunk)], slots, window));
            f.slide = slide;
            f
        }
        /// Paged: the tokens from `start` on of the slot's room written to its blocks, and every token the pass reads (a full group's all, a sliding group's the window before `start`
        /// on) found in the block its table names.
        fn write_blocks(&mut self, slot: usize, start: usize) {
            let Some(p) = &self.paged else { return };
            let room = &self.rooms[slot];
            for g in 0..2 {
                let t = p.table(slot, g);
                for (i, &tok) in room.iter().enumerate().skip(start) {
                    assert!(t[i / PAGE] != 0, "token {i} of slot {slot} written to no block of group {g}");
                    self.blocks[g].entry(t[i / PAGE]).or_default()[i % PAGE] = tok;
                }
                let from = if g == 0 { 0 } else { start.saturating_sub(self.slide) };
                for (i, &tok) in room.iter().enumerate().skip(from) {
                    let b = t[i / PAGE];
                    assert!(
                        b != 0 && self.blocks[g].get(&b).map(|x| x[i % PAGE]) == Some(tok),
                        "a pass at {start} of slot {slot} reads token {i} from a block of group {g} that does not hold it"
                    );
                }
            }
        }
        /// A step of these slots: none of them is being copied back; counted where another is.
        fn stepping(&mut self, slots: impl Iterator<Item = i32>) {
            for s in slots {
                assert_eq!(self.loading[s as usize], 0, "a step of slot {s}, whose copy back is not done");
            }
            self.beside_copies += self.loading.iter().any(|&l| l > 0) as usize;
        }
        fn put(&mut self, slot: &[i32], start: usize, tokens: &[u32]) -> u32 {
            assert_eq!(slot.len(), 1, "a slot step's pages are its slot");
            if start == 0 {
                self.begun.push(tokens[0]);
            }
            if self.budget.is_some() {
                assert!(start + tokens.len() <= self.held[slot[0] as usize], "tokens past the room the slot was given");
            }
            if self.whole_only && self.snap_every > 0 {
                let next = (start / self.snap_every + 1) * self.snap_every;
                assert!(start + tokens.len() <= next, "{} tokens from {start}, past the boundary at {next}", tokens.len());
            }
            let room = &mut self.rooms[slot[0] as usize];
            assert_eq!(room.len(), start, "tokens at {start} where the slot holds {}", room.len());
            room.extend_from_slice(tokens);
            assert!(room.len() <= self.window, "past the slot's room");
            let next = next_of(room);
            self.write_blocks(slot[0] as usize, start);
            self.most = self.most.max(self.rooms.iter().filter(|r| !r.is_empty()).count());
            next
        }
    }

    impl ModelStep for SlotFake {
        fn window(&self) -> usize {
            self.window
        }
        fn chunk(&self) -> usize {
            self.chunk
        }
        fn pages(&self) -> usize {
            0
        }
        fn slots(&self) -> Option<usize> {
            Some(self.rooms.len())
        }
        fn eos(&self) -> Vec<i64> {
            vec![0]
        }
        fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
            self.stepping(pages.iter().copied());
            self.ran += tokens.len();
            let t = self.put(pages, start, tokens);
            let s = last.then_some(Sample { token: t, logit: 0.0, logprob: None });
            if let Some(s) = s {
                self.rows = vec![row_of(s.token)];
            }
            Ok(s)
        }
        fn prefill_rows(&self) -> usize {
            self.batch
        }
        fn prefill_many(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
            self.stepping(jobs.iter().map(|j| j.pages[0]));
            self.ran += jobs.iter().map(|j| j.tokens.len()).sum::<usize>();
            let mut rows = Vec::new();
            let out: Vec<Option<Sample>> = jobs
                .iter()
                .map(|j| {
                    let t = self.put(j.pages, j.start, j.tokens);
                    j.last.then(|| {
                        rows.push(row_of(t));
                        Sample { token: t, logit: 0.0, logprob: None }
                    })
                })
                .collect();
            self.batches += (jobs.len() > 1) as usize;
            if !rows.is_empty() {
                self.rows = rows;
            }
            Ok(out)
        }
        fn one_step(&self) -> bool {
            self.one_step
        }
        fn prompt_budget(&self, rows: usize) -> usize {
            self.beside.unwrap_or(self.chunk.saturating_sub(rows))
        }
        fn round(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<(Vec<Sample>, Vec<Option<Sample>>)> {
            assert!(self.one_step, "a step of both, of a model whose rounds are two passes");
            let tokens: usize = jobs.iter().map(|j| j.tokens.len()).sum();
            assert!(rows.len() + tokens <= self.chunk, "{} rows and {tokens} prompt tokens in a step that holds {}", rows.len(), self.chunk);
            let mut seen: Vec<i32> = rows.iter().map(|r| r.pages[0]).chain(jobs.iter().map(|j| j.pages[0])).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), rows.len() + jobs.len(), "two parts of a step in one slot");
            self.stepped.push((rows.len(), tokens));
            self.ran += tokens;
            // (the rows are checked and counted by `decode`)
            match rows.is_empty() {
                true => self.stepping(jobs.iter().map(|j| j.pages[0])),
                false => assert!(jobs.iter().all(|j| self.loading[j.pages[0] as usize] == 0), "a prompt's chunk in a slot whose copy back is not done"),
            }
            let decoded = if rows.is_empty() { Vec::new() } else { self.decode(rows)? };
            let mut all = if rows.is_empty() { Vec::new() } else { std::mem::take(&mut self.rows) };
            let prompted: Vec<Option<Sample>> = jobs
                .iter()
                .map(|j| {
                    let t = self.put(j.pages, j.start, j.tokens);
                    j.last.then(|| {
                        all.push(row_of(t));
                        Sample { token: t, logit: 0.0, logprob: None }
                    })
                })
                .collect();
            self.rows = all;
            Ok((decoded, prompted))
        }
        fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
            let mut seen: Vec<i32> = rows.iter().map(|r| r.pages[0]).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), rows.len(), "two rows of a step in one slot");
            self.stepping(rows.iter().map(|r| r.pages[0]));
            self.steps.push(rows.len());
            let out: Vec<Sample> = rows.iter().map(|r| Sample { token: self.put(r.pages, r.pos, &[r.token]), logit: 0.0, logprob: None }).collect();
            self.rows = out.iter().map(|s| row_of(s.token)).collect();
            Ok(out)
        }
        fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
            assert!(self.out.is_empty(), "a step's logits read after a later step was launched over them");
            self.rows.get(row).cloned().ok_or_else(|| crate::ffi::Error("no logits".into()))
        }
        fn overlaps(&self) -> bool {
            self.overlap
        }
        fn launch(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<Launched> {
            if !self.overlap {
                assert!(rows.iter().all(|r| r.after.is_none()), "a row whose token the step before chose, on a step that does not overlap");
                let done = match (self.one_step, jobs.is_empty()) {
                    (true, _) => self.round(rows, jobs)?,
                    (false, true) => (self.decode(rows)?, Vec::new()),
                    (false, false) => (Vec::new(), self.prefill_many(jobs)?),
                };
                return Ok(Launched { done: Some(done), tag: 0 });
            }
            let toks: Vec<u32> = rows.iter().map(|r| r.after.map_or(r.token, |h| self.heads[h])).collect();
            let rows: Vec<Row> = rows.iter().zip(&toks).map(|(r, &t)| Row::new(t, r.pos, r.pages)).collect();
            let out = self.round(&rows, jobs)?;
            self.heads = out.0.iter().map(|x| x.token).chain(out.1.iter().flatten().map(|x| x.token)).collect();
            self.launches += 1;
            self.ahead += !self.out.is_empty() as usize;
            self.out.push_back((self.launches, out));
            Ok(Launched { done: None, tag: self.launches })
        }
        fn finish(&mut self, l: Launched) -> Result<Samples> {
            if let Some(d) = l.done {
                return Ok(d);
            }
            let (tag, out) = self.out.pop_front().expect("a step launched");
            assert_eq!(tag, l.tag, "steps finished out of their order");
            Ok(out)
        }
        fn abandon(&mut self) {
            self.abandoned += self.out.len();
            self.out.clear();
        }
        fn draft(&mut self, row: &Row, sampler: Option<&mut Sampler>) -> Result<Option<Vec<u32>>> {
            if self.drafts == 0 {
                return Ok(None);
            }
            (self.rounds, self.drawn_rounds) = (self.rounds + 1, self.drawn_rounds + sampler.is_some() as usize);
            let m = 1 + row.pos % (self.drafts + 1);
            let mut out = vec![self.put(row.pages, row.pos, &[row.token])];
            while out.len() < m {
                let (at, t) = (self.rooms[row.pages[0] as usize].len(), out[out.len() - 1]);
                out.push(self.put(row.pages, at, &[t]));
            }
            Ok(Some(out))
        }
        fn release(&mut self, pages: &[i32]) -> Result<()> {
            for &p in pages {
                self.rooms[p as usize].clear();
                self.held[p as usize] = 0;
                self.marks[p as usize] = 0;
                self.loading[p as usize] = 0;
                self.emptied += 1;
                if let Some(pg) = &mut self.paged {
                    pg.release(p as usize);
                }
            }
            Ok(())
        }
        fn swap_cost(&self, slot: i32) -> Option<(usize, f64)> {
            (self.host_room > 0 && !self.rooms[slot as usize].is_empty()).then(|| (self.rooms[slot as usize].len() * self.token_bytes, self.link))
        }
        fn swap_out(&mut self, slot: i32) -> Result<Option<u64>> {
            let s = slot as usize;
            let held: usize = self.host.values().map(|h| h.0.len()).sum();
            assert_eq!(self.loading[s], 0, "slot {s} swapped out while it was copied back");
            if self.rooms[s].is_empty() || held + self.rooms[s].len() > self.host_room {
                return Ok(None);
            }
            self.tickets += 1;
            self.host.insert(self.tickets, (std::mem::take(&mut self.rooms[s]), self.marks[s]));
            (self.held[s], self.marks[s]) = (0, 0);
            Ok(Some(self.tickets))
        }
        fn swap_in(&mut self, slot: i32, ticket: u64, upto: usize) -> Result<Option<usize>> {
            let s = slot as usize;
            assert!(self.rooms[s].is_empty() && self.held[s] == 0, "a copy back into slot {s}, which holds {} tokens", self.rooms[s].len());
            let n = self.host.get(&ticket).expect("a ticket given").0.len();
            assert!(upto > n, "room for {upto} tokens asked for a sequence of {n}");
            if let Some(budget) = self.budget {
                if self.held.iter().sum::<usize>() + upto.div_ceil(PAGE) * PAGE > budget {
                    return Ok(None);
                }
                self.held[s] = upto.div_ceil(PAGE) * PAGE;
            }
            let (room, mark) = self.host.remove(&ticket).expect("a ticket given");
            (self.rooms[s], self.marks[s], self.loading[s]) = (room, mark, self.load_polls);
            Ok(Some(n))
        }
        fn swap_ready(&mut self, slot: i32, wait: bool) -> Result<bool> {
            let l = &mut self.loading[slot as usize];
            if wait || *l == 0 {
                *l = 0;
                return Ok(true);
            }
            *l -= 1;
            Ok(false)
        }
        fn swap_drop(&mut self, ticket: u64) {
            self.host.remove(&ticket).expect("a ticket given");
            self.dropped += 1;
        }
        fn shares_starts(&self) -> bool {
            // (paged: no start is shared, the rooms are the blocks' alone)
            self.keeps && self.paged.is_none() && (!self.whole_only || self.snap_every > 0)
        }
        fn snap_every(&self) -> usize {
            if self.whole_only {
                self.snap_every
            } else {
                0
            }
        }
        fn take_removed(&mut self) -> Vec<u128> {
            std::mem::take(&mut self.removed)
        }
        fn name(&mut self, slot: i32, first: usize, keys: &[u128]) -> Result<()> {
            let room = &self.rooms[slot as usize];
            assert!((first + keys.len()) * PAGE <= room.len(), "blocks {first} to {} named of a slot that holds {} tokens", first + keys.len(), room.len());
            for (i, &k) in keys.iter().enumerate() {
                self.named.entry(k).or_insert_with(|| room[(first + i) * PAGE..(first + i + 1) * PAGE].to_vec());
            }
            // (the state at a boundary the last block named ends on, where the slot ends there too)
            let end = (first + keys.len()) * PAGE;
            if let Some(&k) = keys.last().filter(|_| self.whole_only && self.snap_every > 0 && end == room.len() && end % self.snap_every == 0) {
                let state = room.clone();
                self.snaps.entry(k).or_insert(state);
            }
            Ok(())
        }
        fn held_by_name(&self, keys: &[u128]) -> usize {
            if self.whole_only {
                let per = (self.snap_every / PAGE).max(1);
                return (1..=keys.len() / per)
                    .rev()
                    .map(|k| k * per)
                    .find(|&j| self.snaps.contains_key(&keys[j - 1]) && keys[..j].iter().all(|k| self.named.contains_key(k)))
                    .map_or(0, |j| j * PAGE);
            }
            keys.iter().take_while(|k| self.named.contains_key(k)).count() * PAGE
        }
        fn adopt(&mut self, slot: i32, keys: &[u128]) -> Result<usize> {
            assert!(self.rooms[slot as usize].is_empty(), "a start looked for into slot {slot}, which holds {}", self.rooms[slot as usize].len());
            if self.whole_only {
                let n = self.held_by_name(keys);
                let state = if n > 0 { self.snaps[&keys[n / PAGE - 1]].clone() } else { Vec::new() };
                let blocks: Vec<u32> = keys[..n / PAGE].iter().flat_map(|k| self.named[k].iter().copied()).collect();
                assert_eq!(state, blocks, "a kept state that is not its blocks' tokens");
                self.adopted.push((slot, n));
                self.rooms[slot as usize] = state;
                return Ok(n);
            }
            let found: Vec<u32> = keys.iter().map_while(|k| self.named.get(k)).flatten().copied().collect();
            self.adopted.push((slot, found.len()));
            self.rooms[slot as usize] = found;
            Ok(self.rooms[slot as usize].len())
        }
        fn share(&mut self, from: i32, to: i32, upto: usize) -> Result<usize> {
            assert!(from != to && self.rooms[to as usize].is_empty(), "a start shared into slot {to}, which holds {}", self.rooms[to as usize].len());
            assert!(upto <= self.rooms[from as usize].len(), "{upto} tokens shared of a slot that holds {}", self.rooms[from as usize].len());
            let mark = self.marks[from as usize];
            let n = match (self.whole_only, mark > 0 && mark <= upto) {
                (false, _) => upto / PAGE * PAGE,
                (true, true) => mark,
                (true, false) => 0,
            };
            self.rooms[to as usize] = self.rooms[from as usize][..n].to_vec();
            if self.whole_only {
                self.marks[to as usize] = n;
            }
            self.shared.push((from, to, upto, n));
            Ok(n)
        }
        fn mark(&mut self, slot: i32, n: usize) -> Result<()> {
            assert_eq!(self.rooms[slot as usize].len(), n, "a mark at {n} of a slot that holds {}", self.rooms[slot as usize].len());
            if self.marking {
                self.marks[slot as usize] = n;
            }
            Ok(())
        }
        fn keeps(&self) -> bool {
            self.keeps
        }
        fn resume(&mut self, slot: i32, upto: usize) -> Result<usize> {
            let room = &mut self.rooms[slot as usize];
            assert!(upto > 0 && upto <= room.len(), "a slot of {} tokens resumed at {upto}", room.len());
            let mark = self.marks[slot as usize];
            let kept = match (self.whole_only && upto < room.len(), mark > 0 && mark <= upto) {
                (false, _) => upto,
                (true, true) => mark,
                (true, false) => 0,
            };
            // (paged: where its blocks hold what a step there reads, as `Dyn::resume`)
            let kept = self.paged.as_mut().map_or(kept, |p| p.shrink(slot as usize, kept, self.chunk));
            if kept > 0 {
                room.truncate(kept);
            }
            self.resumed.push((slot, upto, kept));
            Ok(kept)
        }
        fn grow(&mut self, slot: i32, upto: usize) -> Result<bool> {
            if let Some(p) = &mut self.paged {
                let (s, n) = (slot as usize, self.rooms[slot as usize].len());
                return Ok(if self.jumps { p.grow(s, upto) } else { p.grow_from(s, n, upto, self.chunk) });
            }
            let Some(budget) = self.budget else { return Ok(true) };
            let want = upto.div_ceil(PAGE) * PAGE;
            let have = self.held[slot as usize];
            if want <= have {
                return Ok(true);
            }
            if self.held.iter().sum::<usize>() + want - have > budget {
                return Ok(false);
            }
            self.held[slot as usize] = want;
            Ok(true)
        }
    }

    /// Every sequence on a slot step gives the tokens it gives alone, however many run together, with prompts one a pass or several in one; no more run at once than there are slots, and every slot is
    /// empty again at the end.
    #[test]
    fn a_slot_step_gives_each_sequence_its_tokens_alone() {
        for (slots, users, batch, chunk) in [(1, 4, 1, 16), (3, 4, 1, 16), (4, 4, 4, 16), (4, 8, 4, 64), (8, 8, 8, 7), (24, 64, 24, 512)] {
            let mut f = SlotFake::new(slots, 256, chunk);
            f.batch = batch;
            let mut s = Sched::new(f, users);
            assert_eq!(s.users(), users.min(slots));
            let reqs: Vec<(Vec<u32>, usize)> = (0..slots.max(9) + 3).map(|k| (prompt(k as u32 + 1, 3 + 11 * (k % 9)), 5 + 3 * (k % 9))).collect();
            let rx: Vec<_> = reqs.iter().map(|(p, n)| add(&mut s, p.clone(), *n).0).collect();
            run(&mut s);
            for ((p, n), rx) in reqs.iter().zip(&rx) {
                assert_eq!(collect(rx), (reference(p, *n), Finish::Length), "slots {slots}, batch {batch}, chunk {chunk}");
            }
            assert!(s.be.most <= slots);
            assert!(s.be.rooms.iter().all(Vec::is_empty) && s.be.emptied == reqs.len());
            assert_eq!(s.available(), slots);
            if batch > 1 && slots > 1 && chunk >= 64 {
                assert!(s.be.batches > 0, "prompts ran together");
            }
            if slots > 1 {
                assert!(s.be.steps.iter().any(|&r| r > 1), "decode steps of several rows");
            }
        }
    }

    /// Admission is first come, first served: with one slot each sequence's prompt begins when the one before it has ended, in the order the requests came; a later short one does not pass an earlier
    /// long one. (The order is the step's, as it sees each prompt begin: the events reach the requests on a thread of their own.)
    #[test]
    fn admission_is_in_the_order_requests_come() {
        let mut s = Sched::new(SlotFake::new(1, 256, 16), 4);
        let lens = [40usize, 2, 30, 1];
        let prompts: Vec<Vec<u32>> = lens.iter().map(|&n| prompt(n as u32, 5)).collect();
        let rx: Vec<_> = prompts.iter().zip(lens).map(|(p, n)| add(&mut s, p.clone(), n).0).collect();
        run(&mut s);
        let begun: Vec<usize> = s.be.begun.iter().map(|t| prompts.iter().position(|p| p[0] == *t).expect("a prompt")).collect();
        assert_eq!(begun, vec![0, 1, 2, 3]);
        for ((p, n), r) in prompts.iter().zip(lens).zip(&rx) {
            assert_eq!(collect(r), (reference(p, n), Finish::Length));
        }
    }

    /// Fairness while decoding: every round with a decode step gives every sequence that is decoding one token, so the replies of sequences that started together never differ by more than a token
    /// until one ends; a new prompt runs beside them (they wait for it at most a few rounds, never until it is done). (Counted in the step's rooms, which the rounds fill, not by the events, which reach
    /// the requests on a thread of their own.)
    #[test]
    fn every_decoding_sequence_gets_a_token_each_step() {
        let mut s = Sched::new(SlotFake::new(4, 4096, 32), 4);
        let rx: Vec<_> = (0..3).map(|k| add(&mut s, prompt(k + 1, 8), 200).0).collect();
        let mut late: Option<Receiver<Ev>> = None;
        let (mut rounds_waited, mut longest_wait) = (0, 0);
        // (slots 0, 1 and 2 are the three's: the lowest free slot is taken first)
        let made = |s: &Sched<SlotFake>| -> Vec<usize> { (0..3).map(|k| s.be.rooms[k].len().saturating_sub(8)).collect() };
        for round in 0..400 {
            if s.idle() {
                break;
            }
            if round == 20 {
                // a long prompt comes while the three decode: 2,000 tokens, 63 chunks
                late = Some(add(&mut s, prompt(99, 2000), 4).0);
            }
            let before = made(&s);
            s.tick().unwrap();
            let got = made(&s);
            if got.iter().all(|&g| g > 0 && g < 199) {
                let (lo, hi) = (got.iter().min().unwrap(), got.iter().max().unwrap());
                assert!(hi - lo <= 1, "round {round}: {got:?}");
            }
            if got == before && got.iter().any(|&g| g > 0 && g < 199) {
                rounds_waited += 1;
                longest_wait = longest_wait.max(rounds_waited);
            } else {
                rounds_waited = 0;
            }
        }
        run(&mut s);
        assert!(longest_wait <= MAX_LONG_SKIPS as usize + 1, "decoding sequences waited {longest_wait} rounds in a row");
        for (k, r) in rx.iter().enumerate() {
            assert_eq!(collect(r), (reference(&prompt(k as u32 + 1, 8), 200), Finish::Length));
        }
        assert_eq!(collect(&late.unwrap()).0, reference(&prompt(99, 2000), 4));
    }

    /// A step that keeps: a slot fake of `slots` rooms whose finished sequences stay in them.
    fn keeping(slots: usize) -> Sched<SlotFake> {
        let mut f = SlotFake::new(slots, 512, 16);
        f.keeps = true;
        Sched::new(f, slots)
    }

    /// One request run to its end: its reply.
    fn turn(s: &mut Sched<SlotFake>, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let (rx, _) = add(s, prompt.to_vec(), max_new);
        run(s);
        let (got, f) = collect(&rx);
        assert_eq!(f, Finish::Length);
        got
    }

    /// R4c's first step, a conversation: its next turn sends the turns before it again and goes on in the slot the last one finished in, from the tokens the slot holds; every reply is
    /// the reply the whole prompt gives from its first token (the fake's next token is a function of every token before it, and it refuses tokens that are not put where its slot
    /// ends). The prompt tokens found are counted; a regenerated reply (the same prompt again) goes on from the prompt's end.
    #[test]
    fn a_conversations_next_turn_goes_on_in_its_slot() {
        let mut s = keeping(2);
        let p1 = prompt(1, 40);
        let r1 = turn(&mut s, &p1, 8);
        assert_eq!(r1, reference(&p1, 8));
        assert_eq!((s.kept(), s.be.emptied, s.available()), (1, 0, 2), "the finished sequence stays in its slot, which can still be had");
        // the next turn: the prompt, the reply, and more
        let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 10)[..]].concat();
        let r2 = turn(&mut s, &p2, 8);
        assert_eq!(r2, reference(&p2, 8));
        // (the slot held the prompt and the reply but its last token, whose keys and values no step wrote)
        assert_eq!(s.be.resumed, vec![(0, 47, 47)]);
        assert_eq!((s.stats.prompt, s.stats.hit), (40 + 58, 47));
        assert_eq!(s.be.begun, vec![p1[0]], "the second turn did not begin at its first token");
        // a third turn, and the second's reply asked for again (from the end of its prompt: the slot is cut back)
        let p3: Vec<u32> = [&p2[..], &r2[..], &prompt(3, 5)[..]].concat();
        assert_eq!(turn(&mut s, &p3, 4), reference(&p3, 4));
        assert_eq!(s.be.resumed[1], (0, 65, 65));
        assert_eq!(turn(&mut s, &p3, 6), reference(&p3, 6));
        assert_eq!(s.be.resumed[2], (0, 70, 70), "the prompt but its last token, which is run for its logits");
        assert_eq!((s.be.emptied, s.be.begun.len(), s.kept()), (0, 1, 1));
    }

    /// What a request found of its prompt is told to it (`Submit::cached`, for its usage): nothing the first time, the kept tokens on the next turn; and on pages, the full
    /// pages an earlier prompt left.
    #[test]
    fn a_request_is_told_the_prompt_tokens_it_found() {
        let told = |s: &mut dyn FnMut(Submit), prompt: &[u32]| -> (Receiver<Ev>, Arc<AtomicU32>) {
            let (tx, rx) = channel();
            let cached = Arc::new(AtomicU32::new(0));
            s(Submit {
                prompt: prompt.to_vec(),
                reply: reply(8),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: cached.clone(),
                created: Default::default(),
                salt: 0,
                pin: 0,
                images: Vec::new(),
            });
            (rx, cached)
        };
        // a slot step that keeps
        let mut s = keeping(2);
        let p1 = prompt(1, 40);
        let (rx, c1) = told(&mut |x| s.submit(x), &p1);
        run(&mut s);
        let r1 = collect(&rx).0;
        let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 10)[..]].concat();
        let (rx, c2) = told(&mut |x| s.submit(x), &p2);
        run(&mut s);
        assert_eq!(collect(&rx).0, reference(&p2, 8));
        assert_eq!((c1.load(Relaxed), c2.load(Relaxed)), (0, 47));
        // pages: what an earlier prompt left is found a full page at a time
        let mut s = Sched::new(Fake::new(64, 32), 2);
        let (rx, c1) = told(&mut |x| s.submit(x), &p1);
        run(&mut s);
        collect(&rx);
        let (rx, c2) = told(&mut |x| s.submit(x), &p2);
        run(&mut s);
        assert_eq!(collect(&rx).0, reference(&p2, 8));
        // (the first left 47 tokens in the cache, the prompt and the reply but its last: 2 full pages, and 15 tokens of a third, which is not kept)
        assert_eq!((c1.load(Relaxed), c2.load(Relaxed)), (0, 2 * PAGE as u32));
    }

    /// R4c: a prompt that begins as another conversation's does finds that start's full blocks by name and runs the rest, in a slot of its own; the other conversation
    /// stays whole for its own next turn. Every reply is the one the whole prompt gives from its first token.
    #[test]
    fn a_prompt_shares_the_start_of_another_conversation() {
        let mut s = keeping(3);
        let system = prompt(1, 60);
        let pa: Vec<u32> = [&system[..], &prompt(2, 100)[..]].concat();
        let ra = turn(&mut s, &pa, 8);
        assert_eq!((ra.clone(), s.kept()), (reference(&pa, 8), 1));
        // another conversation with the same system prompt: 60 tokens of the 167 kept, so the kept one is not cut down for it; its 3 full blocks are found by name
        let pb: Vec<u32> = [&system[..], &prompt(3, 30)[..]].concat();
        assert_eq!(turn(&mut s, &pb, 8), reference(&pb, 8));
        assert_eq!((s.be.adopted.clone(), s.be.resumed.len(), s.stats.hit, s.kept()), (vec![(0, 0), (1, 48)], 0, 48, 2));
        assert_eq!(s.be.begun, vec![pa[0]], "the second did not begin at its first token");
        // the first conversation's next turn goes on from all it left
        let a2: Vec<u32> = [&pa[..], &ra[..], &prompt(4, 9)[..]].concat();
        assert_eq!(turn(&mut s, &a2, 8), reference(&a2, 8));
        assert_eq!((s.be.resumed.clone(), s.stats.hit), (vec![(0, 167, 167)], 48 + 167));
        // a third conversation: the longest start that is held (90 tokens of the first, which holds 191 now; 60 of the second)
        let pc: Vec<u32> = [&a2[..90], &prompt(5, 40)[..]].concat();
        assert_eq!(turn(&mut s, &pc, 8), reference(&pc, 8));
        assert_eq!(s.be.adopted[2], (2, 80));
        // another scope shares nothing
        let (tx, rx) = channel();
        s.submit(Submit {
            prompt: pb.clone(),
            reply: reply(8),
            out: tx,
            abort: Arc::new(AtomicBool::new(false)),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 5,
            pin: 0,
            images: Vec::new(),
        });
        run(&mut s);
        assert_eq!((collect(&rx).0, s.be.adopted.last().map(|x| x.1), s.be.shared.len()), (reference(&pb, 8), Some(0), 0));
    }

    /// #241: a request with a cache marker is told the prompt tokens it left findable by name that it did not find (Anthropic's cache creation); the next with the same
    /// start finds those and is told the rest it left; a request with no marker is told none.
    #[test]
    fn a_request_with_a_cache_marker_is_told_what_it_left_findable() {
        let mut s = keeping(3);
        let go = |s: &mut Sched<SlotFake>, p: &[u32], pin: u32| -> (u32, u32) {
            let (tx, rx) = channel();
            let (cached, created) = (Arc::new(AtomicU32::new(0)), Arc::new(AtomicU32::new(0)));
            s.submit(Submit {
                prompt: p.to_vec(),
                reply: reply(8),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: cached.clone(),
                created: created.clone(),
                salt: 0,
                pin,
                images: Vec::new(),
            });
            run(s);
            collect(&rx);
            (cached.load(Relaxed), created.load(Relaxed))
        };
        let system = prompt(1, 64);
        // (94 tokens: the 93 before its last, 5 full blocks)
        let pa: Vec<u32> = [&system[..], &prompt(2, 30)[..]].concat();
        assert_eq!(go(&mut s, &pa, 300), (0, 80));
        let pb: Vec<u32> = [&system[..], &prompt(3, 30)[..]].concat();
        assert_eq!(go(&mut s, &pb, 300), (64, 16));
        let pc: Vec<u32> = [&system[..], &prompt(4, 30)[..]].concat();
        assert_eq!(go(&mut s, &pc, 0).1, 0);
    }

    /// #142: the cache's events where the context keeps them: a request's full blocks stored as they are named, with their tokens and their names (the chain of its
    /// scope's), each sequence's first stored block with no parent; a step's removed blocks first in its batch. Where the context does not keep them, nothing is kept.
    #[test]
    fn the_cache_tells_its_stored_and_removed_blocks() {
        use crate::kv_events::Event;
        let mut f = SlotFake::new(2, 512, 16);
        f.keeps = true;
        let rt = Context::with_kv_events();
        let mut s = Sched::new(f, 2).in_context(rt.clone());
        let p = prompt(1, 40);
        let r = turn(&mut s, &p, 8);
        let log = rt.kv_events.as_ref().unwrap();
        let (batches, gap, _) = log.since(1, std::time::Duration::ZERO);
        let stored: Vec<(Vec<u128>, Option<u128>, Vec<u32>)> = batches
            .iter()
            .flat_map(|b| b.events.iter())
            .filter_map(|e| match e {
                Event::Stored { blocks, parent, tokens } => Some((blocks.clone(), *parent, tokens.clone())),
                _ => None,
            })
            .collect();
        let tokens: Vec<u32> = stored.iter().flat_map(|x| x.2.iter().copied()).collect();
        let names: Vec<u128> = stored.iter().flat_map(|x| x.0.iter().copied()).collect();
        // (the prompt and its reply but the last token: 47 tokens, 2 full blocks)
        let seq: Vec<u32> = [&p[..], &r[..r.len() - 1]].concat();
        assert_eq!((gap, tokens.len(), &tokens[..]), (false, 2 * PAGE, &seq[..2 * PAGE]));
        assert_eq!(names, crate::kv::block_names(crate::kv::block_root(0), &tokens));
        assert_eq!(stored[0].1, None, "a sequence's first block has no parent");
        assert!(stored.windows(2).all(|w| w[1].1 == w[0].0.last().copied()), "each later stored run's parent is the block before it");
        // a step's removed blocks: first in its batch
        let before = log.since(0, std::time::Duration::ZERO).2;
        s.be.removed = vec![9, 7];
        turn(&mut s, &prompt(2, 40), 8);
        let (batches, _, _) = log.since(before, std::time::Duration::ZERO);
        let with = batches.iter().find(|b| b.events.iter().any(|e| matches!(e, Event::Removed { .. }))).expect("a batch with the removed blocks");
        assert_eq!(with.events[0], Event::Removed { blocks: vec![9, 7] });
        // a context that does not keep them: nothing kept
        let mut f = SlotFake::new(2, 512, 16);
        f.keeps = true;
        let mut s = Sched::new(f, 2).in_context(Context::new());
        turn(&mut s, &p, 8);
        assert!(s.kv_out.is_empty() && s.rt.kv_events.is_none());
    }

    /// #142: a recurrent model that keeps its states at boundaries (every 32 tokens here, prompts in chunks of 64) has a start another conversation ran found by name to
    /// the newest boundary within it, where the other's mark is past what the two share; no prompt piece runs across a boundary (the step's own check); the other
    /// conversation still goes on from all it left at its next turn. Every reply is the one the whole prompt gives from its first token.
    #[test]
    fn a_recurrent_model_finds_a_start_at_its_newest_kept_boundary() {
        for ahead in [false, true] {
            recurrent_starts(ahead);
        }
    }

    /// `a_recurrent_model_finds_a_start_at_its_newest_kept_boundary`, with steps launched before the one before lands (`ahead`): the states at a boundary and at a prompt's
    /// end are copied as the next step is launched, and are the same states.
    fn recurrent_starts(ahead: bool) {
        let mut f = SlotFake::new(3, 512, 64);
        (f.keeps, f.whole_only, f.marking, f.snap_every) = (true, true, true, 32);
        (f.one_step, f.batch, f.overlap) = (ahead, 3, ahead);
        let mut s = Sched::new(f, 3);
        let system = prompt(1, 70);
        let pa: Vec<u32> = [&system[..], &prompt(2, 100)[..]].concat();
        let ra = turn(&mut s, &pa, 8);
        assert_eq!(ra, reference(&pa, 8));
        // 70 tokens in common, the other's mark at its prompt's end (170): the boundary at 64, by name
        let pb: Vec<u32> = [&system[..], &prompt(3, 30)[..]].concat();
        assert_eq!(turn(&mut s, &pb, 8), reference(&pb, 8));
        assert_eq!((s.be.adopted.last(), s.stats.hit), (Some(&(1, 64)), 64));
        // the first conversation's next turn: from all it left, in its own slot
        let a2: Vec<u32> = [&pa[..], &ra[..], &prompt(4, 9)[..]].concat();
        assert_eq!(turn(&mut s, &a2, 8), reference(&a2, 8));
        assert_eq!(s.be.resumed.last().map(|r| r.2), Some(pa.len() + ra.len() - 1));
        // a start shorter than a boundary: nothing found
        let pc: Vec<u32> = [&system[..20], &prompt(5, 10)[..]].concat();
        assert_eq!(turn(&mut s, &pc, 8), reference(&pc, 8));
        assert_eq!(s.be.adopted.last().map(|a| a.1), Some(0));
        assert_eq!(s.be.ahead > 0, ahead, "steps launched ahead");
    }

    /// A request that comes while another with the same start still runs finds what that one has run; and where the only slot holds a kept conversation whose start it
    /// shares, the slot is emptied for it and the start is found still: blocks are found by their names, not by who holds them.
    #[test]
    fn a_start_is_found_while_its_sequence_runs_and_after_its_slot_was_taken() {
        let mut s = keeping(2);
        let system = prompt(1, 100);
        let pa: Vec<u32> = [&system[..], &prompt(2, 20)[..]].concat();
        let (rx_a, _) = add(&mut s, pa.clone(), 60);
        // (the first's prompt runs, and some of its reply)
        for _ in 0..12 {
            s.tick().unwrap();
        }
        let pb: Vec<u32> = [&system[..], &prompt(3, 20)[..]].concat();
        let (rx_b, _) = add(&mut s, pb.clone(), 8);
        run(&mut s);
        assert_eq!((collect(&rx_a).0, collect(&rx_b).0), (reference(&pa, 60), reference(&pb, 8)));
        assert_eq!(s.be.adopted, vec![(0, 0), (1, 96)]);
        // one slot, and a kept conversation in it: a prompt that shares its start (too little of it to be its next turn) has the slot, emptied, and the start by name
        let mut s = keeping(1);
        let long: Vec<u32> = [&system[..], &prompt(4, 150)[..]].concat();
        let rl = turn(&mut s, &long, 8);
        assert_eq!(turn(&mut s, &pb, 8), reference(&pb, 8));
        assert_eq!((s.be.resumed.len(), s.be.adopted.last(), s.be.emptied, s.be.begun.len()), (0, Some(&(0, 96)), 1, 1));
        // and the conversation whose slot was taken finds its own full blocks, all 16 of them, at its next turn
        let next: Vec<u32> = [&long[..], &rl[..], &prompt(5, 6)[..]].concat();
        assert_eq!(turn(&mut s, &next, 8), reference(&next, 8));
        assert_eq!((s.be.adopted.last(), s.be.begun.len()), (Some(&(0, 256)), 1));
    }

    /// Requests that begin alike and come together: the first runs the start, the others wait while at least a chunk of it is left to run and share what is run by
    /// then, each running only the rest, where the step shares starts of any length; a request with another start is not kept waiting by them. A recurrent model, which shares at a mark alone, holds nothing back.
    #[test]
    fn requests_that_begin_alike_wait_for_the_first_to_run_the_start() {
        let system = prompt(1, 200);
        let ps: Vec<Vec<u32>> = (0..3).map(|k| [&system[..], &prompt(10 + k, 20)[..]].concat()).collect();
        let other = prompt(50, 60);
        let mut s = keeping(4);
        let rx: Vec<_> = ps.iter().chain([&other]).map(|p| add(&mut s, p.clone(), 8).0).collect();
        // (one round: the first of the three and the other prompt are in; the two that begin as the first wait)
        s.tick().unwrap();
        assert_eq!((s.active.len(), s.waiting.len()), (2, 2));
        run(&mut s);
        for (p, r) in ps.iter().chain([&other]).zip(&rx) {
            assert_eq!(collect(r), (reference(p, 8), Finish::Length));
        }
        // the start was run once but for its end: the two were let in when less than a chunk (32 tokens) of it was left to run, and shared the 176 tokens run by then
        assert_eq!(s.be.adopted.iter().map(|x| x.1).collect::<Vec<_>>(), vec![0, 0, 176, 176]);
        assert_eq!((s.be.begun.len(), s.stats.hit), (2, 2 * 176));
        // a recurrent model: all three begin at their first token, at once
        let mut s = keeping(4);
        (s.be.whole_only, s.be.marking) = (true, true);
        let rx: Vec<_> = ps.iter().map(|p| add(&mut s, p.clone(), 8).0).collect();
        s.tick().unwrap();
        assert_eq!((s.active.len(), s.waiting.len()), (3, 0));
        run(&mut s);
        for (p, r) in ps.iter().zip(&rx) {
            assert_eq!(collect(r), (reference(p, 8), Finish::Length));
        }
        assert_eq!(s.be.begun.len(), 3);
    }

    /// A recurrent model shares a start only at a mark (its states as they were at a prompt's end): a request that begins with a running sequence's whole prompt goes
    /// on from there; one that leaves it before its end shares nothing.
    #[test]
    fn a_recurrent_model_shares_a_start_at_a_mark_only() {
        let mut s = keeping(3);
        (s.be.whole_only, s.be.marking) = (true, true);
        let pa = prompt(1, 70);
        let (rx_a, _) = add(&mut s, pa.clone(), 60);
        for _ in 0..12 {
            s.tick().unwrap();
        }
        // the same prompt and more, while the first is still writing its reply: from the first's prompt's end
        let pb: Vec<u32> = [&pa[..], &prompt(2, 20)[..]].concat();
        let (rx_b, _) = add(&mut s, pb.clone(), 8);
        // a prompt that shares 40 tokens: before the mark, nothing
        let pc: Vec<u32> = [&pa[..40], &prompt(3, 20)[..]].concat();
        let (rx_c, _) = add(&mut s, pc.clone(), 8);
        run(&mut s);
        assert_eq!((collect(&rx_a).0, collect(&rx_b).0, collect(&rx_c).0), (reference(&pa, 60), reference(&pb, 8), reference(&pc, 8)));
        // (the third shares 40 tokens with both the others; either way it is before their mark)
        assert_eq!(s.be.shared, vec![(0, 1, 70, 70), (1, 2, 40, 0)]);
        assert_eq!(s.stats.hit, 70);
    }

    /// A request that begins as a kept sequence does, with most of it, where the sequence cannot be gone on with from there (here a recurrent model with no mark): the
    /// kept sequence stays as it is while a slot is free, and its own next turn still goes on in it; when no slot is free it is the one that gives its slot, before
    /// another conversation's.
    #[test]
    fn a_kept_sequence_that_cannot_be_cut_stays_while_a_slot_is_free() {
        fn ask(s: &mut Sched<SlotFake>, p: &[u32]) -> Vec<u32> {
            let (rx, _) = add(s, p.to_vec(), 8);
            run(s);
            let (out, end) = collect(&rx);
            assert_eq!((&out, end), (&reference(p, 8), Finish::Length));
            out
        }
        let (pa, pb) = (prompt(1, 60), prompt(2, 60));
        let mut s = keeping(3);
        s.be.whole_only = true;
        let ra = ask(&mut s, &pa);
        ask(&mut s, &pb);
        // 50 of the first's 67 tokens and others: refused, a free slot taken, both conversations kept still
        let edit: Vec<u32> = [&pa[..50], &prompt(3, 20)[..]].concat();
        ask(&mut s, &edit);
        assert_eq!((s.be.resumed.clone(), s.be.emptied, s.kept()), (vec![(0, 50, 0)], 0, 3));
        // the first conversation's next turn goes on in its slot, whole
        let next: Vec<u32> = [&pa[..], &ra[..], &prompt(4, 5)[..]].concat();
        ask(&mut s, &next);
        assert_eq!(s.be.resumed.last(), Some(&(0, 67, 67)));
        // no slot free now: a request that begins as the second does and cannot go on in it has that one's slot, not the oldest's
        let edit: Vec<u32> = [&pb[..50], &prompt(5, 20)[..]].concat();
        ask(&mut s, &edit);
        assert_eq!((s.be.resumed.last(), s.be.emptied, s.kept()), (Some(&(1, 50, 0)), 1, 3));
        assert!(s.kept.iter().any(|k| k.slot == 1 && k.tokens[..50] == pb[..50] && k.tokens[50] != pb[50]), "the slot of the sequence that refused, taken");
    }

    /// A kept sequence is its scope's (`Submit::salt`): a request of another scope that begins with its very tokens runs from its first token in another slot, and is told
    /// it found nothing; the next request of the scope still goes on in it. On pages the same: a scope's pages are not another's.
    #[test]
    fn a_kept_sequence_is_found_by_its_scope_alone() {
        fn ask<B: ModelStep>(s: &mut Sched<B>, prompt: &[u32], salt: u64) -> (Vec<u32>, u32) {
            let (tx, rx) = channel();
            let cached = Arc::new(AtomicU32::new(0));
            s.submit(Submit {
                prompt: prompt.to_vec(),
                reply: reply(8),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: cached.clone(),
                created: Default::default(),
                salt,
                pin: 0,
                images: Vec::new(),
            });
            run(s);
            (collect(&rx).0, cached.load(Relaxed))
        }
        let p1 = prompt(1, 40);
        let mut s = keeping(3);
        let (r1, _) = ask(&mut s, &p1, 7);
        let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 10)[..]].concat();
        // another scope, and no scope: the same tokens, nothing found
        assert_eq!(ask(&mut s, &p2, 9), (reference(&p2, 8), 0));
        assert_eq!(ask(&mut s, &p2, 0), (reference(&p2, 8), 0));
        assert_eq!((s.be.resumed.len(), s.be.begun.len(), s.kept()), (0, 3, 3));
        // its own scope: from what the first left
        assert_eq!(ask(&mut s, &p2, 7), (reference(&p2, 8), 47));
        assert_eq!(s.be.resumed, vec![(0, 47, 47)]);
        // pages
        let mut s = Sched::new(Fake::new(64, 32), 2);
        ask(&mut s, &p1, 7);
        assert_eq!(ask(&mut s, &p2, 9), (reference(&p2, 8), 0));
        assert_eq!(ask(&mut s, &p2, 7), (reference(&p2, 8), 2 * PAGE as u32));
        assert_eq!(ask(&mut s, &p2, 0), (reference(&p2, 8), 0));
    }

    /// A request that begins with too little of a kept sequence does not take its slot: fewer than KEEP_LEAST tokens, or less than half of what is kept. It runs in a slot of
    /// its own (with the start's full blocks, found by name), and the kept conversation's next turn still finds its own.
    #[test]
    fn a_kept_sequence_is_not_cut_down_for_a_short_shared_start() {
        let mut s = keeping(3);
        let p1 = prompt(1, 100);
        let r1 = turn(&mut s, &p1, 8);
        // the same first 40 tokens (less than half of the 107 kept), then another prompt's: a slot of its own, the start's 2 full blocks found by name
        let other: Vec<u32> = [&p1[..40], &prompt(9, 30)[..]].concat();
        assert_eq!(turn(&mut s, &other, 8), reference(&other, 8));
        assert!(s.be.resumed.is_empty() && s.be.adopted.last() == Some(&(1, 32)) && s.kept() == 2);
        // a short one that shares all of its 20 tokens but the last (fewer than KEEP_LEAST): a slot of its own too, its one full block found
        let short = p1[..20].to_vec();
        assert_eq!(turn(&mut s, &short, 4), reference(&short, 4));
        assert!(s.be.resumed.is_empty() && s.be.adopted.last() == Some(&(2, 16)) && s.be.begun.len() == 1);
        // the first conversation's next turn finds its slot as it left it
        let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 6)[..]].concat();
        assert_eq!(turn(&mut s, &p2, 8), reference(&p2, 8));
        assert_eq!(s.be.resumed, vec![(0, 107, 107)]);
        assert_eq!(s.stats.hit, 32 + 16 + 107);
    }

    /// Kept sequences never keep a request waiting: with no slot free the oldest is emptied for it. Of two kept, the one a request begins with more of is the one it goes on in.
    #[test]
    fn the_oldest_kept_sequence_gives_its_slot_to_a_request_that_needs_one() {
        let mut s = keeping(2);
        let (pa, pb) = (prompt(1, 50), prompt(2, 50));
        let (ra, rb) = (turn(&mut s, &pa, 8), turn(&mut s, &pb, 8));
        assert_eq!((s.kept(), s.be.emptied), (2, 0));
        // a third conversation: a's slot (the oldest) is emptied for it
        let pc = prompt(3, 50);
        assert_eq!(turn(&mut s, &pc, 8), reference(&pc, 8));
        assert_eq!((s.kept(), s.be.emptied, s.be.begun.len()), (2, 1, 3));
        // b's next turn still goes on in its slot; a's has c's slot (now the oldest), and finds its 3 full blocks by name though its own slot was emptied
        let b2: Vec<u32> = [&pb[..], &rb[..], &prompt(5, 4)[..]].concat();
        assert_eq!(turn(&mut s, &b2, 8), reference(&b2, 8));
        assert_eq!(s.be.resumed, vec![(1, 57, 57)]);
        let a2: Vec<u32> = [&pa[..], &ra[..], &prompt(6, 4)[..]].concat();
        assert_eq!(turn(&mut s, &a2, 8), reference(&a2, 8));
        assert_eq!((s.be.resumed.len(), s.be.emptied, s.be.begun.len(), s.be.adopted.last().map(|x| x.1)), (1, 2, 3, Some(48)));
        // several at once, more than the slots: every reply right, none waits on a kept slot
        let ps: Vec<Vec<u32>> = (10..15).map(|k| prompt(k, 30 + k as usize)).collect();
        let rx: Vec<_> = ps.iter().map(|p| add(&mut s, p.clone(), 12).0).collect();
        run(&mut s);
        for (p, r) in ps.iter().zip(&rx) {
            assert_eq!(collect(r), (reference(p, 12), Finish::Length));
        }
    }

    /// A model whose state is only what it is after a sequence's last token (a recurrent one): the next turn goes on where the whole of the slot is its start; a prompt that leaves
    /// the kept tokens before their end (a reply asked for again) has the slot emptied and runs from its first token. The replies are right either way.
    #[test]
    fn a_recurrent_models_kept_sequence_goes_on_from_its_end_or_not_at_all() {
        let mut s = keeping(1);
        s.be.whole_only = true;
        let p1 = prompt(1, 40);
        let r1 = turn(&mut s, &p1, 8);
        let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 10)[..]].concat();
        let r2 = turn(&mut s, &p2, 8);
        assert_eq!(r2, reference(&p2, 8));
        assert_eq!((s.be.resumed.clone(), s.be.emptied, s.stats.hit), (vec![(0, 47, 47)], 0, 47));
        // the same prompt again: the slot holds 65 tokens, the prompt shares 57 of them
        assert_eq!(turn(&mut s, &p2, 8), r2);
        assert_eq!((s.be.resumed[1], s.be.emptied, s.stats.hit, s.be.begun.len()), ((0, 57, 0), 1, 47, 2));
    }

    /// P5e: what a request with a pin left (`Submit::pin`: Anthropic's `cache_control`) is held before the other kept sequences when a slot is wanted, for the pin's time;
    /// with only pinned ones kept, the oldest of them gives its slot: a pin never keeps a request waiting.
    #[test]
    fn a_pinned_kept_sequence_is_emptied_after_the_others() {
        fn ask(s: &mut Sched<SlotFake>, prompt: &[u32], pin: u32) -> Vec<u32> {
            let (tx, rx) = channel();
            s.submit(Submit {
                prompt: prompt.to_vec(),
                reply: reply(8),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: Default::default(),
                cached: Default::default(),
                created: Default::default(),
                salt: 0,
                pin,
                images: Vec::new(),
            });
            run(s);
            collect(&rx).0
        }
        let ps: Vec<Vec<u32>> = (1..=4).map(|k| prompt(k, 50)).collect();
        let mut s = keeping(2);
        // the first is pinned, the second is not: a third request takes the second's slot though the first is older
        let r0 = ask(&mut s, &ps[0], 300);
        ask(&mut s, &ps[1], 0);
        assert_eq!(ask(&mut s, &ps[2], 300), reference(&ps[2], 8));
        assert_eq!((s.kept(), s.be.emptied), (2, 1));
        let next: Vec<u32> = [&ps[0][..], &r0[..], &prompt(9, 5)[..]].concat();
        assert_eq!(ask(&mut s, &next, 300), reference(&next, 8));
        assert_eq!(s.be.resumed, vec![(0, 57, 57)], "the pinned conversation was still there");
        // both kept are pinned now: a request that needs a slot has the oldest of them
        assert_eq!(ask(&mut s, &ps[3], 0), reference(&ps[3], 8));
        assert_eq!((s.kept(), s.be.emptied), (2, 2));
        // a pin whose time is over is as none
        let mut s = keeping(2);
        ask(&mut s, &ps[0], 300);
        ask(&mut s, &ps[1], 0);
        s.kept[0].pinned = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        ask(&mut s, &ps[2], 0);
        assert!(s.kept.iter().all(|k| k.slot != 0 || k.tokens[0] == ps[2][0]), "the oldest, its pin over, gave its slot");
    }

    /// A recurrent model that keeps its state at each prompt's end (a mark): a reply asked for again, and a next turn that leaves the reply's first tokens out (a template that
    /// drops the thinking), go on from the mark, the prompt's end; the turn that sends the whole reply goes on from the slot's end.
    #[test]
    fn a_recurrent_models_kept_sequence_goes_on_from_its_prompts_end() {
        for ahead in [false, true] {
            from_the_prompts_end(ahead);
        }
    }

    /// `a_recurrent_models_kept_sequence_goes_on_from_its_prompts_end`, with steps launched ahead or not: a prompt that ends in the step in flight is marked as the next step
    /// is launched, at the same place.
    fn from_the_prompts_end(ahead: bool) {
        let mut s = keeping(1);
        (s.be.whole_only, s.be.marking) = (true, true);
        if ahead {
            (s.be.one_step, s.be.overlap) = (true, true);
            s = s.overlapping(true);
        }
        let p1 = prompt(1, 40);
        assert_eq!(turn(&mut s, &p1, 12), reference(&p1, 12));
        assert_eq!(s.be.marks, vec![40], "the prompt's end is marked");
        // a next turn that leaves the reply out: from the mark (the slot holds 51 tokens, the prompt shares the first 40)
        let longer = [&p1[..], &prompt(7, 20)[..]].concat();
        assert_eq!(turn(&mut s, &longer, 8), reference(&longer, 8));
        assert_eq!((s.be.resumed[0], s.stats.hit, s.be.emptied, s.be.begun.len()), ((0, 40, 40), 40, 0, 1));
        assert_eq!(s.be.marks, vec![60], "and its own prompt's end is the mark now");
        // a turn that sends a prompt and its whole reply goes on from the slot's end, past the mark
        let r = reference(&longer, 8);
        let next: Vec<u32> = [&longer[..], &r[..], &prompt(8, 9)[..]].concat();
        assert_eq!(turn(&mut s, &next, 8), reference(&next, 8));
        assert_eq!((s.be.resumed[1], s.stats.hit), ((0, 67, 67), 107));
        // a prompt that leaves the kept tokens before the mark: nothing to go on from
        let early = [&next[..50], &prompt(9, 40)[..]].concat();
        assert_eq!(turn(&mut s, &early, 4), reference(&early, 4));
        assert_eq!((s.be.resumed[2], s.be.emptied, s.be.begun.len()), ((0, 50, 0), 1, 2));
        assert_eq!(s.be.ahead > 0, ahead, "steps launched ahead");
    }

    /// Rooms that take memory as they fill: a kept sequence's room is given up before a request waits for room or a running sequence is sent back.
    #[test]
    fn kept_sequences_give_their_room_up_first() {
        let mut f = SlotFake::new(3, 512, 16);
        (f.keeps, f.budget) = (true, Some(160));
        let mut s = Sched::new(f, 3);
        let pa = prompt(1, 90);
        assert_eq!(turn(&mut s, &pa, 8), reference(&pa, 8));
        assert_eq!((s.kept(), s.be.held.iter().sum::<usize>()), (1, 112));
        // a prompt that needs more than what is left beside the kept one: the kept one goes, nothing waits
        let pb = prompt(2, 100);
        assert_eq!(turn(&mut s, &pb, 8), reference(&pb, 8));
        assert_eq!((s.kept(), s.be.emptied, s.stats.preempted), (1, 1, 0));
        // a reply that grows past what is left while another is kept: the kept one's room, not a running sequence's
        let pc = prompt(3, 30);
        assert_eq!(turn(&mut s, &pc, 60), reference(&pc, 60));
        assert_eq!((s.be.emptied, s.stats.preempted), (2, 0));
    }

    /// Cancellation: a waiting request asked to end ends stopped without running; a running one ends stopped at its next round, its slot emptied and given to the next.
    #[test]
    fn a_cancelled_request_frees_its_slot_for_the_next() {
        let mut s = Sched::new(SlotFake::new(2, 256, 16), 2);
        let (rx_a, stop_a) = add(&mut s, prompt(1, 10), 100);
        let (rx_b, _) = add(&mut s, prompt(2, 10), 20);
        let (rx_c, stop_c) = add(&mut s, prompt(3, 10), 20);
        let (rx_d, _) = add(&mut s, prompt(4, 10), 20);
        stop_c.store(true, Relaxed);
        for _ in 0..5 {
            s.tick().unwrap();
        }
        assert!(s.be.rooms.iter().all(|r| !r.is_empty()), "a and b run");
        stop_a.store(true, Relaxed);
        run(&mut s);
        let (a, fa) = collect(&rx_a);
        assert_eq!(fa, Finish::Stopped);
        assert!(!a.is_empty() && a.len() < 100 && a[..] == reference(&prompt(1, 10), a.len())[..]);
        assert_eq!(collect(&rx_b), (reference(&prompt(2, 10), 20), Finish::Length));
        assert_eq!(collect(&rx_c), (Vec::new(), Finish::Stopped));
        assert_eq!(collect(&rx_d), (reference(&prompt(4, 10), 20), Finish::Length));
        assert_eq!(s.be.emptied, 3, "a, b and d emptied their slots; c never had one");
        assert!(s.be.rooms.iter().all(Vec::is_empty));
    }

    /// A prompt a slot cannot hold with a token after it ends with an error in words; the others run. A reply ends at the slot's end (the length: the slot full, and the token after it).
    #[test]
    fn a_slot_step_refuses_a_prompt_past_its_window() {
        let mut s = Sched::new(SlotFake::new(2, 64, 16), 2);
        let (big, _) = add(&mut s, prompt(1, 64), 4);
        let (ok, _) = add(&mut s, prompt(2, 60), 10);
        run(&mut s);
        let (none, e) = outcome(&big);
        assert!(none.is_empty() && e.as_ref().is_err_and(|m| m.contains("64 tokens")), "{e:?}");
        assert_eq!(collect(&ok), (reference(&prompt(2, 60), 5), Finish::Length));
    }

    /// Preemption on pages: where pages run out the newest sequence is the one sent back (it is resumed, the oldest never is), and every reply is still whole.
    #[test]
    fn the_newest_sequence_gives_its_pages_back() {
        let mut s = Sched::new(Fake::new(12, 16), 4);
        let mut subs = Vec::new();
        for k in 0..4u32 {
            let (tx, rx) = channel();
            let resumed = Arc::new(AtomicU32::new(0));
            s.submit(Submit {
                prompt: prompt(k + 1, 20),
                reply: reply(30),
                out: tx,
                abort: Arc::new(AtomicBool::new(false)),
                resumed: resumed.clone(),
                cached: Default::default(),
                created: Default::default(),
                salt: 0,
                pin: 0,
                images: Vec::new(),
            });
            subs.push((rx, resumed));
        }
        run(&mut s);
        for (k, (rx, _)) in subs.iter().enumerate() {
            assert_eq!(collect(rx), (reference(&prompt(k as u32 + 1, 20), 30), Finish::Length));
        }
        let resumed: Vec<u32> = subs.iter().map(|(_, r)| r.load(Relaxed)).collect();
        assert!(s.stats.preempted > 0 && resumed[0] == 0, "{resumed:?}");
        assert!(resumed[3] >= resumed[1], "the newest goes back first: {resumed:?}");
    }

    /// Rooms that fill as their sequences grow: where the memory runs out the newest sequence gives its room back and runs again later, and every reply is still the one it has alone; a
    /// prompt that cannot fit even with nothing else running ends with an error in words, and the rest run.
    #[test]
    fn rooms_that_run_out_take_back_the_newest_and_every_reply_is_whole() {
        for (slots, budget) in [(4, 160), (8, 256), (8, 96)] {
            let mut f = SlotFake::new(slots, 512, 16);
            f.budget = Some(budget);
            let mut s = Sched::new(f, slots);
            let reqs: Vec<(Vec<u32>, usize)> = (0..10).map(|k| (prompt(k as u32 + 7, 5 + 7 * k), 20 + 3 * k)).collect();
            let rx: Vec<_> = reqs.iter().map(|(p, n)| add(&mut s, p.clone(), *n).0).collect();
            run(&mut s);
            for ((p, n), rx) in reqs.iter().zip(&rx) {
                match outcome(rx) {
                    (got, Ok(f)) => assert_eq!((got, f), (reference(p, *n), Finish::Length), "budget {budget}"),
                    (_, Err(e)) => assert!(p.len() + n + 1 > budget && e.contains("fewer with nothing else running"), "budget {budget}: {e}"),
                }
            }
            assert!(s.stats.preempted > 0, "budget {budget}: the rooms ran out");
            assert!(s.be.held.iter().all(|&h| h == 0) && s.be.rooms.iter().all(Vec::is_empty));
        }
        let mut f = SlotFake::new(2, 512, 16);
        f.budget = Some(64);
        let mut s = Sched::new(f, 2);
        let (big, _) = add(&mut s, prompt(1, 100), 4);
        let (ok, _) = add(&mut s, prompt(2, 20), 8);
        run(&mut s);
        assert!(outcome(&big).1.is_err_and(|e| e.contains("100 tokens")));
        assert_eq!(collect(&ok), (reference(&prompt(2, 20), 8), Finish::Length));
    }

    /// One sequence alone on the executor's blocks, in pools of one sequence's (a full group of the context, a sliding group of its window and a chunk): a prompt of several windows
    /// runs to its reply, and so does a conversation's next turn that adds more than a chunk to the kept one, each pass reading every token from the block that holds it. (The room for a
    /// whole prompt held the sliding group's blocks of the prompt's last window, so its first pass found none: the sequence went back to wait and was let in again, without end. Gemma 4
    /// served to one user never answered a prompt of 4,000 tokens. In pools of two, the next turn's first pass read the window before it from blocks given back and taken again.)
    #[test]
    fn a_prompt_of_several_windows_runs_alone_in_a_pool_of_one_window() {
        for (slots, one_step) in [(1, false), (1, true), (2, true)] {
            // (a window of 64 tokens and a chunk of 32, as Gemma 4's 1,024 and 512)
            let mut f = SlotFake::paged(slots, 512, 32, 64);
            (f.one_step, f.keeps) = (one_step, true);
            let mut s = Sched::new(f, slots);
            let p1 = prompt(1, 300);
            let r1 = turn(&mut s, &p1, 8);
            assert_eq!(r1, reference(&p1, 8));
            let p2: Vec<u32> = [&p1[..], &r1[..], &prompt(2, 100)[..]].concat();
            assert_eq!(turn(&mut s, &p2, 8), reference(&p2, 8));
            assert_eq!((s.be.resumed.clone(), s.stats.preempted), (vec![(0, 307, 307)], 0), "the next turn went on in its slot");
            s.clear_cache().unwrap();
            let p = s.be.paged.as_ref().expect("paged");
            assert!((0..2).all(|g| p.free(g).0 == p.free(g).1), "every block back");
        }
    }

    /// A sequence alone that its step has no room for, for its next pass, ends at once with an error in words: sent back to wait, it was let in again to the same end, without end.
    /// (A step whose room for a prompt holds a sliding group's blocks of the prompt's last window, as `Pages::grow` gives them: its first pass has none.) The scheduler is idle then,
    /// and the next request is answered.
    #[test]
    fn a_sequence_that_cannot_grow_alone_ends_with_an_error() {
        let mut f = SlotFake::paged(1, 512, 32, 64);
        f.jumps = true;
        let mut s = Sched::new(f, 1);
        let (rx, _) = add(&mut s, prompt(1, 300), 8);
        for _ in 0..3 {
            if !s.idle() {
                s.tick().unwrap();
            }
        }
        assert!(s.idle() && s.stats.preempted == 0, "{} preempted", s.stats.preempted);
        assert!(outcome(&rx).1.is_err_and(|e| e.contains("fewer with nothing else running")));
        let p = prompt(2, 50);
        assert_eq!(turn(&mut s, &p, 8), reference(&p, 8));
    }

    // ---- preemption by swapping (R4d) ----------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Ten requests on rooms that run out (each fits alone), swapped as `mode` says: the replies, the prompt tokens the model ran, and the scheduler's figures, the slots and the host's area
    /// checked empty at the end.
    fn squeezed(slots: usize, budget: usize, mode: SwapMode, set: impl Fn(&mut SlotFake)) -> (Vec<(Vec<u32>, usize)>, Vec<(Vec<u32>, Finish)>, usize, Stats) {
        let mut f = SlotFake::new(slots, 512, 16);
        (f.budget, f.host_room) = (Some(budget), 1 << 20);
        set(&mut f);
        let mut s = Sched::new(f, slots).swapping(mode);
        let reqs: Vec<(Vec<u32>, usize)> = (0..10).map(|k| (prompt(k as u32 + 7, 5 + 7 * k), 20 + 3 * k)).collect();
        assert!(reqs.iter().all(|(p, n)| p.len() + n < budget), "a request that does not fit alone");
        let rx: Vec<_> = reqs.iter().map(|(p, n)| add(&mut s, p.clone(), *n).0).collect();
        run(&mut s);
        let got = rx.iter().map(collect).collect();
        let f = &s.be;
        assert!(f.held.iter().all(|&h| h == 0) && f.rooms.iter().all(Vec::is_empty) && f.loading.iter().all(|&l| l == 0));
        assert!(f.host.is_empty(), "copies left in the host's area");
        assert_eq!(s.available(), slots);
        (reqs, got, f.ran, s.stats)
    }

    /// The acceptance of R4d on the made-up model: where the rooms run out, the sequences sent back are swapped to the host and copied back, and every reply is the reply alone, token for
    /// token; no prompt token is run twice (recompute-only runs the ones sent back again).
    #[test]
    fn swapped_sequences_go_on_with_the_tokens_they_have_alone() {
        // (and where steps are launched before the one before lands: a sequence of the step in flight that is swapped has that step land first)
        for (slots, budget, ahead) in [(4, 160, false), (8, 256, false), (8, 128, false), (2, 128, false), (4, 160, true), (8, 128, true), (2, 128, true)] {
            let launched = |f: &mut SlotFake| {
                if ahead {
                    (f.one_step, f.batch, f.overlap) = (true, slots, true);
                }
            };
            let (reqs, got, ran, st) = squeezed(slots, budget, SwapMode::Always, launched);
            for ((p, n), g) in reqs.iter().zip(&got) {
                assert_eq!(g, &(reference(p, *n), Finish::Length), "{slots} slots, budget {budget}, launched ahead {ahead}");
            }
            let prompts: usize = reqs.iter().map(|(p, _)| p.len()).sum();
            // (every one that held tokens: one sent back before its first chunk has nothing to copy, nor to run again)
            assert!(st.swapped > 0, "{slots} slots, budget {budget}, launched ahead {ahead}: {} sent back, none swapped", st.preempted);
            assert_eq!(ran, prompts, "{slots} slots, budget {budget}, launched ahead {ahead}: prompt tokens run again");
            let (_, again, ran_again, st_again) = squeezed(slots, budget, SwapMode::Never, launched);
            assert_eq!(again, got);
            assert!(st_again.preempted > 0 && st_again.swapped == 0 && ran_again > prompts, "recompute ran {ran_again} prompt tokens, swap {ran}");
        }
    }

    /// Each sequence sent back is swapped where its bytes over the link, out and back, cost less than its tokens' recompute at the rate the prompt passes ran at, and run again where they cost
    /// more; every reply whole either way. (The two rates are a million times apart, so the choice does not turn on the test's timings.)
    #[test]
    fn the_cheaper_way_back_is_chosen_for_each_sequence() {
        for (link, swaps) in [(1e15, true), (1.0, false)] {
            let (reqs, got, _, st) = squeezed(4, 160, SwapMode::ByCost, |f| f.link = link);
            for ((p, n), g) in reqs.iter().zip(&got) {
                assert_eq!(g, &(reference(p, *n), Finish::Length));
            }
            assert!(st.preempted > 0 && st.pass_tokens > 0 && st.pass_ns > 0);
            assert_eq!(st.swapped > 0, swaps, "link {link} B/s: {} sent back, {} swapped", st.preempted, st.swapped);
        }
    }

    /// A host area too small for every sequence sent back: those it cannot take are run again from their tokens, and every reply is still whole.
    #[test]
    fn a_full_swap_area_sends_the_rest_back_to_recompute() {
        let (reqs, got, ran, st) = squeezed(8, 128, SwapMode::Always, |f| f.host_room = 40);
        for ((p, n), g) in reqs.iter().zip(&got) {
            assert_eq!(g, &(reference(p, *n), Finish::Length));
        }
        assert!(st.swapped > 0 && st.swapped < st.preempted, "{} sent back, {} swapped", st.preempted, st.swapped);
        assert!(ran > reqs.iter().map(|(p, _)| p.len()).sum::<usize>());
    }

    /// A copy back takes a while (`swap_ready` says no for a few polls): the sequence steps only once it is done, and the others' steps go on meanwhile.
    #[test]
    fn a_sequence_copied_back_steps_once_its_copy_is_done() {
        let (reqs, got, _, st) = squeezed(4, 160, SwapMode::Always, |f| f.load_polls = 3);
        for ((p, n), g) in reqs.iter().zip(&got) {
            assert_eq!(g, &(reference(p, *n), Finish::Length));
        }
        assert!(st.swapped > 0);
        let mut f = SlotFake::new(4, 512, 16);
        (f.budget, f.host_room, f.load_polls) = (Some(160), 1 << 20, 3);
        let mut s = Sched::new(f, 4).swapping(SwapMode::Always);
        let rx: Vec<_> = (0..6).map(|k| add(&mut s, prompt(k + 3, 30), 40).0).collect();
        run(&mut s);
        for (k, r) in rx.iter().enumerate() {
            assert_eq!(collect(r), (reference(&prompt(k as u32 + 3, 30), 40), Finish::Length));
        }
        assert!(s.stats.swapped > 0 && s.be.beside_copies > 0, "no step ran while a copy back was in flight");
    }

    /// A request stopped while its sequence is swapped (waiting, or being copied back) lets its copy go; the others' replies are whole and the host's area is empty at the end.
    #[test]
    fn a_request_stopped_while_swapped_lets_its_copy_go() {
        for polls in [0, 50] {
            let mut f = SlotFake::new(2, 512, 16);
            (f.budget, f.host_room, f.load_polls) = (Some(64), 1 << 20, polls);
            let mut s = Sched::new(f, 2).swapping(SwapMode::Always);
            let (ra, _) = add(&mut s, prompt(1, 20), 30);
            let (rb, stop_b) = add(&mut s, prompt(2, 20), 30);
            let mut guard = 0;
            while s.stats.swapped == 0 {
                s.tick().unwrap();
                guard += 1;
                assert!(guard < 1000, "nothing swapped");
            }
            stop_b.store(true, Relaxed);
            run(&mut s);
            assert_eq!(collect(&ra), (reference(&prompt(1, 20), 30), Finish::Length));
            let (b, fb) = collect(&rb);
            assert_eq!(fb, Finish::Stopped);
            assert!(b.len() < 30 && b[..] == reference(&prompt(2, 20), b.len())[..]);
            assert!(s.be.host.is_empty() && s.be.loading.iter().all(|&l| l == 0) && s.be.rooms.iter().all(Vec::is_empty));
        }
    }
}
