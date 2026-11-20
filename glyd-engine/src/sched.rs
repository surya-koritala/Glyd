//! Several sequences at once on the paged cache: sequences wait, take pages (a prompt that begins as a finished one did finds its pages in the cache), run their prompt a chunk at a time, and decode together, one
//! step for all of them. Where the pages run out, the newest sequence gives its pages back and goes to the head of the queue with what it has made; it is run again from its tokens (recomputed) and its stream goes on.
//!
//! The cache's extra memory is shared and finite. After every step the backend says which blocks the step could not store exactly (they are named lost) and which it kept in the scarce second extra memory (heavy). A sequence
//! that holds a lost block has the step's output taken back, never used or sent: it gives its pages back and waits until the sequences that were running with it have finished, then runs again from its tokens; one that is lost
//! alone twice, or six times, ends with an error in words. Lost and heavy blocks are never kept in the prefix cache.

use crate::ffi::Result;
use crate::kv::{chain_key, Lost, Pager, ALONE_MESSAGE, PAGE};
use crate::model::{Engine, Finish, PrefillJob, Reply, Row, Sample, FUSED_ROWS, MAX_ROWS};
use crate::sampler::{logprobs_of, Lp, Sampler};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// What a request is sent when its sequence runs: each token as it is chosen, and last how the sequence ended. Where the request asks for log-probabilities, an `Lp` of a token comes before the token
/// (the end-of-sequence token's, which is no token of the reply, before the end).
pub enum Ev {
    Token(u32),
    Lp(Lp),
    End(std::result::Result<Finish, String>),
}

/// A sequence to run: its prompt, what the reply is asked to be, where its tokens go, and a flag that ends it (the request has all it wants, or has gone).
pub struct Submit {
    pub prompt: Vec<u32>,
    pub reply: Reply,
    pub out: Sender<Ev>,
    pub abort: Arc<AtomicBool>,
    /// Counts the times the sequence is sent back to wait and run again (for room, or for a block the cache lost); the caller may read it.
    pub resumed: Arc<AtomicU32>,
    /// The prompt's images in order (their places in it their tokens), for an engine that reads images (src/vision.rs); none for the rest.
    pub images: Vec<Arc<crate::vision::Image>>,
}

/// What a scheduler runs on: the model's steps on pages.
pub trait Backend {
    /// The longest sequence, in tokens.
    fn window(&self) -> usize;
    /// The most prompt tokens a chunk takes.
    fn chunk(&self) -> usize;
    /// Pages in the cache.
    fn pages(&self) -> usize;
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
    /// One step of the rows: the sample after each.
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>>;
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
}

impl Backend for &mut Engine {
    fn window(&self) -> usize {
        self.capacity()
    }
    fn chunk(&self) -> usize {
        Engine::chunk(self)
    }
    fn pages(&self) -> usize {
        self.pool_pages()
    }
    fn eos(&self) -> Vec<i64> {
        self.w.cfg.eos.clone()
    }
    fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        self.prefill_chunk(pages, start, tokens, last)
    }
    fn prefill_rows(&self) -> usize {
        // (the prompts of up to 8 sequences in a pass; GLYD_ENGINE_BATCH_PREFILL=0: one prompt a pass, as before. A pass of more than 64 rows takes the products' route its rows give, so with several prompts at
        // once a reply's last bits can differ from the same prompt's alone (the same token in nearly every case: `glyd-engine batchcheck`); =0 keeps every reply identical to solo)
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_none_or(|v| v != "0")) {
            MAX_ROWS
        } else {
            1
        }
    }
    fn prefill_together(&self) -> usize {
        FUSED_ROWS
    }
    fn prefill_many(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
        self.prefill_batch(jobs)
    }
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        self.decode_batch(rows)
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        self.read_logits_row(row)
    }
    fn release(&mut self, pages: &[i32]) -> Result<()> {
        self.release_pages(pages)
    }
    fn verify(&mut self, pages: &[i32]) -> Result<()> {
        self.verify_pages(pages)
    }
    fn lost(&mut self) -> Lost {
        self.take_lost()
    }
    fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        self.candidates(row)
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
    pub by_rows: [u64; MAX_ROWS + 1],
    pub prefill_rounds: u64,
}

/// The scheduler's figures as the last round left them (one scheduler runs in a process), for the server's /stats.
pub static SHARED: [AtomicU64; 28] = [const { AtomicU64::new(0) }; 28];

/// The figures as JSON: prompt tokens run or found in the cache, those found, sequences sent back to wait for pages, steps, rows in them, sequences running and waiting.
pub fn stats_json() -> String {
    let v: Vec<u64> = SHARED.iter().map(|a| a.load(Relaxed)).collect();
    let rate = if v[0] > 0 { 100.0 * v[1] as f64 / v[0] as f64 } else { 0.0 };
    // a development build's figures are all of them; a release build's are the counts a user reads
    if cfg!(feature = "dev") {
        format!(
            "{{\"prompt_tokens\":{},\"prefix_hit_tokens\":{},\"prefix_hit_rate_percent\":{rate:.2},\"preemptions\":{},\"steps\":{},\"rows\":{},\"running\":{},\"waiting\":{},\"decode_ms\":{},\"prefill_ms\":{},\"rounds_ms\":{},\"release_ms\":{},\"admit_ms\":{},\"after_ms\":{},\"before_ms\":{},\"lost_blocks\":{},\"heavy_blocks\":{},\"taken_back\":{},\"ended_for_loss\":{},\"steps_by_rows\":[{}],\"prefill_rounds\":{},\"decode_prep_ms\":{},\"decode_wait_ms\":{},\"decode_post_ms\":{},\"decode_graph_ms\":{},\"values_checked\":{},\"values_differ\":{},\"values_in_lost_blocks\":{}}}",
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7] / 1_000_000, v[8] / 1_000_000, v[9] / 1_000_000, v[10] / 1_000_000, v[11] / 1_000_000, v[12] / 1_000_000, v[13] / 1_000_000, v[14], v[15], v[16], v[17], v[18..27].iter().map(u64::to_string).collect::<Vec<_>>().join(","), v[27], crate::model::DECODE_PREP_NS.load(Relaxed) / 1_000_000, crate::model::DECODE_WAIT_NS.load(Relaxed) / 1_000_000, crate::model::DECODE_POST_NS.load(Relaxed) / 1_000_000, crate::model::DECODE_GPU_NS.load(Relaxed) / 1_000_000, crate::model::VERIFY_CHECKED.load(Relaxed), crate::model::VERIFY_DIFFER.load(Relaxed), crate::model::VERIFY_LOST.load(Relaxed)
        )
    } else {
        format!("{{\"prompt_tokens\":{},\"prefix_hit_tokens\":{},\"prefix_hit_rate_percent\":{rate:.2},\"preemptions\":{},\"steps\":{},\"rows\":{},\"running\":{},\"waiting\":{}}}", v[0], v[1], v[2], v[3], v[4], v[5], v[6])
    }
}

struct Seq {
    id: u64,
    out: Sender<Ev>,
    abort: Arc<AtomicBool>,
    resumed: Arc<AtomicU32>,
    reply: Reply,
    greedy: bool,
    sampler: Sampler,
    /// The prompt and what has been chosen so far. While decoding, the last token is the one the next step takes (its keys and values are not in the cache yet).
    tokens: Vec<u32>,
    generated: usize,
    pages: Vec<i32>,
    /// Tokens whose keys and values are in the pages.
    done: usize,
    /// Full pages given to the cache, and the key of the last.
    published: usize,
    chain: u64,
    prefilling: bool,
    /// Times a step of this sequence was taken back for a lost block, and the sequences that must have ended before it runs again (they were running when it lost one).
    tries: u32,
    after: Vec<u64>,
    /// Hit twice while others ran: it runs alone next (nothing else is let in until it ends).
    solo: bool,
    /// When it was sent to the scheduler and when its last token was written (the metrics' first-token and between-token times).
    began: std::time::Instant,
    last_token: Option<std::time::Instant>,
}

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

/// The most tokens a prompt pass has for the round's decode step to wait on the prompts that are coming in, and the most rounds in a row it does.
const SHORT_PASS: usize = 256;
const MAX_SKIPS: u32 = 4;
const MAX_LONG_SKIPS: u32 = 2;

pub struct Sched<B: Backend> {
    be: B,
    pager: Pager,
    users: usize,
    waiting: VecDeque<Seq>,
    active: Vec<Seq>,
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
}

impl<B: Backend> Sched<B> {
    /// A scheduler for up to `users` sequences at a step (1 to MAX_ROWS).
    pub fn new(be: B, users: usize) -> Sched<B> {
        let pager = Pager::new(be.pages());
        let eos = be.eos();
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
            pager,
            users: users.clamp(1, MAX_ROWS),
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_id: 0,
            eos,
            stats: Stats::default(),
            trace: crate::tune!("GLYD_ENGINE_TRACE_SCHED").is_some(),
            pass_tokens: 0,
            skips: 0,
        }
    }

    pub fn idle(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty()
    }

    /// Pages that can be had now (free or only cached).
    pub fn available(&self) -> usize {
        self.pager.available()
    }

    /// Every cached page that no sequence holds given up and cleared (the extra memory they hold goes back): of a scheduler that is idle, nothing is held.
    pub fn clear_cache(&mut self) -> Result<()> {
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
        self.waiting.push_back(Seq {
            id: self.next_id,
            out: s.out,
            abort: s.abort,
            resumed: s.resumed,
            reply: s.reply,
            greedy,
            sampler,
            tokens: s.prompt,
            generated: 0,
            pages: Vec::new(),
            done: 0,
            published: 0,
            chain: 0,
            prefilling: true,
            tries: 0,
            after: Vec::new(),
            solo: false,
            began: std::time::Instant::now(),
            last_token: None,
        });
        self.gauges();
    }

    fn gauges(&self) {
        let st = &self.stats;
        let pages = self.pager.pages();
        crate::ops::KV_PAGES.store(pages as u64, Relaxed);
        crate::ops::KV_PAGES_HELD.store(pages.saturating_sub(self.pager.available()) as u64, Relaxed);
        for (a, v) in SHARED.iter().zip([
            st.prompt,
            st.hit,
            st.preempted,
            st.steps,
            st.rows,
            self.active.len() as u64,
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
        ]) {
            a.store(v, Relaxed);
        }
    }

    /// One round: the sequences that are asked to end end, waiting ones are taken in while there is room, one prompt chunk, one decode step. On an error nothing is sure of the pages: the caller ends every sequence (`fail`).
    pub fn tick(&mut self) -> Result<()> {
        let t = std::time::Instant::now();
        let ta = std::time::Instant::now();
        self.reap()?;
        let came = self.stats.prompt;
        self.admit()?;
        let admitted = self.stats.prompt != came;
        self.stats.admit_ns += ta.elapsed().as_nanos() as u64;
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
            self.stats.tick_ns += t.elapsed().as_nanos() as u64;
            self.gauges();
            return Ok(());
        }
        self.skips = 0;
        self.decode_step()?;
        self.flush();
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
    pub fn fail(&mut self, msg: &str) {
        for s in self.active.drain(..).chain(self.waiting.drain(..)) {
            crate::ops::ended("error", s.began);
            self.pending.push((s.out.clone(), Ev::End(Err(msg.to_string()))));
        }
        self.pager = Pager::new(self.be.pages());
        // (nothing is sure of the pages: every one is cleared, so that the extra memory they hold goes back)
        let _ = self.be.lost();
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
        let gone: Vec<usize> = self.waiting.iter().enumerate().filter(|(_, s)| s.abort.load(Relaxed)).map(|(i, _)| i).collect();
        for i in gone.into_iter().rev() {
            if let Some(s) = self.waiting.remove(i) {
                crate::ops::ended("stopped", s.began);
                self.pending.push((s.out.clone(), Ev::End(Ok(Finish::Stopped))));
            }
        }
        Ok(())
    }

    /// Whether the sequence with this id is still there (running or waiting).
    fn alive(&self, id: u64) -> bool {
        self.active.iter().any(|s| s.id == id) || self.waiting.iter().any(|s| s.id == id)
    }

    /// A waiting sequence that lost a block waits until the sequences that were running with it are gone.
    fn held(&self, s: &Seq) -> bool {
        s.after.iter().any(|&id| self.alive(id))
    }

    /// Waiting sequences taken in, the oldest first (those that are not held back), while there are places and pages for the whole prompt and a page for each running sequence to grow into.
    fn admit(&mut self) -> Result<()> {
        while self.active.len() < self.users {
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
            let need = self.waiting[k].tokens.len().div_ceil(PAGE) + 1;
            if self.pager.available() < need + self.active.len() {
                if self.active.is_empty() {
                    // nothing will free a page: this one never fits
                    let s = self.waiting.remove(k).expect("a waiting sequence");
                    crate::ops::ended("error", s.began);
                    self.pending.push((s.out.clone(), Ev::End(Err(format!("{} tokens, where the cache holds {} blocks", s.tokens.len(), self.pager.pages())))));
                    continue;
                }
                break;
            }
            let mut s = self.waiting.remove(k).expect("a waiting sequence");
            // the pages of an earlier sequence with the same first tokens (at least the last token is run, for the logits)
            let hit = self.pager.lookup(&s.tokens, s.tokens.len().saturating_sub(1) / PAGE);
            s.chain = 0;
            for (i, _) in hit.iter().enumerate() {
                s.chain = chain_key(s.chain, &s.tokens[i * PAGE..(i + 1) * PAGE]);
            }
            s.published = hit.len();
            s.done = hit.len() * PAGE;
            s.pages = hit;
            s.prefilling = true;
            s.after.clear();
            self.stats.prompt += s.tokens.len() as u64;
            self.stats.hit += s.done as u64;
            self.active.push(s);
        }
        Ok(())
    }

    /// Pages for the sequence `id`'s first `upto` tokens, the newest sequences giving theirs back where there are none to take. False where `id` itself was sent back to wait.
    fn ensure_pages(&mut self, id: u64, upto: usize) -> Result<bool> {
        loop {
            let Some(i) = self.active.iter().position(|s| s.id == id) else { return Ok(false) };
            if self.active[i].pages.len() >= upto.div_ceil(PAGE) {
                return Ok(true);
            }
            match self.pager.alloc() {
                Some(p) => self.active[i].pages.push(p),
                None => {
                    let victim = self.active.iter().enumerate().max_by_key(|(_, s)| s.id).map(|(v, _)| v).expect("an active sequence");
                    self.preempt(victim)?;
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
        Ok(self.active.iter().filter(|s| l.overflow || s.pages.iter().any(|p| void.contains(p))).map(|s| s.id).collect())
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
                let mut s = self.active.remove(i);
                self.give_back(&mut s)?;
                crate::ops::ended("error", s.began);
                self.pending.push((s.out.clone(), Ev::End(Err(ALONE_MESSAGE.to_string()))));
                continue;
            }
            let solo = !alone && self.active[i].tries >= 2;
            self.preempt(i)?;
            let s = self.waiting.front_mut().expect("it was just sent back to wait");
            s.solo |= solo;
            s.after = if s.solo { Vec::new() } else { others.iter().chain(before.iter()).copied().collect() };
            before.push(id);
        }
        Ok(())
    }

    /// A sequence sent back to wait: its full pages go to the cache, the rest are cleared, and it is first in the queue, to be run again from its tokens.
    fn preempt(&mut self, i: usize) -> Result<()> {
        let mut s = self.active.remove(i);
        self.give_back(&mut s)?;
        s.done = 0;
        s.published = 0;
        s.chain = 0;
        s.prefilling = true;
        s.resumed.fetch_add(1, Relaxed);
        self.stats.preempted += 1;
        self.waiting.push_front(s);
        Ok(())
    }

    /// The sequence's full pages given to the cache and all its pages released; those nothing names are cleared on the GPU.
    fn give_back(&mut self, s: &mut Seq) -> Result<()> {
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
        let mut s = self.active.remove(i);
        self.give_back(&mut s)?;
        crate::ops::ended(finish_name(f), s.began);
        self.pending.push((s.out.clone(), Ev::End(Ok(f))));
        Ok(())
    }

    /// Full pages of the prompt as far as it has run, offered to the cache at once (a request with the same first tokens that comes now finds them).
    fn publish(&mut self, i: usize) {
        let s = &mut self.active[i];
        while (s.published + 1) * PAGE <= s.done {
            let k = s.published;
            s.chain = self.pager.publish(s.pages[k], s.chain, &s.tokens[k * PAGE..(k + 1) * PAGE]);
            s.published += 1;
        }
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
        crate::ops::token(s.began, &mut s.last_token);
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
            let n = if picks.is_empty() { left.min(budget) } else { left };
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
        picks
            .retain(|&(id, n)| at(&self.active, id).is_some_and(|i| self.active[i].prefilling && self.active[i].pages.len() * PAGE >= self.active[i].done + n));
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
        let samples = self.be.prefill_many(&jobs)?;
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
            self.publish(i);
            if let Some(sample) = sample {
                self.active[i].prefilling = false;
                let (tok, lp) = self.pick(i, sample, my_row)?;
                if let Some(f) = self.accept(i, tok, lp) {
                    self.finish(i, f)?;
                }
            }
        }
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
        let rows: Vec<Row> =
            idx.iter().map(|&i| Row { token: self.active[i].tokens[self.active[i].done], pos: self.active[i].done, pages: &self.active[i].pages }).collect();
        self.stats.before_ns += tb.elapsed().as_nanos() as u64;
        let t = std::time::Instant::now();
        let samples = self.be.decode(&rows)?;
        self.stats.decode_ns += t.elapsed().as_nanos() as u64;
        self.stats.steps += 1;
        self.stats.rows += rows.len() as u64;
        self.stats.by_rows[rows.len()] += 1;
        // (a row whose token could not be stored exactly has the step's output taken back below: its token is not used)
        let hit = self.after_step()?;
        let tc = std::time::Instant::now();
        let mut ended = Vec::new();
        for (row, (&i, sample)) in idx.iter().zip(samples).enumerate() {
            if hit.contains(&self.active[i].id) {
                continue;
            }
            self.active[i].done += 1;
            self.publish(i);
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

    impl Backend for Fake {
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
        fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
            std::thread::sleep(self.decode_delay);
            self.steps += 1;
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

    pub fn add(s: &mut Sched<Fake>, prompt: Vec<u32>, max_new: usize) -> (Receiver<Ev>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let abort = Arc::new(AtomicBool::new(false));
        s.submit(Submit { prompt, reply: reply(max_new), out: tx, abort: abort.clone(), resumed: Default::default(), images: Vec::new() });
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

    pub fn run(s: &mut Sched<Fake>) {
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

    /// The metrics see what a sequence did: its first token and the gaps after it, its tokens, how it ended, and the cache's pages while it holds some (the counters are the process's, other tests add to them: at least these).
    #[test]
    fn a_sequence_is_counted_in_the_metrics() {
        use crate::ops;
        let (first, gaps, whole) = (ops::TTFT.count(), ops::TOKEN_GAP.count(), ops::REQUEST_TIME.count());
        let mut s = Sched::new(Fake::new(64, 32), 4);
        let (rx, _) = add(&mut s, prompt(2, 20), 6);
        s.tick().unwrap();
        assert!(ops::KV_PAGES.load(Relaxed) >= 64 && ops::KV_PAGES_HELD.load(Relaxed) > 0, "the sequence holds pages while it runs");
        run(&mut s);
        let (tokens, f) = collect(&rx);
        assert_eq!((tokens.len(), f), (6, Finish::Length));
        assert!(ops::TTFT.count() > first, "the first token was not counted");
        assert!(ops::TOKEN_GAP.count() >= gaps + 5, "the gaps of the five tokens after the first were not counted");
        assert!(ops::REQUEST_TIME.count() > whole, "the end was not counted");
        let text = ops::render(&ops::State { model: "m", ready: true, life: &ops::Life::default(), engine: true });
        assert!(text.contains("glyd_sequences_finished_total{reason=\"length\"} "), "{text}");
        // a request that is asked to end before it runs is counted as stopped
        let (rx, abort) = add(&mut s, prompt(3, 20), 6);
        abort.store(true, Relaxed);
        run(&mut s);
        assert_eq!(collect(&rx).1, Finish::Stopped);
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
        assert!(matches!(big.recv().unwrap(), Ev::End(Err(_))));
        assert_eq!(collect(&ok).0, reference(&prompt(2, 10), 5));
    }
    /// The figures of /stats are one JSON object: a development build's name what the cache lost, a release build's are the counts a user reads and no more.
    #[test]
    fn stats_are_json_with_the_losses() {
        let v = glyd_gpu::json::parse(&stats_json()).expect("valid JSON");
        let user = ["prompt_tokens", "prefix_hit_tokens", "prefix_hit_rate_percent", "preemptions", "steps", "rows", "running", "waiting"];
        let dev = ["before_ms", "lost_blocks", "heavy_blocks", "taken_back", "ended_for_loss", "values_checked", "values_differ", "values_in_lost_blocks"];
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
                Ev::End(f) => return (out, f),
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
            images: Vec::new(),
        });
        run(&mut s);
        let _ = events(&rx);
        assert_eq!(s.be.logit_reads, 0, "a draw from the candidates reads no row");
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
}
