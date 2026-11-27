//! A model split across several GPUs in one machine (GLY-300 part 1): each GPU runs a stage of the layers (`Stage`) on a thread of its own, which
//! holds that stage's engine (`Dyn::new_stage`: its layers' matrices, caches and states, the embedding on the first, the head on the last). A pass
//! goes stage by stage; the rows' slots (the residual and what the next layer reads) are handed over through host memory, not copied between the
//! GPUs (some machines' direct copies are not to be trusted). A draft head of the model drafts on the last stage (src/dynamic/head.rs): a round's
//! drafts there, its verify pass through every stage (`Dyn::verify_rows`), every stage gone back to the drafts kept (`Dyn::rewind`), the head again
//! on the kept rows there; the tokens are the steps' one at a time, as on one GPU.

use super::{Dyn, HeadOut, ModelDesc, Opts, Stage};
use crate::ffi::{err, Error, Result};
use crate::runtime::config::ModelConfig;
use crate::sched::Sample;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};

enum Cmd {
    Reset,
    /// Rows through the stage; `chunk`: a prompt's chunk (the stage that drafts runs its head on its rows too).
    Rows {
        tokens: Vec<u32>,
        at: usize,
        io: Vec<u16>,
        head: bool,
        chunk: bool,
    },
    /// A round's verify pass through the stage (the head on every row on the last).
    Verify {
        tokens: Vec<u32>,
        at: usize,
        io: Vec<u16>,
    },
    /// The drafting stage's drafts after a token (or the ones given, after its head's own first pass).
    Drafts {
        last: u32,
        given: Option<Vec<u32>>,
    },
    /// The last pass's head on its last rows (each row's greedy token and logit; its logits where asked).
    HeadOut {
        rows: usize,
        words: bool,
    },
    Rewind(usize),
    /// The drafting stage's head again on a round's kept rows.
    HeadAgain {
        at: usize,
        kept: usize,
        next: Vec<u32>,
    },
    Drafting,
    Sample,
    Logits,
    Logprob(f32),
    /// A question of the last pass's head rows, answered on the last stage's GPU as one GPU's step answers it (`RowAsk`).
    Row(RowAsk),
    Describe,
    /// The sequence's prompt and its images (`Dyn::set_images`): the first stage runs the tower over them, each stage turns its rows by their positions.
    Images(Vec<u32>, Vec<std::sync::Arc<crate::vision::Image>>),
}

enum Out {
    Done(Result<()>),
    Rows(Result<Vec<u16>>),
    Tokens(Result<Vec<u32>>),
    Head(Result<HeadOut>),
    Flag(bool),
    Sample(Result<Sample>),
    Logits(Result<Vec<f32>>),
    Logprob(Result<f32>),
    Row(RowGot),
    Text(String),
}

struct Worker {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Out>,
    h: Option<std::thread::JoinHandle<()>>,
}

/// Builds a stage's engine on the GPU of the options it is given.
pub type MakeStage = std::sync::Arc<dyn Fn(Stage, &Opts) -> Result<Dyn> + Send + Sync>;

pub struct Split {
    w: Vec<Worker>,
    /// Tokens in the cache.
    pub n: usize,
    ctx: usize,
    chunk: usize,
    eos: Vec<i64>,
    /// The last stage drafts with a head of the model (`spec_round`).
    drafting: bool,
    /// The images of the next reply's prompt (`Gen::see`).
    next: Vec<std::sync::Arc<crate::vision::Image>>,
}

fn gone() -> Error {
    Error("a GPU stage's thread ended".into())
}

impl Split {
    /// The model of `dir` described by `d`, its layers cut before each of `cuts` (ascending, inside 1..layers), stage k on GPU `devices[k]`
    /// (one more device than cuts; a device may repeat: two stages on one GPU check the hand-over on a machine with one).
    pub fn new(dir: &Path, d: ModelDesc, cuts: &[usize], devices: &[i32], o: &Opts) -> Result<Split> {
        let (nl, eos, dir, m) = (d.layers.len(), d.eos.clone(), dir.to_path_buf(), std::sync::Arc::new(ModelConfig::load(dir)?));
        Split::with(nl, eos, cuts, devices, o, std::sync::Arc::new(move |st: Stage, o: &Opts| Dyn::new_stage(&dir, &m, d.clone(), None, st, o)))
    }

    /// A model of `nl` layers whose stage `make` builds on the GPU of the options it is given (a loader of the dynamic route's: `StageLoad`).
    pub fn with(nl: usize, eos: Vec<i64>, cuts: &[usize], devices: &[i32], o: &Opts, make: MakeStage) -> Result<Split> {
        if devices.len() != cuts.len() + 1 || cuts.windows(2).any(|w| w[0] >= w[1]) || cuts.iter().any(|&c| c == 0 || c >= nl) {
            return err(format!("cuts {cuts:?} over {nl} layers on GPUs {devices:?}: ascending cuts inside the layers, one GPU more than cuts"));
        }
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let mut w = Vec::new();
        for (k, &device) in devices.iter().enumerate() {
            let st = Stage { layers: (bounds[k], bounds[k + 1]), first: k == 0, last: k + 1 == devices.len() };
            let (tx, crx) = channel::<Cmd>();
            let (otx, rx) = channel::<Out>();
            let (make, mut o) = (make.clone(), o.clone());
            o.device = device;
            let h = std::thread::Builder::new()
                .name(format!("glyd-stage-{k}"))
                .spawn(move || {
                    // (the stage's engine lives and dies on this thread: its GPU context is this thread's)
                    let mut e = match make(st, &o) {
                        Ok(e) => e,
                        Err(x) => {
                            let _ = otx.send(Out::Done(Err(x)));
                            return;
                        }
                    };
                    let _ = otx.send(Out::Done(Ok(())));
                    while let Ok(c) = crx.recv() {
                        let out = match c {
                            Cmd::Reset => Out::Done(e.reset()),
                            Cmd::Rows { tokens, at, mut io, head, chunk } => Out::Rows(
                                if chunk { e.stage_chunk(&tokens, at, &mut io, head as usize) } else { e.stage_rows(&tokens, at, &mut io, head as usize) }
                                    .map(|_| io),
                            ),
                            Cmd::Verify { tokens, at, mut io } => Out::Rows(e.verify_rows(&tokens, at, &mut io, tokens.len()).map(|_| io)),
                            Cmd::Drafts { last, given } => Out::Tokens(e.stage_drafts(last, given.as_deref())),
                            Cmd::HeadOut { rows, words } => Out::Head(e.head_out(rows, words)),
                            Cmd::Rewind(keep) => Out::Done(e.rewind(keep)),
                            Cmd::HeadAgain { at, kept, next } => Out::Done(e.stage_head_again(at, kept, &next)),
                            Cmd::Drafting => Out::Flag(e.drafts_with_head()),
                            Cmd::Sample => Out::Sample(e.sample()),
                            Cmd::Logits => Out::Logits(e.logits()),
                            Cmd::Logprob(l) => Out::Logprob(e.logprob(l)),
                            Cmd::Row(a) => Out::Row(row_ask(&mut e, a)),
                            Cmd::Describe => Out::Text(e.describe()),
                            Cmd::Images(prompt, images) => Out::Done(e.set_images(&prompt, &images)),
                        };
                        if otx.send(out).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|x| Error(format!("a GPU stage's thread: {x}")))?;
            w.push(Worker { tx: Some(tx), rx, h: Some(h) });
        }
        // (the stages load at once, a thread each)
        for wk in &w {
            match wk.rx.recv() {
                Ok(Out::Done(r)) => r?,
                _ => return Err(gone()),
            }
        }
        let mut s = Split { w, n: 0, ctx: o.ctx.div_ceil(16) * 16, chunk: o.chunk.max(1), eos, drafting: false, next: Vec::new() };
        let last = s.w.len() - 1;
        s.drafting = matches!(s.ask(last, Cmd::Drafting)?, Out::Flag(true));
        Ok(s)
    }

    fn ask(&self, k: usize, c: Cmd) -> Result<Out> {
        self.w[k].tx.as_ref().ok_or_else(gone)?.send(c).map_err(|_| gone())?;
        self.w[k].rx.recv().map_err(|_| gone())
    }

    /// A command every stage runs in turn that returns nothing.
    fn each(&self, c: &dyn Fn() -> Cmd) -> Result<()> {
        for k in 0..self.w.len() {
            match self.ask(k, c())? {
                Out::Done(r) => r?,
                _ => return Err(gone()),
            }
        }
        Ok(())
    }

    /// The rows through every stage in turn, the first at `at`; the head on the last row where `head`; `chunk`: a prompt's chunk.
    fn rows(&mut self, tokens: &[u32], at: usize, head: bool, chunk: bool) -> Result<()> {
        let mut io = Vec::new();
        for k in 0..self.w.len() {
            match self.ask(k, Cmd::Rows { tokens: tokens.to_vec(), at, io, head, chunk })? {
                Out::Rows(r) => io = r?,
                _ => return Err(gone()),
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.each(&|| Cmd::Reset)?;
        self.n = 0;
        Ok(())
    }

    /// The sequence begins with `prompt`, whose images are `images` in order (none: a text): every stage told before its first chunk, the first running the
    /// tower as the chunks reach their rows (src/dynamic/images.rs). A model with no vision tower refuses images in words.
    pub fn set_images(&self, prompt: &[u32], images: &[std::sync::Arc<crate::vision::Image>]) -> Result<()> {
        self.each(&|| Cmd::Images(prompt.to_vec(), images.to_vec()))
    }

    /// Whether the last stage drafts with a head of the model (`spec_round`).
    pub fn drafting(&self) -> bool {
        self.drafting
    }

    /// One drafting round, as `Dyn::spec_round` runs it on one GPU: from `last` (the token after the cache's, not in it yet) the last stage's head drafts
    /// (or `given` stand for its drafts), every stage runs `last` and them as one verify pass (each row a step's), the drafts are kept up to the model's first
    /// disagreement and its own token there (greedy) or each by `sampler`, every stage goes back to the rows kept and the head runs again on them. The tokens
    /// returned (1 to k + 1) are what steps would give one by one; the cache then holds `last` and the kept drafts.
    pub fn spec_round_with(&mut self, last: u32, given: Option<&[u32]>, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let (n0, end) = (self.n, self.w.len() - 1);
        let drafts = match self.ask(end, Cmd::Drafts { last, given: given.map(<[u32]>::to_vec) })? {
            Out::Tokens(r) => r?,
            _ => return Err(gone()),
        };
        let seq: Vec<u32> = std::iter::once(last).chain(drafts).collect();
        let mut io = Vec::new();
        for k in 0..self.w.len() {
            match self.ask(k, Cmd::Verify { tokens: seq.clone(), at: n0, io })? {
                Out::Rows(r) => io = r?,
                _ => return Err(gone()),
            }
        }
        let (rows, _) = self.head_out(seq.len(), false)?;
        let big: Vec<u32> = rows.iter().map(|r| r.0).collect();
        // (each draft's check made of its row on the last stage's GPU, as one GPU's)
        let (out, kept) = super::accept(&seq, &big, &mut Last(&*self), sampler)?;
        self.each(&|| Cmd::Rewind(kept + 1))?;
        let next: Vec<u32> = seq[1..=kept].iter().copied().chain(out.last().copied()).collect();
        match self.ask(end, Cmd::HeadAgain { at: n0, kept, next })? {
            Out::Done(r) => r?,
            _ => return Err(gone()),
        }
        self.n = n0 + kept + 1;
        Ok(out)
    }

    /// `spec_round_with` of the head's own drafts.
    pub fn spec_round(&mut self, last: u32, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        self.spec_round_with(last, None, sampler)
    }

    /// The last pass's head on its last `rows` rows (the last stage's `Dyn::head_out`): each row's greedy token and logit, and where `words` the rows' logits.
    pub fn head_out(&self, rows: usize, words: bool) -> Result<HeadOut> {
        match self.ask(self.w.len() - 1, Cmd::HeadOut { rows, words })? {
            Out::Head(r) => r,
            _ => Err(gone()),
        }
    }

    /// The prompt after the tokens already in the cache, in chunks; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.ctx));
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            self.rows(&tokens[at..at + n], self.n, at + n == tokens.len(), true)?;
            self.n += n;
            at += n;
        }
        self.sample()
    }

    /// One token through the model; the token after it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err("the cache is full");
        }
        self.rows(&[token], self.n, true, false)?;
        self.n += 1;
        self.sample()
    }

    fn sample(&self) -> Result<Sample> {
        match self.ask(self.w.len() - 1, Cmd::Sample)? {
            Out::Sample(r) => r,
            _ => Err(gone()),
        }
    }

    /// The last pass's logits (the last stage's).
    pub fn logits(&self) -> Result<Vec<f32>> {
        match self.ask(self.w.len() - 1, Cmd::Logits)? {
            Out::Logits(r) => r,
            _ => Err(gone()),
        }
    }

    /// A question of the last pass's head rows, put to the last stage (`row_ask`); None where its thread is gone.
    fn ask_row(&self, a: RowAsk) -> Option<RowGot> {
        match self.ask(self.w.len() - 1, Cmd::Row(a)) {
            Ok(Out::Row(g)) => Some(g),
            _ => None,
        }
    }

    /// The log-probability of the last pass's `logit` (the last stage's, on its GPU).
    pub fn logprob(&self, logit: f32) -> Result<f32> {
        match self.ask(self.w.len() - 1, Cmd::Logprob(logit))? {
            Out::Logprob(r) => r,
            _ => Err(gone()),
        }
    }

    pub fn describe(&self) -> String {
        (0..self.w.len())
            .map(|k| match self.ask(k, Cmd::Describe) {
                Ok(Out::Text(t)) => format!("stage {k}: {t}"),
                _ => format!("stage {k}: gone"),
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub fn capacity(&self) -> usize {
        self.ctx
    }

    pub fn eos(&self) -> &[i64] {
        &self.eos
    }
}

impl Drop for Split {
    fn drop(&mut self) {
        // (closing a stage's channel ends its loop; its engine is freed on its own thread)
        for wk in &mut self.w {
            wk.tx = None;
        }
        for wk in &mut self.w {
            if let Some(h) = wk.h.take() {
                let _ = h.join();
            }
        }
    }
}

/// A model on one GPU, or split across several: what a check or a caller runs either way.
pub enum Run {
    One(Dyn),
    Split(Split),
    /// Every layer across the GPUs (src/dynamic/tp.rs).
    Tp(super::tp::Tp),
}

impl Run {
    /// One GPU where `cuts` is empty, else `Split::with`'s stages (`devices`: one more than cuts; a device may repeat), each made by `make`.
    pub fn new(nl: usize, eos: Vec<i64>, cuts: &[usize], devices: &[i32], o: &Opts, make: MakeStage) -> Result<Run> {
        if cuts.is_empty() {
            let mut o = o.clone();
            o.device = devices.first().copied().unwrap_or(o.device);
            return Ok(Run::One(make(Stage::whole(nl), &o)?));
        }
        Ok(Run::Split(Split::with(nl, eos, cuts, devices, o, make)?))
    }
    pub fn describe(&self) -> String {
        match self {
            Run::One(e) => e.describe(),
            Run::Split(e) => e.describe(),
            Run::Tp(e) => e.describe(),
        }
    }
    pub fn prefill(&mut self, t: &[u32]) -> Result<Sample> {
        match self {
            Run::One(e) => e.prefill(t),
            Run::Split(e) => e.prefill(t),
            Run::Tp(e) => e.prefill(t),
        }
    }
    pub fn step(&mut self, t: u32) -> Result<Sample> {
        match self {
            Run::One(e) => e.step(t),
            Run::Split(e) => e.step(t),
            Run::Tp(e) => e.step(t),
        }
    }
    pub fn logits(&self) -> Result<Vec<f32>> {
        match self {
            Run::One(e) => e.logits(),
            Run::Split(e) => e.logits(),
            Run::Tp(e) => e.logits(),
        }
    }
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Run::One(e) => e.reset(),
            Run::Split(e) => e.reset(),
            Run::Tp(e) => e.reset(),
        }
    }
    /// The next prompt's images, after `reset` (`Dyn::set_images`; a split's every stage, `Split::set_images`); every layer across GPUs reads none yet.
    pub fn set_images(&mut self, prompt: &[u32], images: &[std::sync::Arc<crate::vision::Image>]) -> Result<()> {
        match self {
            Run::One(e) => e.set_images(prompt, images),
            Run::Split(e) => e.set_images(prompt, images),
            Run::Tp(_) if images.is_empty() => Ok(()),
            Run::Tp(_) => err("a model with every layer across GPUs reads no images yet"),
        }
    }
}

/// `--tp GPU,..` of a command line: every layer across those GPUs (a GPU may repeat: places on one GPU check the gathers), or none.
pub fn tp_args(a: &crate::args::Args) -> Result<Option<Vec<i32>>> {
    a.flag("tp").map(|v| v.split(',').map(|x| x.trim().parse::<i32>().map_err(|_| Error(format!("--tp {v}: GPU numbers")))).collect()).transpose()
}

/// `--split CUT,..` and `--devices GPU,..` of a command line: the cuts (none: one GPU) and the devices (one more than the cuts; GPU 0 for each where not given).
pub fn split_args(a: &crate::args::Args) -> Result<(Vec<usize>, Vec<i32>)> {
    let num = |v: &str, what: &str| -> Result<Vec<i64>> {
        v.split(',').map(|x| x.trim().parse::<i64>().map_err(|_| Error(format!("--{what} {v}: numbers")))).collect()
    };
    let cuts: Vec<usize> = match a.flag("split") {
        Some(v) => num(v, "split")?.into_iter().map(|x| x as usize).collect(),
        None => Vec::new(),
    };
    let devices: Vec<i32> = match a.flag("devices") {
        Some(v) => num(v, "devices")?.into_iter().map(|x| x as i32).collect(),
        None => vec![0; cuts.len() + 1],
    };
    Ok((cuts, devices))
}

impl Split {
    /// A reply as `Dyn::reply` gives one (drafted a round at a time where the last stage drafts); a sampled token from the last stage's logits row on the host.
    pub(super) fn reply(&mut self, prompt: &[u32], o: &crate::sched::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::sched::Finish)> {
        use crate::sched::Finish;
        if prompt.is_empty() || prompt.len() + 1 > self.ctx {
            return err(format!("{} prompt tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        // (the prompt's images, where `see` gave them for this reply: none for a reply it did not)
        let images = std::mem::take(&mut self.next);
        self.set_images(prompt, &images)?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        // (a drawn token: from what the last stage's GPU makes of the pass's head row, the request's changes put on it first, as one GPU's: `sched::draw_row`)
        let draw = |e: &Split, sampler: &mut crate::sampler::Sampler| crate::sched::draw_row(&mut Last(e), 0, sampler).map(|x| x.0);
        self.ask_row(RowAsk::Procs(vec![sampler.proc()]));
        let mut s = self.prefill(prompt)?;
        let mut out = Vec::new();
        if self.drafting {
            // drafted: the model's tokens a round at a time (the last of a round is not in the cache yet: the next round starts from it)
            let first = if greedy { s.token } else { draw(self, &mut sampler)? };
            let mut pending = std::collections::VecDeque::from([first]);
            while let Some(tok) = pending.pop_front() {
                if !o.ignore_eos && (self.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
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
            let tok = if greedy { s.token } else { draw(self, &mut sampler)? };
            if !o.ignore_eos && (self.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                return Ok((out, Finish::Stop));
            }
            out.push(tok);
            if !each(tok) {
                return Ok((out, Finish::Stopped));
            }
            if out.len() >= o.max_new || self.n + 1 > self.ctx {
                return Ok((out, Finish::Length));
            }
            self.ask_row(RowAsk::Procs(vec![sampler.proc()]));
            s = self.step(tok)?;
        }
    }
}

/// A question of a pass's head rows that a stage's engine answers on its GPU (a split's last stage, a group's first GPU: `row_ask`), each as one GPU's scheduler step asks it
/// (`ModelStep`'s methods of these names): its candidates (the row the draw sees, the model's), the model's sum of exponentials, whether the candidates are of the row the draw
/// sees, the request's changes for the next pass's rows (`Procs`) or made again on a row (`Remake`), a draw's total, bins and tokens of a query, a token's logit, the row, the
/// logits of some tokens.
pub(super) enum RowAsk {
    Top(usize),
    TopRaw(usize),
    Mass(usize, f32),
    Processed(usize),
    Procs(Vec<Option<crate::sampler::Proc>>),
    Remake(usize, Option<crate::sampler::Proc>),
    Total(usize, f32, f32),
    Bins(usize, crate::sampler::Query),
    Tokens(usize, crate::sampler::Query, usize),
    Logit(usize, u32),
    Logits(usize),
    Gather(usize, Vec<u32>),
}

pub(super) enum RowGot {
    Cands(Option<Vec<(u32, f32)>>),
    Sum(Option<f64>),
    Flag(bool),
    Total(Option<u64>),
    Bins(Option<crate::sampler::Bins>),
    Logit(Option<f32>),
    Row(Result<Vec<f32>>),
    Gathered(Result<Vec<(u32, f32)>>),
}

/// `a` answered by a stage's engine (its thread's), as its scheduler's step answers it.
pub(super) fn row_ask(e: &mut Dyn, a: RowAsk) -> RowGot {
    use crate::sched::ModelStep as M;
    match a {
        RowAsk::Top(r) => RowGot::Cands(M::top(e, r)),
        RowAsk::TopRaw(r) => RowGot::Cands(M::top_raw(e, r)),
        RowAsk::Mass(r, t) => RowGot::Sum(M::mass(e, r, t)),
        RowAsk::Processed(r) => RowGot::Flag(M::processed(e, r)),
        RowAsk::Procs(p) => {
            M::procs(e, p);
            RowGot::Flag(true)
        }
        RowAsk::Remake(r, p) => RowGot::Flag(M::remake(e, r, p)),
        RowAsk::Total(r, m, t) => RowGot::Total(M::total(e, r, m, t)),
        RowAsk::Bins(r, q) => RowGot::Bins(M::bins(e, r, &q)),
        RowAsk::Tokens(r, q, cap) => RowGot::Cands(M::tokens(e, r, &q, cap)),
        RowAsk::Logit(r, t) => RowGot::Logit(M::logit(e, r, t)),
        RowAsk::Logits(r) => RowGot::Row(M::logits(e, r)),
        RowAsk::Gather(r, t) => RowGot::Gathered(M::gather(e, r, &t)),
    }
}

/// What answers a question of a pass's head rows on another thread's engine (a split's last stage, a group's first GPU).
pub(super) trait AsksRows {
    fn ask_rows(&self, a: RowAsk) -> Option<RowGot>;
}

impl AsksRows for Split {
    fn ask_rows(&self, a: RowAsk) -> Option<RowGot> {
        self.ask_row(a)
    }
}

/// The head rows of a split's last stage or a group's first GPU as the rows a draw reads (`sched::Rows`), each question answered on that engine's GPU.
pub(super) struct Last<'a, T: AsksRows>(pub(super) &'a T);

impl<T: AsksRows> Last<'_, T> {
    fn cands(&self, a: RowAsk) -> Option<Vec<(u32, f32)>> {
        match self.0.ask_rows(a) {
            Some(RowGot::Cands(c)) => c,
            _ => None,
        }
    }
    fn flag(&self, a: RowAsk) -> bool {
        matches!(self.0.ask_rows(a), Some(RowGot::Flag(true)))
    }
    pub(super) fn top_raw(&self, row: usize) -> Option<Vec<(u32, f32)>> {
        self.cands(RowAsk::TopRaw(row))
    }
    pub(super) fn mass(&self, row: usize, temperature: f32) -> Option<f64> {
        match self.0.ask_rows(RowAsk::Mass(row, temperature)) {
            Some(RowGot::Sum(m)) => m,
            _ => None,
        }
    }
    pub(super) fn procs(&self, procs: Vec<Option<crate::sampler::Proc>>) {
        self.0.ask_rows(RowAsk::Procs(procs));
    }
    pub(super) fn gather(&self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
        match self.0.ask_rows(RowAsk::Gather(row, tokens.to_vec())) {
            Some(RowGot::Gathered(r)) => r,
            _ => Err(gone()),
        }
    }
}

impl<T: AsksRows> crate::sched::Rows for Last<'_, T> {
    fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        self.cands(RowAsk::Top(row))
    }
    fn processed(&mut self, row: usize) -> bool {
        self.flag(RowAsk::Processed(row))
    }
    fn remake(&mut self, row: usize, proc: Option<crate::sampler::Proc>) -> bool {
        self.flag(RowAsk::Remake(row, proc))
    }
    fn total(&mut self, row: usize, max: f32, temperature: f32) -> Option<u64> {
        match self.0.ask_rows(RowAsk::Total(row, max, temperature)) {
            Some(RowGot::Total(t)) => t,
            _ => None,
        }
    }
    fn bins(&mut self, row: usize, q: &crate::sampler::Query) -> Option<crate::sampler::Bins> {
        match self.0.ask_rows(RowAsk::Bins(row, *q)) {
            Some(RowGot::Bins(b)) => b,
            _ => None,
        }
    }
    fn tokens(&mut self, row: usize, q: &crate::sampler::Query, cap: usize) -> Option<Vec<(u32, f32)>> {
        self.cands(RowAsk::Tokens(row, *q, cap))
    }
    fn logit(&mut self, row: usize, t: u32) -> Option<f32> {
        match self.0.ask_rows(RowAsk::Logit(row, t)) {
            Some(RowGot::Logit(x)) => x,
            _ => None,
        }
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        match self.0.ask_rows(RowAsk::Logits(row)) {
            Some(RowGot::Row(r)) => r,
            _ => Err(gone()),
        }
    }
}

impl crate::conv::Gen for Run {
    fn capacity(&self) -> usize {
        match self {
            Run::One(e) => crate::conv::Gen::capacity(e),
            Run::Split(e) => e.capacity(),
            Run::Tp(e) => e.capacity(),
        }
    }
    fn reply(&mut self, prompt: &[u32], o: &crate::sched::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::sched::Finish)> {
        match self {
            Run::One(e) => crate::conv::Gen::reply(e, prompt, o, each),
            Run::Split(e) => e.reply(prompt, o, each),
            Run::Tp(e) => e.reply(prompt, o, each),
        }
    }
    fn held(&self) -> Option<crate::dev::Held> {
        match self {
            Run::One(e) => crate::conv::Gen::held(e),
            Run::Split(_) | Run::Tp(_) => None,
        }
    }
    /// One GPU: the engine's pools are the plan's (`Dyn::take_plan`); stages and layers across GPUs hold one sequence, its rings made at the load.
    fn take_plan(&mut self, users: usize, kv: u64) {
        if let Run::One(e) = self {
            e.take_plan(users, kv);
        }
    }
    /// The engine's own images (src/dynamic/images.rs), on one GPU or cut into stages (the next reply's: `Split::reply` tells every stage); every layer across
    /// GPUs reads none yet.
    fn see(&mut self, images: Vec<std::sync::Arc<crate::vision::Image>>) -> Result<()> {
        match self {
            Run::One(e) => crate::conv::Gen::see(e, images),
            Run::Split(e) => {
                e.next = images;
                Ok(())
            }
            Run::Tp(_) if images.is_empty() => Ok(()),
            Run::Tp(_) => err("a model with every layer across GPUs reads no images yet"),
        }
    }
    /// One GPU: the engine's own step (src/dynamic/rows.rs); stages and layers across GPUs: a step of one sequence at a time (`ModelStep for Run`), its slot empty (what a
    /// reply before this left in the cache is not a scheduler's sequence, which begins at its first token: `serve_rows` empties the executor's the same way).
    fn as_step(&mut self, users: usize) -> Result<Option<(usize, &mut dyn crate::sched::ModelStep)>> {
        match self {
            Run::One(e) => crate::conv::Gen::as_step(e, users),
            Run::Split(_) | Run::Tp(_) => {
                self.reset()?;
                Ok(Some((1, self as &mut dyn crate::sched::ModelStep)))
            }
        }
    }
}

impl Run {
    /// `f` of the rows a draw reads (a split's last stage, a group's first GPU); None on one GPU (its own step is the scheduler's).
    fn rows_of<R>(&mut self, f: impl FnOnce(&mut dyn crate::sched::Rows) -> R) -> Option<R> {
        match self {
            Run::Split(e) => Some(f(&mut Last(&*e))),
            Run::Tp(e) => Some(f(&mut Last(&*e))),
            Run::One(_) => None,
        }
    }

    /// Tokens in the cache.
    fn held(&self) -> usize {
        match self {
            Run::One(e) => e.n,
            Run::Split(e) => e.n,
            Run::Tp(e) => e.held(),
        }
    }
}

/// Stages or layers across GPUs as a scheduler's step of one sequence at a time (R4a #138: every model is served by `Sched`, none by a loop of its own): one slot, emptied
/// at the sequence's end; a prompt in one call (the stages run it a chunk at a time), a token a decode step, a draw read from the last stage's logits row. Across
/// GPUs a step holds one sequence: `stage_rows` and the gathers take one.
impl crate::sched::ModelStep for Run {
    fn window(&self) -> usize {
        crate::conv::Gen::capacity(self)
    }
    fn chunk(&self) -> usize {
        crate::conv::Gen::capacity(self)
    }
    fn pages(&self) -> usize {
        0
    }
    fn slots(&self) -> Option<usize> {
        Some(1)
    }
    /// One GPU: its engine's context (it runs on this thread); stages and every layer across GPUs run on threads of their own, each its GPU's.
    fn make_current(&mut self) -> Result<()> {
        match self {
            Run::One(e) => e.dev.make_current(),
            _ => Ok(()),
        }
    }
    fn eos(&self) -> Vec<i64> {
        match self {
            Run::One(e) => e.desc().eos.clone(),
            Run::Split(e) => e.eos().to_vec(),
            Run::Tp(e) => e.eos().to_vec(),
        }
    }
    /// The sequence's images: every stage of a split told (`Split::set_images`); every layer across GPUs refuses them.
    fn images(&mut self, slot: i32, prompt: &[u32], images: &[std::sync::Arc<crate::vision::Image>]) -> Result<()> {
        match self {
            Run::One(e) => crate::sched::ModelStep::images(e, slot, prompt, images),
            Run::Split(e) => e.set_images(prompt, images),
            Run::Tp(_) if images.is_empty() => Ok(()),
            Run::Tp(_) => err("a model with every layer across GPUs reads no images yet"),
        }
    }
    fn prefill(&mut self, _pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        if start != self.held() {
            return err(format!("a chunk at {start} of a sequence that holds {}", self.held()));
        }
        let s = Run::prefill(self, tokens)?;
        Ok(last.then_some(s))
    }
    fn decode(&mut self, rows: &[crate::sched::Row]) -> Result<Vec<Sample>> {
        match rows {
            [r] if r.pos == self.held() => Ok(vec![Run::step(self, r.token)?]),
            [r] => err(format!("a token at {} of a sequence that holds {}", r.pos, self.held())),
            _ => err(format!("a step of {} rows on a model that runs one sequence at a time", rows.len())),
        }
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        match self.rows_of(|r| crate::sched::Rows::logits(r, row)) {
            Some(r) => r,
            None => Run::logits(self),
        }
    }
    /// Stages, and every layer across GPUs: the last stage's (the group's first GPU's) candidates, sums, totals, bins and tokens of its head row, made on its GPU by the
    /// kernels and launches of one GPU's step, so a draw and the log-probabilities are one GPU's bits (the row read here would be drawn and summed on the host).
    fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        self.rows_of(|r| crate::sched::Rows::top(r, row)).flatten()
    }
    fn top_raw(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        match self {
            Run::Split(e) => Last(&*e).top_raw(row),
            Run::Tp(e) => Last(&*e).top_raw(row),
            Run::One(_) => None,
        }
    }
    fn mass(&mut self, row: usize, temperature: f32) -> Option<f64> {
        match self {
            Run::Split(e) => Last(&*e).mass(row, temperature),
            Run::Tp(e) => Last(&*e).mass(row, temperature),
            Run::One(_) => None,
        }
    }
    fn procs(&mut self, procs: Vec<Option<crate::sampler::Proc>>) {
        match self {
            Run::Split(e) => Last(&*e).procs(procs),
            Run::Tp(e) => Last(&*e).procs(procs),
            Run::One(_) => {}
        }
    }
    fn processed(&mut self, row: usize) -> bool {
        self.rows_of(|r| crate::sched::Rows::processed(r, row)).unwrap_or(false)
    }
    fn remake(&mut self, row: usize, proc: Option<crate::sampler::Proc>) -> bool {
        self.rows_of(|r| crate::sched::Rows::remake(r, row, proc)).unwrap_or(false)
    }
    fn total(&mut self, row: usize, max: f32, temperature: f32) -> Option<u64> {
        self.rows_of(|r| crate::sched::Rows::total(r, row, max, temperature)).flatten()
    }
    fn bins(&mut self, row: usize, q: &crate::sampler::Query) -> Option<crate::sampler::Bins> {
        self.rows_of(|r| crate::sched::Rows::bins(r, row, q)).flatten()
    }
    fn tokens(&mut self, row: usize, q: &crate::sampler::Query, cap: usize) -> Option<Vec<(u32, f32)>> {
        self.rows_of(|r| crate::sched::Rows::tokens(r, row, q, cap)).flatten()
    }
    fn logit(&mut self, row: usize, t: u32) -> Option<f32> {
        self.rows_of(|r| crate::sched::Rows::logit(r, row, t)).flatten()
    }
    fn gather(&mut self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
        match self {
            Run::Split(e) => Last(&*e).gather(row, tokens),
            Run::Tp(e) => Last(&*e).gather(row, tokens),
            Run::One(_) => err("a gather of a model on one GPU through stages' steps"),
        }
    }
    fn release(&mut self, _pages: &[i32]) -> Result<()> {
        self.reset()
    }
    /// Stages whose last drafts with a head of the model: the sequence's decode a drafting round (`Split::spec_round`), as a reply alone runs it.
    fn draft(&mut self, row: &crate::sched::Row, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Option<Vec<u32>>> {
        match self {
            Run::Split(e) if e.drafting() => {
                if row.pos != e.n {
                    return err(format!("a token at {} of a sequence that holds {}", row.pos, e.n));
                }
                e.spec_round(row.token, sampler).map(Some)
            }
            _ => Ok(None),
        }
    }
}

/// A greedy reply of `tokens` tokens to `prompt` from the model alone (`Gen::reply`), and the same through the scheduler with the model as its step (`as_step`: across GPUs, a
/// step of one slot): both, for a check that they are the same (`tpcheck`).
#[cfg(feature = "dev")]
pub fn sched_reply(e: &mut Run, prompt: &[u32], tokens: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    use crate::conv::Gen;
    let reply = crate::sched::Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let (alone, _) = e.reply(prompt, &reply, &mut |_| true)?;
    Ok((alone, served(e, prompt, tokens, None)?.0))
}

/// A greedy reply of `tokens` tokens to `prompt` through the scheduler with the model as its step (`as_step`: across GPUs, a step of one slot), and where `k` is
/// given each token's log-probability and its `k` likeliest tokens' as a server sends them (`splitcheck`: a split's are one GPU's bits).
#[cfg(feature = "dev")]
pub fn served(e: &mut Run, prompt: &[u32], tokens: usize, k: Option<usize>) -> Result<(Vec<u32>, Vec<crate::sampler::Lp>)> {
    use crate::conv::Gen;
    use crate::sched::{Ev, Reply, Sched, Submit};
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: k,
        stop_ids: Vec::new(),
    };
    let Some((n, step)) = e.as_step(1)? else { return err("no scheduler's step") };
    let mut s = Sched::new(step, n);
    let (tx, rx) = channel();
    s.submit(Submit {
        prompt: prompt.to_vec(),
        reply,
        out: tx,
        abort: Default::default(),
        resumed: Default::default(),
        cached: Default::default(),
        created: Default::default(),
        salt: 0,
        pin: 0,
        images: Vec::new(),
    });
    while !s.idle() {
        s.tick()?;
    }
    let (mut via, mut lps) = (Vec::new(), Vec::new());
    loop {
        match rx.recv().map_err(|_| Error("the scheduler dropped the sequence".into()))? {
            Ev::Token(t) => via.push(t),
            Ev::Lp(l) => lps.push(l),
            Ev::End(r) => {
                r?;
                return Ok((via, lps));
            }
        }
    }
}

/// The name of the support row of the model whose config.json is `text`, where the row's layers split over GPUs on the dynamic route (`Support::split_levels`)
/// or the row is sized as the dynamic route's on one GPU (`Sizing::Dynamic`): the families this module opens from their release's directory; None for another
/// path's model.
pub fn family(text: &str) -> Option<&'static str> {
    dynamic_row(crate::runtime::config::family_of(text)?)
}

/// `family` of a config.json parsed once.
pub fn family_in(c: &crate::runtime::config::ModelConfig) -> Option<&'static str> {
    dynamic_row(c.family()?)
}

fn dynamic_row(f: crate::support::Family) -> Option<&'static str> {
    let row = crate::support::row(f);
    (!row.split_levels.is_empty() || row.route.sizing == crate::support::Sizing::Dynamic).then_some(row.name)
}

/// The model of `dir` as `plan` says it loads (src/plan.rs `size_dynamic`, a lossy level's stages `sizing::lossy_stages`): its layers cut over the plan's GPUs
/// (`Spread::cuts`, `devices`) or across them (`across`), its context, prompt chunk and layout (the plan's: penguin's 12-bit one where `twelve`, a lossy level's
/// block layout), and where its experts are in the computer's memory each stage's places of them, the memory the plan gives them and the last layers' dense
/// matrices there (`dense`), with the expert cache's copy-or-compute policy `offload`; on one GPU its sequences' keys, values and states in the plan's bytes
/// (`Dyn::take_plan`). Nothing is sized here. `m`: the directory's config.json, parsed once (the plan's): every stage's loader reads the model from it.
pub fn open_plan(dir: &Path, m: &ModelConfig, plan: &crate::plan::LoadPlan, offload: crate::offload::Offload) -> Result<Run> {
    let sp = plan.spread.as_ref().ok_or_else(|| Error("a plan of the dynamic route spreads the model (Spread)".into()))?;
    let o = plan_opts(plan)?;
    let ram = plan.experts.filter(|c| c.ram < c.host_bytes()).map(|c| crate::offload::HostMemory { bytes: c.ram, total: c.host_bytes() });
    let (cuts, devices) = (&sp.cuts, &sp.devices);
    let mut run = if sp.host_bytes == 0 {
        if sp.across {
            open_tp_of(dir, m, &o, devices)?
        } else {
            open_of(dir, m, &o, cuts, devices)?
        }
    } else {
        // (each stage's experts in the expert cache: a hummingbird directory's, or the release's in places of their own; the release's experts of the latent block
        // across the GPUs where the plan says so, their cache's host tier and plans shared: src/dynamic/tp_cache.rs)
        let cachings: Vec<crate::hbg_store::HbCaching> =
            sp.places.iter().map(|&places| crate::hbg_store::HbCaching { places, ram, offload, dense: sp.dense }).collect();
        if sp.across && plan.level != "hummingbird" && latent_experts(dir, m)? {
            let c = cachings.into_iter().next().ok_or_else(|| Error("a plan across GPUs with no places for its experts".into()))?;
            open_tp_cached_of(dir, m, &o, devices, c)?
        } else {
            open_cached_of(dir, m, &o, cuts, devices, cachings)?
        }
    };
    // (its sequences' keys, values and states the plan's: on one GPU their pools take its bytes, no more)
    crate::conv::Gen::take_plan(&mut run, plan.users, plan.kv);
    Ok(run)
}

/// The options a plan that spreads the model (`Spread`) loads it with: its context, prompt chunk and layout, on the GPU of its first stage (the plan's GPU where
/// it is whole; each stage on its own, `Run::new`). The GPU is the plan's whichever it is: nothing here or after takes GPU 0 for it.
pub fn plan_opts(plan: &crate::plan::LoadPlan) -> Result<Opts> {
    let device =
        plan.spread.as_ref().and_then(|sp| sp.devices.first().copied()).ok_or_else(|| Error("a plan of the dynamic route names its GPUs (Spread)".into()))?;
    Ok(Opts { ctx: plan.context, chunk: plan.chunk, graph: true, device, layout: Some(plan.layout) })
}

/// Whether the model of `dir` has experts of the Moe kind's latent block (`experts::latent_block`: a latent of their own projections whose sum is normed),
/// whose expert cache runs across a group of GPUs (src/dynamic/tp_cache.rs), by its description (`m`: its config.json, parsed once).
fn latent_experts(dir: &Path, m: &ModelConfig) -> Result<bool> {
    Ok(latent_experts_of(&crate::runtime::describe::describe_dir_of(dir, m)?.0))
}

/// `latent_experts` of a description.
fn latent_experts_of(d: &crate::runtime::describe::Described) -> bool {
    use crate::runtime::kinds::{Body, Ffn};
    d.desc
        .blocks
        .iter()
        .any(|b| matches!(&b.body, Body::Ffn(Ffn::Moe(m)) if m.latent.as_ref().is_some_and(|l| l.norm_out.is_some() && l.expert_norm.is_none())))
}

/// The model of `dir` on the dynamic engine, by its description's loader (`loader`): on one GPU where `cuts` is empty, else in stages (`devices`: one more than
/// `cuts`). The experts at `o.layout`'s level where the loader takes a level (penguin's lossless where none is given), else as the release has them, exact.
pub fn open(dir: &Path, o: &Opts, cuts: &[usize], devices: &[i32]) -> Result<Run> {
    open_of(dir, &ModelConfig::load(dir)?, o, cuts, devices)
}

/// `open` of a config.json parsed once (`m`: `dir`'s).
pub fn open_of(dir: &Path, m: &ModelConfig, o: &Opts, cuts: &[usize], devices: &[i32]) -> Result<Run> {
    let (nl, eos, make) = maker(dir, m)?;
    Run::new(nl, eos, cuts, devices, o, make)
}

/// The model of `dir` with every layer across `devices` (src/dynamic/tp.rs).
pub fn open_tp(dir: &Path, o: &Opts, devices: &[i32]) -> Result<Run> {
    open_tp_of(dir, &ModelConfig::load(dir)?, o, devices)
}

/// `open_tp` of a config.json parsed once.
pub fn open_tp_of(dir: &Path, m: &ModelConfig, o: &Opts, devices: &[i32]) -> Result<Run> {
    let (nl, eos, make) = maker(dir, m)?;
    Ok(Run::Tp(super::tp::Tp::with(nl, eos, devices, o, make)?))
}

/// The layers, end tokens and loader of a stage of the model of `dir` (its description's loader's, every expert as the stage's level holds it).
fn maker(dir: &Path, m: &ModelConfig) -> Result<(usize, Vec<i64>, MakeStage)> {
    let (nl, eos, load) = cached_stage(dir, m)?;
    Ok((nl, eos, std::sync::Arc::new(move |st, o: &Opts| load(st, o, None))))
}

/// What loads a stage of a model on the dynamic route, the same for every description: the stage's layers on the options' GPU, the experts at the options'
/// level where the loader takes one (penguin's lossless where none is given), in the expert cache as the caching says where one is given. A loader whose
/// experts the expert cache does not take refuses a caching, in words.
pub type StageLoad = fn(&Path, &ModelConfig, &Opts, Stage, Option<&crate::hbg_store::HbCaching>) -> Result<Dyn>;

/// A loader of the dynamic route: the model's layers and end tokens, from its directory and its config.json parsed once (`ends`), and a stage of it (`stage`).
#[derive(Clone, Copy)]
pub struct Loader {
    pub ends: fn(&Path, &ModelConfig) -> Result<(usize, Vec<i64>)>,
    pub stage: StageLoad,
}

/// The layer kinds a loader of the dynamic route lowers, as a description tells them apart (`lowering`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lowering {
    /// A compressed attention (src/dynamic/lower_comp.rs; DeepSeek-V4's).
    Compressed,
    /// Mamba-2, attention without a positional encoding and experts, a block a layer, plain residuals (`lower::nemotron`; Nemotron-H's).
    Mamba2,
    /// The Kimi delta attention and the latent attention with residual streams (`lower::glm`; GLM-5.3-Flash's).
    DeltaStreams,
    /// The Kimi delta attention and the latent attention with the blocks' depth mix (`lower::kimi`; Kimi-K3's).
    DeltaDepth,
    /// Gated residuals (`lower::muse`; Muse Spark's).
    GatedResidual,
    /// The gated residuals' streams (`lower::qwen4x`; Qwen3.8-Flash-Next's).
    GatedStreams,
    /// The gated delta rule with plain residuals, the feed-forwards dense or experts (`lower::qwen35`; the Qwen3.5 layout's).
    GatedDelta,
    /// The latent attention with plain residuals, the feed-forwards dense or experts (`lower::latent_moe`; GLM-5.3's).
    LatentPlain,
}

/// The lowering of the description `d`, by the layer kinds it holds, in this order: a compressed attention; Mamba-2 with plain residuals; the Kimi delta attention
/// with residual streams, or with the blocks' depth mix; gated residuals; the gated residuals' streams; the gated delta rule with plain residuals; the latent
/// attention with plain residuals. A description of other kinds is refused in words: no loader of the dynamic route lowers them.
fn lowering(d: &crate::runtime::describe::Described) -> Result<Lowering> {
    use crate::runtime::kinds::{Body, LinearKind, Mixer, Residual};
    let blocks = &d.desc.blocks;
    let mixer = |f: &dyn Fn(&Mixer) -> bool| blocks.iter().any(|b| matches!(&b.body, Body::Mixer(m) if f(m)));
    let residual = |f: &dyn Fn(&Residual) -> bool| blocks.iter().any(|b| f(&b.residual));
    let linear = |k: LinearKind| mixer(&|m| matches!(m, Mixer::Linear(l) if l.kind == k));
    let plain = blocks.iter().all(|b| matches!(b.residual, Residual::Add));
    Ok(if mixer(&|m| matches!(m, Mixer::Compressed(_))) {
        Lowering::Compressed
    } else if mixer(&|m| matches!(m, Mixer::Mamba2(_))) && plain {
        Lowering::Mamba2
    } else if linear(LinearKind::KimiDelta) && residual(&|r| matches!(r, Residual::Streams(_))) {
        Lowering::DeltaStreams
    } else if linear(LinearKind::KimiDelta) && residual(&|r| matches!(r, Residual::DepthMix(_))) {
        Lowering::DeltaDepth
    } else if residual(&|r| matches!(r, Residual::Gated(_))) {
        Lowering::GatedResidual
    } else if residual(&|r| matches!(r, Residual::GatedStreams(_))) {
        Lowering::GatedStreams
    } else if linear(LinearKind::GatedDeltaNet) && plain {
        Lowering::GatedDelta
    } else if mixer(&|m| matches!(m, Mixer::Mla(_))) && plain {
        Lowering::LatentPlain
    } else {
        return err(
            "no loader of the dynamic route lowers its layer kinds (a compressed attention; Mamba-2 with plain residuals; the Kimi delta attention with residual \
             streams or the blocks' depth mix; gated residuals; the gated residuals' streams; the gated delta rule with plain residuals; the latent attention \
             with plain residuals)",
        );
    })
}

/// The loader of the model whose description is `d`: the one that lowers the layer kinds it holds (`lowering`), its config's layers and end tokens as that
/// lowering's reader takes them.
pub fn loader(d: &crate::runtime::describe::Described) -> Result<Loader> {
    Ok(match lowering(d)? {
        // (its layers and end tokens by the description: DeepSeek-V4's flat config and V4.1's nested one alike)
        Lowering::Compressed => Loader { ends: |dir, m| super::deepseek_v4::ends_of(dir, m).map(|(l, e, _)| (l, e)), stage: super::deepseek_v4::load_stage_hb },
        Lowering::Mamba2 => {
            Loader { ends: |dir, m| m.read_with(dir, crate::nemotron::ConfigNemotronH::of).map(|c| (c.layers, c.eos)), stage: super::nemotron::stage }
        }
        Lowering::DeltaStreams => Loader { ends: |dir, m| m.read_with(dir, crate::glm::ConfigGlm::of).map(|c| (c.layers, c.eos)), stage: super::glm::stage },
        Lowering::DeltaDepth => Loader { ends: |dir, m| m.read_with(dir, crate::kimi::ConfigKimi::of).map(|c| (c.layers, c.eos)), stage: super::kimi::stage },
        Lowering::GatedResidual => {
            Loader { ends: |dir, m| m.read_with(dir, crate::muse::ConfigMuse::of).map(|c| (c.layers, c.eos)), stage: super::muse::load_stage_hb }
        }
        Lowering::GatedStreams => {
            Loader { ends: |dir, m| m.read_with(dir, crate::qwen4x::ConfigQx::of).map(|c| (c.layers, c.eos)), stage: super::qwen4x::stage }
        }
        Lowering::GatedDelta => Loader { ends: super::qwen35::ends, stage: super::qwen35::stage },
        Lowering::LatentPlain => Loader { ends: super::glm::plain_ends, stage: super::glm::stage_plain },
    })
}

/// The description of the model whose config is `c` lowered as its loader lowers it (`loader`), the whole model for sequences of up to `max_pos` positions, from the
/// config alone: the rings and states its sequences take on the executor, which a plan counts before a file is here (src/plan.rs `sizing::size_dynamic_row`).
/// The end tokens and the checkpoint's dtypes are not read (none of them changes a ring or a state).
pub fn lowered_of(c: &crate::runtime::config::ModelConfig, max_pos: usize) -> Result<ModelDesc> {
    lowered_described(&crate::runtime::describe::describe_config(c)?, c.num("num_hidden_layers").unwrap_or(0), max_pos)
}

/// `lowered_of` of the description `d` of a model of `nl` layers, and of the kinds whose loader is the one-GPU route's alone: attention whose layers each end in a
/// scalar after their residual's add (`Residual::AddThenScale`: `lower::gemma4`, src/cli.rs `open_lossy`). What a lossy level's plan counts (src/plan.rs
/// `sizing::level_of`).
pub fn lowered_described(d: &crate::runtime::describe::Described, nl: usize, max_pos: usize) -> Result<ModelDesc> {
    use super::lower;
    if d.desc.blocks.iter().any(|b| matches!(b.residual, crate::runtime::kinds::Residual::AddThenScale)) {
        return lower::gemma4(d, &|n: &str| d.shapes.get(n).cloned(), Vec::new(), max_pos);
    }
    // (layers of plain attention with a plain or a scaled add: the dense lowering, the one-GPU route's too: `support::Family::Layers`)
    use crate::runtime::kinds::{Body, Mixer, Residual};
    let plain_attention = |b: &crate::runtime::kinds::Block| !matches!(b.body, Body::Mixer(_)) || matches!(b.body, Body::Mixer(Mixer::Attention(_)));
    if d.desc.blocks.iter().all(|b| plain_attention(b) && matches!(b.residual, Residual::Add | Residual::AddScaled(_))) {
        return lower::dense(d, Vec::new(), max_pos);
    }
    let st = Stage::whole(nl);
    match lowering(d)? {
        Lowering::Compressed => lower::deepseek_v4(d, Vec::new(), max_pos, &st, &super::deepseek_v4::V4Opts::of(crate::weights::Layout::Tiered, max_pos)),
        Lowering::Mamba2 => lower::nemotron(d, Vec::new(), max_pos, &st),
        Lowering::DeltaStreams => lower::glm(d, Vec::new(), max_pos, &st),
        Lowering::DeltaDepth => lower::kimi(d, Vec::new(), max_pos, &st, &vec![(true, true); nl], false),
        Lowering::GatedResidual => lower::muse(d, Vec::new(), max_pos, &st),
        Lowering::GatedStreams => lower::qwen4x(d, Vec::new(), max_pos),
        Lowering::GatedDelta => lower::qwen35(d, Vec::new(), max_pos, &st),
        Lowering::LatentPlain => lower::latent_moe(d, Vec::new(), max_pos, &st),
    }
}

/// A number of the config (its `text_config`'s for a model that nests one).
pub fn config_num(text: &str, key: &str) -> Option<usize> {
    crate::runtime::config::ModelConfig::parse(text).ok()?.num(key)
}

/// Where the dynamic route's layers go and how many of their experts each GPU holds: the planner's (src/plan.rs `sizing`).
pub use crate::plan::sizing::{
    fit, fit_table, fit_table_with, hb_layer_bytes, layer_bytes, layer_bytes_table, layer_bytes_where, layer_bytes_with, places_plan, places_plan_at,
    places_plan_dense, plan_cuts, ram_bytes, ram_bytes_table, CODED_PCT, PLAIN_PCT, TIERED_PCT, TWELVE_PCT,
};

/// A model of the dynamic route in stages (`open`'s), stage k's layers with their experts in the expert cache as `cachings[k]` says: a hummingbird directory's (src/hbg_store.rs: places on its GPU, the computer's memory, the disk),
/// or the release's where its experts are each the same bytes (src/tiering.rs, R12: DeepSeek-V4's FP4). A loader whose experts the cache does not take refuses it.
pub fn open_cached(dir: &Path, o: &Opts, cuts: &[usize], devices: &[i32], cachings: Vec<crate::hbg_store::HbCaching>) -> Result<Run> {
    open_cached_of(dir, &ModelConfig::load(dir)?, o, cuts, devices, cachings)
}

/// `open_cached` of a config.json parsed once.
pub fn open_cached_of(dir: &Path, m: &ModelConfig, o: &Opts, cuts: &[usize], devices: &[i32], cachings: Vec<crate::hbg_store::HbCaching>) -> Result<Run> {
    let (nl, eos, load) = cached_stage(dir, m)?;
    let firsts: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).collect();
    let make: MakeStage = std::sync::Arc::new(move |st, o: &Opts| {
        let k = firsts.iter().position(|&f| f == st.layers.0).unwrap_or(0);
        load(st, o, cachings.get(k).or(cachings.last()))
    });
    Run::new(nl, eos, cuts, devices, o, make)
}

/// The model of `dir` with every layer across `devices` (src/dynamic/tp.rs), its experts in the expert cache as `caching` says: `places` of a layer's in each
/// GPU's slices of its places, the rest in the cache's host tier, shared by the group (src/dynamic/tp_cache.rs).
pub fn open_tp_cached(dir: &Path, o: &Opts, devices: &[i32], caching: crate::hbg_store::HbCaching) -> Result<Run> {
    open_tp_cached_of(dir, &ModelConfig::load(dir)?, o, devices, caching)
}

/// `open_tp_cached` of a config.json parsed once.
pub fn open_tp_cached_of(dir: &Path, m: &ModelConfig, o: &Opts, devices: &[i32], caching: crate::hbg_store::HbCaching) -> Result<Run> {
    let (nl, eos, load) = cached_stage(dir, m)?;
    Ok(Run::Tp(super::tp::Tp::with(nl, eos, devices, o, std::sync::Arc::new(move |st, o: &Opts| load(st, o, Some(&caching))))?))
}

/// A stage of a model with its experts as a caching says (none: as the stage's level holds them).
type CachedStage = std::sync::Arc<dyn Fn(Stage, &Opts, Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> + Send + Sync>;

/// The layers, end tokens and stage loader (`CachedStage`) of the model of `dir` (`m`: its config.json, parsed once, which each stage's loader reads from), by its
/// description's loader (`loader`).
fn cached_stage(dir: &Path, m: &ModelConfig) -> Result<(usize, Vec<i64>, CachedStage)> {
    let (d, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let l = loader(&d).map_err(|e| e.context(&label))?;
    let (nl, eos) = (l.ends)(dir, m)?;
    let (owned, m) = (dir.to_path_buf(), std::sync::Arc::new(m.clone()));
    Ok((nl, eos, std::sync::Arc::new(move |st, o: &Opts, c| (l.stage)(&owned, &m, o, st, c))))
}

#[cfg(test)]
mod tests {
    use super::{fit, layer_bytes, plan_cuts, TWELVE_PCT};
    use crate::sizes::layer_of;

    #[test]
    fn cuts_follow_the_room() {
        let per = vec![10u64; 8];
        assert_eq!(plan_cuts(&per, 4, &[(0, 50), (1, 100)]), Some((vec![], vec![1]))); // (the freest holds it all)
        assert_eq!(plan_cuts(&per, 4, &[(0, 50), (1, 50)]), Some((vec![4], vec![0, 1]))); // (even: 2 + 40 and 40 + 2)
        assert_eq!(plan_cuts(&per, 4, &[(0, 30), (1, 30)]), None); // (84 bytes, 60 of room)
        assert_eq!(plan_cuts(&per, 4, &[(0, 5), (1, 45), (2, 45)]), Some((vec![4], vec![1, 2]))); // (no layer fits beside the embedding on GPU 0)
        assert_eq!(layer_of("model.layers.12.mlp.experts.3.w1.weight"), Some(12));
        assert_eq!(layer_of("lm_head.weight"), None);
    }

    /// The matrices a layer in the computer's memory keeps there (src/dense_ram.rs) are the ones the executor's loader puts there: every matrix of the layer's
    /// own products, Muse Spark's attention and latent projections, Kimi-K3's delta and latent attention's and its dense MLP's (not a latent attention's kv_b
    /// or indexer, small tensors, nor a block of experts' own matrices); none of DeepSeek-V4's, nor of the Qwen3.5 layout's, dense or with experts (its row keeps
    /// no layer there: every expert is on its GPU).
    #[test]
    fn the_matrices_a_layer_keeps_in_memory_are_its_loaders() {
        let m = |n: &str| crate::plan::sizing::in_ram("Muse Spark", n);
        assert!(m("model.language_model.layers.3.self_attn.q_proj.weight") && m("model.language_model.layers.3.self_attn.o_proj.weight"));
        assert!(m("model.language_model.layers.0.mlp.pre_expert_proj.weight") && m("model.language_model.layers.0.mlp.post_expert_proj.weight"));
        assert!(!m("model.language_model.layers.0.self_attn.q_norm.weight") && !m("model.language_model.layers.0.mlp.experts.gate_up_proj"));
        assert!(!m("model.language_model.layers.0.mlp.gate.weight") && !m("model.language_model.embed_tokens.weight"));
        let k = |n: &str| crate::plan::sizing::in_ram("Kimi-K3", n);
        let l = |x: &str| format!("language_model.model.layers.{x}");
        assert!(k(&l("2.self_attn.f_a_proj.weight")) && k(&l("0.self_attn.o_proj.weight")) && k(&l("0.mlp.gate_proj.weight")));
        assert!(k(&l("3.self_attn.q_a_proj.weight")) && k(&l("3.self_attn.kv_a_proj_with_mqa.weight")) && k(&l("3.self_attn.o_proj.weight")));
        assert!(!k(&l("3.self_attn.kv_b_proj.weight")) && !k(&l("3.self_attn.indexer.wq_b.weight")) && !k(&l("3.self_attn.indexer.weights_proj.weight")));
        assert!(!k(&l("1.self_attn.q_conv1d.weight")) && !k(&l("1.block_sparse_moe.shared_experts.up_proj.weight")));
        assert!(!k(&l("1.block_sparse_moe.routed_expert_down_proj.weight")) && !k(&l("1.block_sparse_moe.experts.0.w1.weight_packed")));
        assert!(!crate::plan::sizing::in_ram("DeepSeek-V4", "model.layers.0.self_attn.wo.weight"));
        let q = |n: &str| crate::plan::sizing::in_ram("Qwen3.5", n);
        assert!(
            !q("model.layers.3.self_attn.q_proj.weight")
                && !q("model.layers.0.linear_attn.in_proj_qkv.weight")
                && !q("model.layers.0.mlp.experts.gate_up_proj")
        );
        assert!(!q("model.layers.0.mlp.shared_expert.up_proj.weight") && !q("model.layers.0.mlp.gate.weight"));
    }

    /// A plan of experts in the computer's memory across GPUs opens the latent block's cache across them (src/dynamic/tp_cache.rs) by the description's
    /// kind, not the family's name: Kimi-K3's released config has such experts; GLM-5.3-Flash's (experts on the layer's input) and DeepSeek-V4's do not.
    #[test]
    fn the_cache_across_gpus_is_the_latent_blocks() {
        let describe = |text: &str| crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
        let fixture = |f: &str| std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap();
        let config = |f: &str| {
            let h: serde_json::Value = serde_json::from_str(&fixture(f)).unwrap();
            h["config"].as_str().unwrap().to_string()
        };
        assert!(super::latent_experts_of(&describe(&config("headers/Kimi-K3.json"))));
        assert!(!super::latent_experts_of(&describe(&fixture("glm5_next_config.json"))));
        assert!(!super::latent_experts_of(&describe(&config("headers/DeepSeek-V4-Flash-0731.json"))));
    }

    /// The dynamic route's loader is chosen by the layer kinds a description holds, not by a family's name: each released config (the random models' where none
    /// is here) is lowered by the loader of its kinds, and a description of kinds no loader of the route lowers (Gemma 4's, the plain dense kind's) is refused
    /// in words. The family a support row names is its row's (its layers split on the route).
    #[test]
    fn the_loader_is_chosen_by_the_descriptions_kinds() {
        use super::Lowering::*;
        let fixture = |f: &str| std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap();
        let header = |f: &str| {
            let h: serde_json::Value = serde_json::from_str(&fixture(f)).unwrap();
            h["config"].as_str().unwrap().to_string()
        };
        let lowering = |text: &str| super::lowering(&crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap());
        let q35: serde_json::Value = serde_json::from_str(&fixture("qwen35_configs.json")).unwrap();
        let q38 = q35["Qwen/Qwen3.8-27B"].to_string();
        for (text, want) in [
            (header("headers/DeepSeek-V4-Flash-0731.json"), Compressed),
            (fixture("nemotron_h_config.json"), Mamba2),
            (fixture("nemotron_h_moetoy_config.json"), Mamba2),
            (fixture("glm5_next_config.json"), DeltaStreams),
            (fixture("glm5_next_tiny_config.json"), DeltaStreams),
            (header("headers/Kimi-K3.json"), DeltaDepth),
            (q38.clone(), GatedDelta),
            (fixture("qwen35moe_tiny_config.json"), GatedDelta),
            (fixture("qwen4x_config.json"), GatedStreams),
        ] {
            assert_eq!(lowering(&text).unwrap(), want, "{}", &text[..text.len().min(120)]);
        }
        #[cfg(feature = "dev")]
        assert_eq!(lowering(crate::muse_ref::RANDOM_CONFIG).unwrap(), GatedResidual);
        let (gemma, dense) = (header("headers/gemma-4-12B-it.json"), header("dense_tiny_manifest.json"));
        for text in [&gemma, &dense] {
            assert!(lowering(text).unwrap_err().msg.contains("no loader of the dynamic route lowers its layer kinds"));
        }
        assert_eq!((super::family(&q38), super::family(&fixture("nemotron_h_config.json"))), (Some("Qwen3.5"), Some("Nemotron-H")));
        assert_eq!((super::family(&gemma), super::family(&dense)), (None, None));
    }

    /// The expert cache's plan: the dense bytes cut over every GPU (even where one would hold them), each stage's room after them as places for its experts.
    #[test]
    fn places_follow_the_room_left() {
        use super::{places_plan, places_plan_dense};
        // (8 layers of 10 dense bytes, the first with no experts, 4 bytes besides; 2 GPUs of 100: 42 and 42 dense, 58 left each for 3 and 4 expert layers of 2-byte experts)
        let (dense, moe) = (vec![10u64; 8], (0..8).map(|l| l > 0).collect::<Vec<_>>());
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 100), (1, 100)], 2, 64, 4), Some((vec![4], vec![0, 1], vec![9, 7])));
        // (at most the experts a layer has; fewer than `least` places: none)
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 100), (1, 100)], 2, 5, 4).map(|p| p.2), Some(vec![5, 5]));
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 50), (1, 50)], 2, 64, 4), None);
        // (the same places for every layer first: 3 GPUs of 100 for 8 layers of 10, 7 of them with experts; 11 places a layer (32 bytes) fit them as 3, 3 and 2
        // layers, 12 do not; then each GPU's room left over: 68 for 2 expert layers, 70 for 3, 78 for 2)
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 100), (1, 100), (2, 100)], 2, 64, 4), Some((vec![3, 6], vec![0, 1, 2], vec![17, 11, 19])));
        // (one GPU too small for a token's experts besides the layers' weights: the last layers' dense matrices in the computer's memory, as few as fit, and one slot of the largest on the GPU)
        let ram = vec![3u64; dense.len()];
        assert_eq!(
            places_plan_dense(&dense, 4, &moe, &[(0, 100)], 2, 64, 4, &ram, &[]).map(|p| p.3),
            places_plan(&dense, 4, &moe, &[(0, 100)], 2, 64, 4).map(|_| 0)
        );
        let tight = dense.iter().sum::<u64>() + 4 + moe.iter().filter(|&&m| m).count() as u64 * 4 * 2 - 7;
        let got = places_plan_dense(&dense, 4, &moe, &[(0, tight)], 2, 64, 4, &ram, &[]).expect("fits with dense layers in the computer's memory");
        assert!(places_plan(&dense, 4, &moe, &[(0, tight)], 2, 64, 4).is_none());
        assert!(got.3 >= 1 && got.4 == 3 * got.3 as u64 - 3, "{got:?}");
        assert_eq!(places_plan_dense(&dense, 4, &moe, &[(0, 10)], 2, 64, 4, &ram, &[]), None);
    }

    /// An FP8 release on HBM GPUs: plain where that fits, coded where only that does, no cut where neither; coded alone on other GPUs.
    #[test]
    fn fp8_is_plain_where_it_fits_and_coded_where_only_that_does() {
        let dir = std::env::temp_dir().join(format!("glyd-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // (two layers of one FP8 matrix each, 256 x 1,024 bytes, and a bf16 head of 256 x 256)
        let header = r#"{"model.layers.0.mlp.w.weight":{"dtype":"F8_E4M3","shape":[256,1024],"data_offsets":[0,262144]},"model.layers.1.mlp.w.weight":{"dtype":"F8_E4M3","shape":[256,1024],"data_offsets":[262144,524288]},"lm_head.weight":{"dtype":"BF16","shape":[256,256],"data_offsets":[524288,655360]}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend(header.as_bytes());
        f.resize(f.len() + 655360, 0);
        std::fs::write(dir.join("model.safetensors"), f).unwrap();
        let (raw, base) = layer_bytes(&dir, 2, 100).unwrap();
        assert_eq!((raw, base), (vec![262144, 262144], 131072 * 72 / 100));
        let (plain, coded) = (2 * (262144 * 113 / 100) + base, 2 * (262144 * 90 / 100) + base);
        let one = |room: u64, hbm: bool| {
            fit(&dir, 2, &[(0, room)], hbm, false).map(|(b, c, switched, _)| (b.0.iter().sum::<u64>() + b.1, c.is_some(), switched)).unwrap()
        };
        assert_eq!(one(plain, true), (plain, true, false));
        assert_eq!(one(plain - 1, true), (coded, true, true));
        assert_eq!(one(coded - 1, true), (coded, false, false));
        assert_eq!(one(coded, false), (coded, true, false));
        // (a loader that takes the 12-bit layout: the bf16 head in it where it fits, else tiered, plain FP8 first)
        let base12 = 131072 * TWELVE_PCT / 100;
        let (plain12, coded12) = (plain - base + base12, coded - base + base12);
        let one12 = |room: u64| {
            fit(&dir, 2, &[(0, room)], true, true).map(|(b, c, switched, twelve)| (b.0.iter().sum::<u64>() + b.1, c.is_some(), switched, twelve)).unwrap()
        };
        assert_eq!(one12(plain12), (plain12, true, false, true));
        assert_eq!(one12(plain12 - 1), (plain, true, false, false));
        assert_eq!(one12(plain - 1), (coded12, true, true, true));
        assert_eq!(one12(coded12 - 1), (coded, true, true, false));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
