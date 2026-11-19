//! A model's layers across the GPUs of one machine, every layer on all of them (tensor-parallel): each GPU holds a slice of the rows of every big matrix (a
//! `PackBuf` or an `ExpertStack` whose `tp_rows` is the whole matrix's rows), runs its slice's product and gathers the others' slices over the GPUs' links
//! (kernels/tp.cu), so that every GPU has the whole output; every other operation (norms, attention, the router, the small matrices) every GPU runs itself,
//! on the same data. Each row of a product is the same sum as on one GPU, so a model gives one GPU's digits (the products' kernels as one GPU's choose them).
//! A split by layers (`split::Split`) has one GPU work at a time; here all of them read their weights at once.
//!
//! Each GPU has a thread of its own (`Tp`), its engine and its `Comm`, which the thread holds (`comm`): the products find it there, so no call between needs to
//! carry it.

use super::{Opts, Stage};
use crate::dev::{Buf, Dev, Kernel};
use crate::ffi::{cu, driver, err, Error, Result, Stream};
use crate::launch;
use crate::model::Sample;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

#[cfg(feature = "dev")]
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}
#[cfg(not(feature = "dev"))]
include!(concat!(env!("OUT_DIR"), "/kernel_names.rs"));

/// kernels/tp.cu's blocks a gather.
const BLOCKS: usize = 16;
/// A sender's room in a GPU's staging, a parity: a gather of more rows goes in pieces of as many rows as fit; a block's share of it.
const CAP: usize = 4 << 20;
const SUB: usize = CAP / BLOCKS;
/// Streams a GPU gathers on (each its own sequence of gathers): the engine's and the shared expert's beside it.
const CHANNELS: usize = 2;
/// The bytes a matrix has at least for its rows to be sliced (a smaller one every GPU holds whole: its product is shorter than a gather). A development build's
/// GLYD_TP_MIN_MB sets it.
const MIN_BYTES: usize = 16 << 20;

/// What the GPUs' threads share while they set up: the GPUs, and each one's staging and flags to tell the others.
pub struct Group {
    devices: Vec<i32>,
    posted: Mutex<Vec<Option<[[u64; 2]; CHANNELS]>>>,
    barrier: Meet,
    /// Two places of the group on one GPU (a check on a machine with one): each gather waits for every place's thread to come to it first, as a place's
    /// allocation on that GPU waits for the other place's gather already spinning there.
    shared: bool,
    /// What the GPUs' engines share beyond the gathers, made by the first that asks (`Comm::shared`): the expert cache's host tier and plans.
    extra: Mutex<Option<Arc<dyn std::any::Any + Send + Sync>>>,
}

impl Group {
    pub fn new(devices: &[i32]) -> Arc<Group> {
        let shared = devices.iter().enumerate().any(|(i, d)| devices[..i].contains(d));
        Arc::new(Group { devices: devices.to_vec(), posted: Mutex::new(vec![None; devices.len()]), barrier: Meet::new(devices.len()), shared, extra: Mutex::new(None) })
    }

    /// A GPU's thread of the group has failed: every wait of the others' (now or later) fails too, instead of waiting for it.
    pub fn fail(&self) {
        self.barrier.fail();
    }
}

/// A barrier that one failed thread breaks for all: (the threads come so far, the round, broken) and its condition.
struct Meet {
    n: usize,
    state: Mutex<(usize, u64, bool)>,
    cv: Condvar,
}

impl Meet {
    fn new(n: usize) -> Meet {
        Meet { n, state: Mutex::new((0, 0, false)), cv: Condvar::new() }
    }

    fn wait(&self) -> Result<()> {
        let broken = || Error("another GPU's thread of the group failed".into());
        let mut g = self.state.lock().map_err(|_| broken())?;
        if g.2 {
            return Err(broken());
        }
        let round = g.1;
        g.0 += 1;
        if g.0 == self.n {
            (g.0, g.1) = (0, round + 1);
            self.cv.notify_all();
            return Ok(());
        }
        while g.1 == round && !g.2 {
            g = self.cv.wait(g).map_err(|_| broken())?;
        }
        if g.1 == round { Err(broken()) } else { Ok(()) }
    }

    fn fail(&self) {
        if let Ok(mut g) = self.state.lock() {
            g.2 = true;
        }
        self.cv.notify_all();
    }
}

/// A stream's gathers: the staging and flags the other GPUs write, the sequence of this GPU's, the table of every GPU's staging and flags, the room for this
/// GPU's slice of a product.
struct Chan {
    _stage: Buf,
    flags: Buf,
    ctr: Buf,
    tab: Buf,
    temp: RefCell<(Buf, Vec<Buf>)>,
}

/// This GPU's place among the GPUs of a group, and what it gathers with.
pub struct Comm {
    pub rank: usize,
    pub n: usize,
    group: Arc<Group>,
    gather: Kernel,
    ch: Vec<Chan>,
    streams: RefCell<Vec<Stream>>,
    min_bytes: usize,
    /// One GPU loaded as the first of `n` would be, its gathers left out (`emulated`, a development build's GLYD_TP_EMULATE): the y buffers seen.
    emulated: bool,
    seen: RefCell<std::collections::HashSet<u64>>,
}

thread_local! {
    static CURRENT: RefCell<Option<Rc<Comm>>> = const { RefCell::new(None) };
}

/// The first GPU's gathers so far (every GPU of a group makes the same): `bench` counts a step's, the formula of src/ceiling.rs is checked by them.
pub static GATHERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The group this thread's GPU is in (None: a GPU on its own).
pub fn comm() -> Option<Rc<Comm>> {
    CURRENT.with(|c| c.borrow().clone())
}

impl Comm {
    /// This thread's GPU (`dev`, current) joins the group as `rank`: it can reach every other GPU of the group (peer access; two places on one GPU share its
    /// memory), makes its staging and flags, and waits for the others' to make its table of them. Every GPU of the group calls it at once.
    pub fn join(g: &Arc<Group>, rank: usize, dev: &Dev) -> Result<Rc<Comm>> {
        let (n, d) = (g.devices.len(), driver()?);
        for &peer in g.devices.iter().filter(|&&p| p != dev.ordinal) {
            let (mut can, mut ctx, mut pd) = (0, std::ptr::null_mut(), 0);
            // SAFETY: host out-pointers; the peer's primary context is retained for the process, as `Dev::new` does; 704: access already on.
            unsafe {
                cu("cuDeviceCanAccessPeer", (d.cuDeviceCanAccessPeer)(&mut can, dev.ordinal, peer))?;
                if can == 0 {
                    return err(format!("GPU {} cannot reach GPU {peer} directly (no peer access): the layers across GPUs need it", dev.ordinal));
                }
                cu("cuDeviceGet", (d.cuDeviceGet)(&mut pd, peer))?;
                cu("cuDevicePrimaryCtxRetain", (d.cuDevicePrimaryCtxRetain)(&mut ctx, pd))?;
                let r = (d.cuCtxEnablePeerAccess)(ctx, 0);
                if r != 0 && r != 704 {
                    cu("cuCtxEnablePeerAccess", r)?;
                }
            }
        }
        dev.make_current()?;
        let mut ch = Vec::new();
        for _ in 0..CHANNELS {
            ch.push(Chan { _stage: Buf::new(2 * n * CAP)?, flags: Buf::zeroed(n * BLOCKS * 4)?, ctr: Buf::zeroed(BLOCKS * 4)?, tab: Buf::new(2 * n * 8)?, temp: RefCell::new((Buf::new(16)?, Vec::new())) });
        }
        let mine: [[u64; 2]; CHANNELS] = std::array::from_fn(|c| [ch[c]._stage.ptr, ch[c].flags.ptr]);
        g.posted.lock().map_err(|_| Error("the GPUs' group is poisoned".into()))?[rank] = Some(mine);
        g.barrier.wait()?;
        let all = g.posted.lock().map_err(|_| Error("the GPUs' group is poisoned".into()))?.clone();
        for (c, chan) in ch.iter().enumerate() {
            let mut tab = vec![0u64; 2 * n];
            for (q, p) in all.iter().enumerate() {
                let p = p.ok_or_else(|| Error(format!("GPU {q} of the group did not set up")))?;
                (tab[q], tab[n + q]) = (p[c][0], p[c][1]);
            }
            chan.tab.write(0, &tab)?;
        }
        let min_bytes = crate::tune!("GLYD_TP_MIN_MB").and_then(|v| v.parse::<usize>().ok()).map_or(MIN_BYTES, |m| m << 20);
        Ok(Rc::new(Comm { rank, n, group: g.clone(), gather: dev.kernel(kname!("tp_gather"))?, ch, streams: RefCell::new(Vec::new()), min_bytes, emulated: false, seen: RefCell::new(Default::default()) }))
    }

    /// This GPU (`dev`) as the first of `n` GPUs would load and run a model, alone: each sliced product its slice's alone, its gathers left out, the
    /// outputs not the model's. What a GPU of a group runs, timed on one (tpbench --one with GLYD_TP_EMULATE=N; docs/ceiling.md).
    pub fn emulated(n: usize, dev: &Dev) -> Result<Rc<Comm>> {
        let g = Group::new(&[dev.ordinal]);
        let mut c = Comm::join(&g, 0, dev)?;
        let m = Rc::get_mut(&mut c).ok_or_else(|| Error("an emulated group shared".into()))?;
        (m.n, m.emulated) = (n, true);
        Ok(c)
    }

    /// The one `T` of the group: made by the first GPU's thread that asks, the same for every other.
    pub fn shared<T: std::any::Any + Send + Sync>(&self, make: impl FnOnce() -> T) -> Result<Arc<T>> {
        let mut x = self.group.extra.lock().map_err(|_| Error("the GPUs' group is poisoned".into()))?;
        let any = x.get_or_insert_with(|| Arc::new(make()) as Arc<dyn std::any::Any + Send + Sync>).clone();
        any.downcast::<T>().map_err(|_| Error("the GPUs' group shares something else".into()))
    }

    /// Every GPU's thread of the group comes here before any goes on: the engines are made before the first pass (whose gathers would otherwise spin on a
    /// GPU for as long as the slowest load takes, past kernels/tp.cu's 10 s).
    pub fn barrier(&self) -> Result<()> {
        self.group.barrier.wait()
    }

    /// Whether a matrix [rows, cols] (bf16) is held as slices of its rows: big enough, and rows in whole blocks of 64 for each GPU.
    pub fn slices(&self, rows: usize, cols: usize) -> bool {
        self.slicer().slices(rows, cols)
    }

    /// This GPU's rows of a matrix of `rows`: the first and how many.
    pub fn rows(&self, rows: usize) -> (usize, usize) {
        self.slicer().rows(rows)
    }

    /// The same answers as plain data, for a loader's other threads.
    pub fn slicer(&self) -> Slicer {
        Slicer { rank: self.rank, n: self.n, min_bytes: self.min_bytes }
    }

    /// The channel of stream `s`: the streams take the channels in the order they first gather, the same on every GPU.
    fn chan(&self, s: Stream) -> Result<&Chan> {
        let mut st = self.streams.borrow_mut();
        let i = match st.iter().position(|x| x.0 == s.0) {
            Some(i) => i,
            None => {
                st.push(s);
                st.len() - 1
            }
        };
        self.ch.get(i).ok_or_else(|| Error(format!("gathers on {} streams: {CHANNELS} are set up", i + 1)))
    }

    /// Room for `bytes` of this GPU's slice of a product on stream `s` (grown where too small; a buffer outgrown is kept, as a captured graph may name it).
    pub fn temp(&self, s: Stream, bytes: usize) -> Result<u64> {
        let mut t = self.chan(s)?.temp.borrow_mut();
        if t.0.bytes < bytes {
            let old = std::mem::replace(&mut t.0, Buf::new(bytes.next_multiple_of(1 << 20))?);
            t.1.push(old);
        }
        Ok(t.0.ptr)
    }

    /// y [m, n sl] (rows `ld` values apart) of every GPU's slice x [m, sl] (this one's here), on stream `s`: this GPU's columns rank sl on.
    ///
    /// # Safety
    /// x and y are live device buffers of those sizes, 16-byte aligned, on this GPU; every GPU of the group gathers the same shape on its stream in turn.
    pub unsafe fn gather(&self, s: Stream, x: u64, m: usize, sl: usize, y: u64, ld: usize) -> Result<()> {
        // (one GPU as the first of n: no gather; an output buffer zeroed the first time it comes, so the next operations read finite values)
        if self.emulated {
            if self.seen.borrow_mut().insert(y) {
                crate::dev::memset_async(y, 0, m * ld * 2, s)?;
            }
            return Ok(());
        }
        if !sl.is_multiple_of(8) || !ld.is_multiple_of(8) || !x.is_multiple_of(16) || !y.is_multiple_of(16) || ld < self.n * sl {
            return err(format!("a gather of slices of {sl} into rows of {ld} at {x:#x} and {y:#x}: widths of 8s, 16-byte aligned"));
        }
        let c = self.chan(s)?;
        if self.group.shared {
            self.barrier()?;
        }
        if self.rank == 0 {
            GATHERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // (a piece of rows a call, the blocks as many as its vectors take of a block's region: SUB bytes each)
        let piece = (CAP / (sl * 2)).max(1);
        let mut r0 = 0;
        while r0 < m {
            let rn = piece.min(m - r0);
            let (xs, ys) = (x + (r0 * sl * 2) as u64, y + (r0 * ld * 2) as u64);
            let blocks = (rn * sl * 2).div_ceil(SUB).clamp(1, BLOCKS) as u32;
            launch!(self.gather, [blocks, 1, 1], [256, 1, 1], s; xs, rn as i32, (sl / 8) as i32, ys, (ld / 8) as i64, self.rank as i32, self.n as i32, c.tab.ptr, c.flags.ptr, c.ctr.ptr, (SUB / 16) as i64)?;
            r0 += rn;
        }
        Ok(())
    }
}

/// What slices a GPU of a group holds (`Comm::slices`, `Comm::rows`), as plain data.
#[derive(Clone, Copy, Debug)]
pub struct Slicer {
    pub rank: usize,
    pub n: usize,
    min_bytes: usize,
}

impl Slicer {
    /// The slicing of `n` GPUs as a model's load makes it (the formula of src/ceiling.rs asks).
    pub fn of(n: usize) -> Slicer {
        let min_bytes = crate::tune!("GLYD_TP_MIN_MB").and_then(|v| v.parse::<usize>().ok()).map_or(MIN_BYTES, |m| m << 20);
        Slicer { rank: 0, n, min_bytes }
    }

    pub fn slices(&self, rows: usize, cols: usize) -> bool {
        self.n > 1 && rows.is_multiple_of(64 * self.n) && rows * cols * 2 >= self.min_bytes
    }

    pub fn rows(&self, rows: usize) -> (usize, usize) {
        let k = rows / self.n;
        (self.rank * k, k)
    }
}

/// The slicer of this thread's group, if it is one's.
pub fn slicer() -> Option<Slicer> {
    comm().map(|c| c.slicer())
}

enum Cmd {
    Reset,
    Prefill(Vec<u32>),
    Step(u32),
    Logits,
    Logprob(f32),
    Describe,
}

enum Out {
    Done(Result<()>),
    Sample(Result<Sample>),
    Logits(Result<Vec<f32>>),
    Logprob(Result<f32>),
    Text(String),
}

struct Worker {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Out>,
    h: Option<std::thread::JoinHandle<()>>,
}

/// A model with every layer on each of `devices`, a thread each: every command goes to all of them, and the first one's answer is the model's.
pub struct Tp {
    w: Vec<Worker>,
    n: usize,
    ctx: usize,
    eos: Vec<i64>,
}

fn gone() -> Error {
    Error("a GPU's thread of the group ended".into())
}

impl Tp {
    /// The model `make` builds (`Stage::whole`) on each of `devices` (two places on one GPU check the gathers on a machine with one), its rows sliced
    /// across them.
    pub fn with(nl: usize, eos: Vec<i64>, devices: &[i32], o: &Opts, make: super::split::MakeStage) -> Result<Tp> {
        if devices.len() < 2 {
            return err("the layers across GPUs need two GPUs at least");
        }
        let g = Group::new(devices);
        let mut w = Vec::new();
        for (rank, &device) in devices.iter().enumerate() {
            let (tx, crx) = channel::<Cmd>();
            let (otx, rx) = channel::<Out>();
            let (make, mut o, g) = (make.clone(), o.clone(), g.clone());
            o.device = device;
            let h = std::thread::Builder::new()
                .name(format!("glyd-gpu-{rank}"))
                .spawn(move || {
                    let made = Dev::new(device).and_then(|dev| Comm::join(&g, rank, &dev)).and_then(|c| {
                        CURRENT.with(|x| *x.borrow_mut() = Some(c));
                        make(Stage::whole(nl), &o)
                    });
                    let mut e = match made {
                        Ok(e) => e,
                        Err(x) => {
                            // (the others' waits for this one fail rather than wait)
                            g.fail();
                            let _ = otx.send(Out::Done(Err(x)));
                            return;
                        }
                    };
                    let _ = otx.send(Out::Done(Ok(())));
                    while let Ok(c) = crx.recv() {
                        let out = match c {
                            Cmd::Reset => Out::Done(e.reset()),
                            Cmd::Prefill(t) => Out::Sample(e.prefill(&t)),
                            Cmd::Step(t) => Out::Sample(e.step(t)),
                            Cmd::Logits => Out::Logits(e.logits()),
                            Cmd::Logprob(l) => Out::Logprob(e.logprob(l)),
                            Cmd::Describe => Out::Text(e.describe()),
                        };
                        if otx.send(out).is_err() {
                            break;
                        }
                    }
                    // (the engine is freed on its own thread, its context current)
                    drop(e);
                    CURRENT.with(|x| *x.borrow_mut() = None);
                })
                .map_err(|x| Error(format!("a GPU's thread: {x}")))?;
            w.push(Worker { tx: Some(tx), rx, h: Some(h) });
        }
        // (every GPU's load answered; the first error that is not another's failure is the one told)
        let mut errs = Vec::new();
        for wk in &w {
            match wk.rx.recv() {
                Ok(Out::Done(Ok(()))) => {}
                Ok(Out::Done(Err(e))) => errs.push(e),
                _ => errs.push(gone()),
            }
        }
        if !errs.is_empty() {
            let pick = errs.iter().position(|e| !e.0.contains("another GPU's thread")).unwrap_or(0);
            return Err(errs.swap_remove(pick));
        }
        Ok(Tp { w, n: 0, ctx: o.ctx.div_ceil(16) * 16, eos })
    }

    /// `c` to every GPU's thread (at once: their gathers wait for each other), the first one's answer, or the first error any gave.
    fn all(&self, c: impl Fn() -> Cmd) -> Result<Out> {
        for wk in &self.w {
            wk.tx.as_ref().ok_or_else(gone)?.send(c()).map_err(|_| gone())?;
        }
        let mut first = None;
        let mut bad = None;
        for wk in &self.w {
            let o = wk.rx.recv().map_err(|_| gone())?;
            let failed = match &o {
                Out::Done(Err(e)) => Some(e.0.clone()),
                Out::Sample(Err(e)) => Some(e.0.clone()),
                Out::Logits(Err(e)) => Some(e.0.clone()),
                Out::Logprob(Err(e)) => Some(e.0.clone()),
                _ => None,
            };
            bad = bad.or(failed);
            first.get_or_insert(o);
        }
        match bad {
            Some(e) => Err(Error(e)),
            None => first.ok_or_else(gone),
        }
    }

    fn one(&self, c: Cmd) -> Result<Out> {
        self.w[0].tx.as_ref().ok_or_else(gone)?.send(c).map_err(|_| gone())?;
        self.w[0].rx.recv().map_err(|_| gone())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.all(|| Cmd::Reset)?;
        self.n = 0;
        Ok(())
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        match self.all(|| Cmd::Prefill(tokens.to_vec()))? {
            Out::Sample(s) => {
                self.n += tokens.len();
                s
            }
            _ => Err(gone()),
        }
    }

    pub fn step(&mut self, token: u32) -> Result<Sample> {
        match self.all(|| Cmd::Step(token))? {
            Out::Sample(s) => {
                self.n += 1;
                s
            }
            _ => Err(gone()),
        }
    }

    /// The last pass's logits (the first GPU's: every GPU has them whole).
    pub fn logits(&self) -> Result<Vec<f32>> {
        match self.one(Cmd::Logits)? {
            Out::Logits(r) => r,
            _ => Err(gone()),
        }
    }

    pub fn logprob(&self, logit: f32) -> Result<f32> {
        match self.one(Cmd::Logprob(logit))? {
            Out::Logprob(r) => r,
            _ => Err(gone()),
        }
    }

    pub fn describe(&self) -> String {
        let first = match self.one(Cmd::Describe) {
            Ok(Out::Text(t)) => t,
            _ => "gone".into(),
        };
        format!("{} GPUs, every layer sliced across them; GPU 0: {first}", self.w.len())
    }

    pub fn capacity(&self) -> usize {
        self.ctx
    }

    /// A reply as `Split::reply` gives one.
    pub fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        use crate::model::Finish;
        if prompt.is_empty() || prompt.len() + 1 > self.ctx {
            return err(format!("{} prompt tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        let mut s = self.prefill(prompt)?;
        let mut out = Vec::new();
        loop {
            let tok = if greedy { s.token } else { sampler.sample_with(None, || self.logits())? };
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
            s = self.step(tok)?;
        }
    }
}

impl Drop for Tp {
    fn drop(&mut self) {
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

/// `tpcheck DIR [--tp 0,0] [--tokens 24] [--chunk 64] [--ctx 1024]` (development): the model of `dir` on the first GPU of `--tp` alone and then with every
/// layer across the GPUs of `--tp`, the same tokens through both (a decode step a token, then the whole list as a prompt in chunks): their logits compared
/// value for value.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    use super::split::{config_num, open, open_tp, tp_args, Run};
    let dir = std::path::Path::new(a.need(1, "DIR")?);
    let gpus = tp_args(a)?.unwrap_or_else(|| vec![0, 0]);
    let n: usize = a.num("tokens", 24)?;
    let o = Opts { ctx: a.num("ctx", 1024)?, chunk: a.num("chunk", 64)?, graph: !a.has("no-graph"), device: gpus[0], layout: a.layout()? };
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let vocab = config_num(&text, "vocab_size").unwrap_or(1000).max(2) as u32;
    let toks: Vec<u32> = (0..n).map(|i| (i as u32 * 37 + 11) % vocab).collect();
    let run = |e: &mut Run| -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        for (i, &t) in toks.iter().enumerate() {
            if i == 0 {
                e.prefill(&[t])?;
            } else {
                e.step(t)?;
            }
            out.push(e.logits()?);
        }
        e.reset()?;
        e.prefill(&toks)?;
        out.push(e.logits()?);
        Ok(out)
    };
    let want = {
        let mut one = open(dir, &o, &[], &[gpus[0]])?;
        println!("one GPU: {}", one.describe());
        run(&mut one)?
    };
    let mut tp = open_tp(dir, &o, &gpus)?;
    println!("{}", tp.describe());
    let got = run(&mut tp)?;
    let mut same = 0;
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        let diff = w.iter().zip(g).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let worst = w.iter().zip(g).fold(0f32, |m, (x, y)| if (x - y).is_nan() { f32::INFINITY } else { m.max((x - y).abs()) });
        same += (diff == 0) as usize;
        if diff > 0 || i < 3 || i + 1 == want.len() {
            println!("{} {i:3}: {diff} of {} logits differ, the largest by {worst:.5}", if i + 1 == want.len() { "prompt" } else { "step  " }, w.len());
        }
    }
    println!("{same} of {} passes the same to the bit", want.len());
    Ok(())
}

/// `tpbench DIR [--tp 0,1] [--split CUT --devices 0,1] [--steps 128] [--ctx 4096]` (development): decode steps' time, a token after another (each step's sample read),
/// with every layer across the GPUs of `--tp` and, where `--split` is given, split by layers over `--devices` as well; the tokens of each, the same where they agree.
#[cfg(feature = "dev")]
pub fn bench(a: &crate::args::Args) -> Result<()> {
    use super::split::{open, open_tp, split_args, tp_args, Run};
    let dir = std::path::Path::new(a.need(1, "DIR")?);
    let steps: usize = a.num("steps", 128)?;
    let o = Opts { ctx: a.num("ctx", 4096)?, chunk: a.num("chunk", 256)?, graph: !a.has("no-graph"), device: 0, layout: a.layout()? };
    let run = |what: &str, e: &mut Run| -> Result<Vec<u32>> {
        // (a pass's gathers counted in the prompt's: decode steps replay a graph, whose gathers no call counts)
        let g0 = GATHERS.load(std::sync::atomic::Ordering::Relaxed);
        let mut s = e.prefill(&[1, 2, 3, 4])?;
        let g = GATHERS.load(std::sync::atomic::Ordering::Relaxed) - g0;
        let mut toks = vec![s.token];
        for _ in 0..8 {
            s = e.step(s.token)?;
        }
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            s = e.step(s.token)?;
            toks.push(s.token);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!("{what}: {ms:.2} ms a step ({:.1} tokens/s) over {steps} steps, {g} gathers a pass", 1e3 / ms);
        Ok(toks)
    };
    // (--one: the model on one GPU alone, the step the formula of src/ceiling.rs is checked against)
    // (GLYD_TP_EMULATE=N in a development build: the one GPU loads and runs as the first of N would, gathers left out: Comm::emulated)
    if let Some(n) = crate::tune!("GLYD_TP_EMULATE").and_then(|v| v.parse::<usize>().ok()).filter(|_| a.has("one")) {
        let dev = Dev::new(o.device)?;
        let c = Comm::emulated(n, &dev)?;
        CURRENT.with(|x| *x.borrow_mut() = Some(c));
    }
    if a.has("one") {
        // (--prof: graphs off, each stage of a step timed on the GPU, ms a step: what the formula's terms are checked against, a term at a time)
        // (GLYD_PROF_GRAPH=1: the stages timed inside the step graph, each replay read)
        let (prof, in_graph) = (a.has("prof"), crate::tune!("GLYD_PROF_GRAPH").as_deref() == Some("1"));
        let mut e = open(dir, &Opts { graph: (!prof || in_graph) && o.graph, ..o.clone() }, &[], &[o.device])?;
        println!("{}", e.describe());
        if !prof {
            run("one GPU", &mut e)?;
            return Ok(());
        }
        let Run::One(d) = &mut e else { return err("--prof: the model on one GPU") };
        if in_graph {
            d.prof_in_graph()?;
        } else {
            d.prof = Some(std::cell::RefCell::new(Vec::new()));
        }
        let mut s = e.prefill(&[1, 2, 3, 4])?;
        for _ in 0..8 {
            s = e.step(s.token)?;
        }
        let Run::One(d) = &e else { unreachable!() };
        d.prof_take()?;
        let mut stages: Vec<(&'static str, f64)> = Vec::new();
        let t0 = std::time::Instant::now();
        for k in 0..steps {
            s = e.step(s.token)?;
            if in_graph || k + 1 == steps {
                let Run::One(d) = &e else { unreachable!() };
                for (name, v) in d.prof_take()? {
                    match stages.iter_mut().find(|x| x.0 == name) {
                        Some(x) => x.1 += v,
                        None => stages.push((name, v)),
                    }
                }
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        let total: f64 = stages.iter().map(|x| x.1).sum::<f64>() / steps as f64;
        println!("one GPU, {}: {ms:.3} ms a step, {total:.3} ms of it in the stages below (GPU time, {steps} steps)", if in_graph { "in the step graph" } else { "graphs off" });
        for (k, v) in &stages {
            println!("{:>44} {:8.3} ms {:5.1}%", k, v / steps as f64, 100.0 * v / steps as f64 / total);
        }
        return Ok(());
    }
    let gpus = tp_args(a)?.unwrap_or_else(|| vec![0, 1]);
    let tp = {
        let mut e = open_tp(dir, &o, &gpus)?;
        println!("{}", e.describe());
        run(&format!("across GPUs {gpus:?}"), &mut e)?
    };
    if a.flag("split").is_some() {
        let (cuts, devices) = split_args(a)?;
        let mut e = open(dir, &o, &cuts, &devices)?;
        let sp = run(&format!("split at {cuts:?} over GPUs {devices:?}"), &mut e)?;
        let same = tp.iter().zip(&sp).take_while(|(x, y)| x == y).count();
        println!("the tokens agree for the first {same} of {}", tp.len());
    }
    Ok(())
}

/// `tpgather [--tp 0,1] [--reps 2000]` (development): the gather's own time on the GPUs of `--tp`, each a thread, for slices a decode step and a prompt
/// chunk gather (rows x a GPU's columns): launched one after another, and as a graph of 100 replayed.
#[cfg(feature = "dev")]
pub fn gather_bench(a: &crate::args::Args) -> Result<()> {
    let gpus = super::split::tp_args(a)?.unwrap_or_else(|| vec![0, 1]);
    let reps: usize = a.num("reps", 2000)?;
    let g = Group::new(&gpus);
    let shapes: [(usize, usize); 5] = [(1, 896), (1, 4608), (16, 768), (64, 4608), (1024, 4608)];
    let hs: Vec<_> = gpus
        .iter()
        .enumerate()
        .map(|(rank, &d)| {
            let g = g.clone();
            std::thread::spawn(move || -> Result<Vec<(f64, f64)>> {
                let dev = Dev::new(d)?;
                let c = Comm::join(&g, rank, &dev)?;
                let s = dev.stream;
                let mut out = Vec::new();
                for &(m, sl) in &shapes {
                    let n = c.n;
                    let (x, y) = (Buf::zeroed(m * sl * 2)?, Buf::zeroed(m * sl * n * 2)?);
                    let k = if m >= 1024 { reps / 20 } else { reps };
                    // SAFETY: live buffers of those sizes on this GPU; every GPU gathers the same shapes in turn.
                    unsafe {
                        for _ in 0..20 {
                            c.gather(s, x.ptr, m, sl, y.ptr, sl * n)?;
                        }
                        dev.sync()?;
                        c.barrier()?;
                        let t0 = std::time::Instant::now();
                        for _ in 0..k {
                            c.gather(s, x.ptr, m, sl, y.ptr, sl * n)?;
                        }
                        dev.sync()?;
                        let plain = t0.elapsed().as_secs_f64() * 1e6 / k as f64;
                        c.barrier()?;
                        let graph = dev.capture(|| {
                            for _ in 0..100 {
                                c.gather(s, x.ptr, m, sl, y.ptr, sl * n)?;
                            }
                            Ok(())
                        })?;
                        c.barrier()?;
                        let t1 = std::time::Instant::now();
                        for _ in 0..(k / 100).max(1) {
                            graph.launch(s)?;
                        }
                        dev.sync()?;
                        let gr = t1.elapsed().as_secs_f64() * 1e6 / ((k / 100).max(1) * 100) as f64;
                        c.barrier()?;
                        out.push((plain, gr));
                    }
                }
                Ok(out)
            })
        })
        .collect();
    let mut all = Vec::new();
    for h in hs {
        all.push(h.join().map_err(|_| Error("a GPU's thread panicked".into()))??);
    }
    println!("{} GPUs {gpus:?}: a gather's time, launched one after another / in a graph (GPU 0's)", gpus.len());
    for (i, &(m, sl)) in shapes.iter().enumerate() {
        let (p, gr) = all[0][i];
        println!("{m:5} rows x {sl:5} a GPU ({:8.1} KB a GPU): {p:7.1} us / {gr:7.1} us", (m * sl * 2) as f64 / 1e3);
    }
    Ok(())
}
