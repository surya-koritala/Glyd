//! The expert cache planned on the GPU (R12b, #327; docs/design/runtime.md §6): a decode call of a MoE layer whose experts are in the expert cache needs no host inside it, so that a step of such calls is
//! one CUDA graph. The layer's table of places is on the GPU (`DevPlan`); kernels/moe.cu's `xp_plan` looks the call's choices up in it, gives each miss a place (a clock over the places, as Kimi-K3's
//! dc_plan) where it has pairs enough (`Params`) and copies it there (`xp_fill`, the host tier read through its mapped address), and sends the other misses' pairs and the tokens' rows of z to the host's
//! thread through mapped host memory (`DevMail`: a request, then its number, written last); the products run over the places meanwhile, a miss the host had a short while before is promoted (filled behind
//! the products), `xp_wait` waits on the GPU for the answer's number and `xp_take` brings the host's sums to the experts' sum. The host's thread answers each request as it comes, with the layer's host half
//! (`HostFp4`: the FP4 experts of the host tier, the kernels of src/cpu_fp4.rs). What SGLang's paged experts decide on the GPU (j:paged-experts-sglang:moe:15) and what its CPU experts compute beside the
//! GPU's (j:cpu-experts-sglang:moe:16), in one step: SGLang copies every miss of a captured step, and computes on the host a fixed set of experts.
//!
//! The store keeps its own plan (`LayerCache`) for the calls it plans on the host (a prompt's chunks): before such a call the GPU's table is read into it, after it the places are written back
//! (`ExpertStore::run_early_at`).
//!
//! Calling it (for a step queued ahead of the last one's end, #138, as for a graph):
//! - Enable: `Offload::device` set on the store before its layers are added (`HbCaching::offload`; a development build's `--device-plan`); `ExpertStore::device_on(layer)` says which layers took
//!   it (a host half of FP4 experts, every expert in the computer's memory), and `MoeLayer::reads_host` is false for a layer that did.
//! - A call: `MoeLayer::via_store` (the MoE kind's; `MoeLayer::cached` and Kimi-K3's block call it), which asks `ExpertStore::device_args(layer, tokens, pairs)` and, where it gives `DevArgs`
//!   (1 to `DEV_ROWS` rows), queues on the engine's stream alone: xp_plan, xp_fill (the copies), `early`, the products (`products(1)`), xp_fill (the promotions), xp_wait, xp_take, the
//!   experts' sum. It reads the router's choices and weights (the scratch's `ids`, int64 [t k], and `wts`, fp32 [t k]) and z (bf16 [t, hidden]) on the device; it writes the pairs' places
//!   (`way`, int64 [t k]), the layer's table and counters (`DevArgs::place_of` .. `meta`), the request (mapped host memory) and the host's sums (the scratch's `cpu`, fp32 [t, hidden]), and y.
//!   No host call waits inside it and none reads the routing: the host's thread (`DevMail`) answers the request as the GPU posts it.
//! - Between calls: nothing. Calls of one store follow each other on one stream (its request and answer are one each: two streams' calls at once would share them), so a step may be queued
//!   before the last one has run. There is no event to wait on: the step's own end (the engine's synchronize before it reads the sample) is the only wait.
//! - After a step has run (after that synchronize, never while a step is queued): `ExpertStore::end_step` reads the GPU's counters into the layers' `LayerStats` and writes the plan's next
//!   settings (`Params`, from the costs measured so far; a synchronous write a queued step would race).
//! - A call the host plans (more than `DEV_ROWS` rows: a prompt's chunk) synchronizes the stream itself first (`run_early_at`: the GPU's table read into the host's plan, written back after).

use crate::dev::{Buf, Pinned};
use crate::ffi::{err, Error, Result};
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The words of a request before its pairs (kernels/moe.cu XP_HDR), the most experts and places a layer's table holds (XP_MAX).
pub const XP_HDR: usize = 16;
pub const XP_MAX: usize = 1024;
/// The most pairs a call sends the host (`CPU_TOKENS` rows of up to 32 choices), and the widest row of z (the experts' input) a request carries.
pub const XP_PAIRS: usize = crate::offload::CPU_TOKENS * 32;
pub const XP_WIDTH: usize = 8192;
/// The GPU's clock cycles `xp_wait` waits for the host's answer before it traps (10 s at 4 GHz).
pub const XP_LIMIT: i64 = 40_000_000_000;

/// The most rows a call the GPU plans has: a decode step of up to eight sequences (the one-row step is the graph). A call of more rows is the store's plan on the host, whose waves copy every miss where the
/// misses outnumber the places (`waves_quicker`), which the plan on the GPU does not do.
pub const DEV_ROWS: usize = 8;

/// What `xp_plan` decides a call's misses by (kernels/moe.cu's params): the store's times (`Costs`: a copy, the host call's fixed part, a miss's first row and each further row, microseconds) and mode, as its
/// `plan` decides them (the misses with the most pairs copied, as many as make the longer of the copies and the host's pairs shortest in auto, the two the same in split; all that find a place in copy, none in
/// cpu); a miss the host computed within `window` calls of the layer before that stays the host's is given a place behind the products while the copies leave the host's time room for its copy, `max_promote`
/// a call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub copy: f32,
    pub fixed: f32,
    pub pair: f32,
    pub row: f32,
    pub mode: crate::offload::Mode,
    pub window: i32,
    pub max_promote: i32,
}

impl Params {
    /// The store's settings and times as the plan on the GPU takes them.
    pub fn of(off: &crate::offload::Offload, costs: &crate::offload::Costs) -> Params {
        Params {
            copy: costs.copy as f32,
            fixed: costs.fixed as f32,
            pair: costs.pair as f32,
            row: costs.row as f32,
            mode: off.mode,
            window: off.window.min(i32::MAX as u64) as i32,
            max_promote: off.promote.min(i32::MAX as usize) as i32,
        }
    }

    fn words(&self) -> [u32; 8] {
        use crate::offload::Mode;
        let mode = match self.mode {
            Mode::Copy => 0,
            Mode::Cpu => 1,
            Mode::Split => 2,
            Mode::Auto => 3,
        };
        [self.copy.to_bits(), self.fixed.to_bits(), self.pair.to_bits(), self.row.to_bits(), mode, self.window as u32, self.max_promote as u32, 0]
    }
}

/// What a layer's call on the device path hands kernels/moe.cu's xp_* (device addresses; `mail`, `ans` and `sums` mapped host memory).
#[derive(Clone, Copy, Debug)]
pub struct DevArgs {
    pub place_of: u64,
    pub expert_of: u64,
    pub last: u64,
    pub hostlast: u64,
    pub refb: u64,
    pub meta: u64,
    pub params: u64,
    pub fills: u64,
    pub promos: u64,
    pub reg: u64,
    pub nreg: usize,
    pub mail: u64,
    pub seq: u64,
    pub ans: u64,
    pub sums: u64,
    pub experts: usize,
    pub slots: usize,
    pub layer: usize,
    pub max_pairs: usize,
}

/// A layer's places as the GPU keeps them (kernels/moe.cu xp_plan): place_of [experts] i32, expert_of [slots] i32, last [slots] u32, hostlast [experts] u32, refb [slots] u8 (to 16 bytes), meta [16] u32;
/// the lists of a call's copies and promotions (int2 [slots] each); the regions a copy reads and writes (`nreg` of four u64: host base, an expert's bytes, place base, a place's bytes); the counters (meta's
/// first eight) as last read.
pub struct DevPlan {
    tab: Buf,
    lists: Buf,
    reg: Buf,
    nreg: usize,
    experts: usize,
    slots: usize,
    seen: [u32; 8],
}

impl DevPlan {
    /// A table of `slots` places, every one free, for `experts` experts whose bytes are `regions` (host base, an expert's bytes, place base, a place's bytes; multiples of 16).
    pub fn new(experts: usize, slots: usize, regions: &[[u64; 4]]) -> Result<DevPlan> {
        if experts == 0 || slots == 0 || experts > XP_MAX || slots > XP_MAX || slots > experts {
            return err(format!("the plan on the GPU holds 1 to {XP_MAX} experts and places, where the layer has {experts} and {slots}"));
        }
        if regions.is_empty() || regions.iter().flatten().any(|x| x % 16 != 0) || regions.iter().any(|r| r[1] > r[3]) {
            return err(format!("the plan on the GPU copies 16-byte pieces of an expert into a place at least as long: {regions:?}"));
        }
        let o = offsets(experts, slots);
        let tab = Buf::zeroed(o[5] + 64)?;
        tab.write(0, &vec![-1i32; experts + slots])?;
        let lists = Buf::zeroed(16 * slots)?;
        let reg = Buf::new(32 * regions.len())?;
        reg.write(0, &regions.iter().flatten().copied().collect::<Vec<u64>>())?;
        Ok(DevPlan { tab, lists, reg, nreg: regions.len(), experts, slots, seen: [0; 8] })
    }

    /// The places as the GPU has them (`expert_of`: a place's expert, -1 where free); the work queued before is waited for by the caller.
    pub fn download(&self) -> Result<Vec<i32>> {
        let mut e = vec![0i32; self.slots];
        self.tab.read(offsets(self.experts, self.slots)[1], &mut e)?;
        Ok(e)
    }

    /// The places made `expert_of` (as `LayerCache::expert_in` gives them: -1 where free), each unused since the clock passed it; the GPU's work before is done (the caller waited).
    pub fn upload(&self, expert_of: &[i32]) -> Result<()> {
        if expert_of.len() != self.slots {
            return err(format!("{} places for a table of {}", expert_of.len(), self.slots));
        }
        let mut place_of = vec![-1i32; self.experts];
        for (p, &e) in expert_of.iter().enumerate() {
            if e >= 0 && (e as usize) < self.experts {
                place_of[e as usize] = p as i32;
            }
        }
        let o = offsets(self.experts, self.slots);
        self.tab.write(o[0], &place_of)?;
        self.tab.write(o[1], expert_of)?;
        self.tab.write(o[2], &vec![0u32; self.slots])?;
        self.tab.write(o[4], &vec![0u8; self.slots])
    }

    /// The counters since the last read (calls, hand (none), experts asked for, found, copied, the host's experts, its pairs, promoted); the GPU's work before is done.
    pub fn counters(&mut self) -> Result<[u32; 8]> {
        let mut m = [0u32; 8];
        self.tab.read(offsets(self.experts, self.slots)[5], &mut m)?;
        let mut d = [0u32; 8];
        for i in 0..8 {
            d[i] = m[i].wrapping_sub(self.seen[i]);
        }
        d[1] = 0;
        self.seen = m;
        Ok(d)
    }

    pub fn args(&self, mail: &DevMail, layer: usize) -> DevArgs {
        let o = offsets(self.experts, self.slots).map(|x| self.tab.ptr + x as u64);
        DevArgs {
            place_of: o[0],
            expert_of: o[1],
            last: o[2],
            hostlast: o[3],
            refb: o[4],
            meta: o[5],
            params: mail.params.ptr,
            fills: self.lists.ptr,
            promos: self.lists.ptr + 8 * self.slots as u64,
            reg: self.reg.ptr,
            nreg: self.nreg,
            mail: mail.req.ptr as u64,
            seq: mail.seq.ptr,
            ans: mail.ans.ptr as u64,
            sums: mail.ans.ptr as u64 + 64,
            experts: self.experts,
            slots: self.slots,
            layer,
            max_pairs: XP_PAIRS,
        }
    }
}

/// Byte offsets in a table of place_of, expert_of, last, hostlast, refb and meta.
fn offsets(experts: usize, slots: usize) -> [usize; 6] {
    let (e, s) = (4 * experts, 4 * slots);
    let refb = 2 * e + 2 * s;
    [0, e, e + s, e + 2 * s, refb, refb + slots.next_multiple_of(16)]
}

/// A layer's FP4 experts as the host's thread reads them: each matrix's codes and scales (host tier address, an expert's bytes), the widths, the layout and the activation (src/offload.rs's `HostHalf::Fp4`).
/// The addresses are the host tier's, which lives as long as the store; the store stops the thread before it frees it.
#[derive(Clone, Copy, Debug)]
pub struct HostFp4 {
    pub up: [(usize, usize); 2],
    pub down: [(usize, usize); 2],
    pub inter: usize,
    pub latent: usize,
    pub packed: bool,
    pub act: crate::cpu_fp4::Act,
    pub experts: usize,
}

impl HostFp4 {
    /// Expert `e` (below `experts`) as src/cpu_fp4.rs reads it.
    ///
    /// # Safety
    /// The addresses hold `experts` experts' bytes end to end, not written while the view lives.
    unsafe fn expert(&self, e: usize) -> crate::cpu_fp4::Fp4Expert<'static> {
        let view = |m: &[(usize, usize); 2], rows: usize, cols: usize| crate::cpu_fp4::Fp4View {
            data: std::slice::from_raw_parts((m[0].0 as *const u8).add(e * m[0].1), m[0].1),
            blocks: std::slice::from_raw_parts((m[1].0 as *const u8).add(e * m[1].1), m[1].1),
            rows,
            cols,
            packed: self.packed,
        };
        crate::cpu_fp4::Fp4Expert { up: view(&self.up, 2 * self.inter, self.latent), down: view(&self.down, self.latent, self.inter) }
    }
}

/// The host's side of the device path, one for a store: the request (mapped host memory: `XP_HDR` words, `XP_PAIRS` pairs of three, then z, bf16 [CPU_TOKENS, XP_WIDTH]), the answer (mapped: its number,
/// whether it failed, then the sums, fp32 [CPU_TOKENS, XP_WIDTH] from byte 64), the last request's number (on the GPU), the plan's `Params` (on the GPU), and the host's thread with the layers it computes
/// and the time it took for each since `host_ms` was last asked.
pub struct DevMail {
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    req: Pinned,
    ans: Pinned,
    seq: Buf,
    params: Buf,
    set: Params,
    layers: Arc<Mutex<Vec<Option<HostFp4>>>>,
    times: Arc<Mutex<Vec<f64>>>,
}

/// The bytes of a request and of an answer.
pub const REQ_BYTES: usize = 4 * (XP_HDR + 3 * XP_PAIRS) + 2 * crate::offload::CPU_TOKENS * XP_WIDTH;
pub const ANS_BYTES: usize = 64 + 4 * crate::offload::CPU_TOKENS * XP_WIDTH;

impl DevMail {
    /// The buffers and the host's thread (`threads` of the crew, `pin`ned as the store's host half is), with `params` to start.
    pub fn new(threads: usize, pin: bool, params: Params) -> Result<DevMail> {
        let (req, ans) = (Pinned::mapped(REQ_BYTES)?, Pinned::mapped(ANS_BYTES)?);
        // SAFETY: the buffers hold REQ_BYTES and ANS_BYTES; nothing else uses them yet.
        unsafe {
            std::ptr::write_bytes(req.ptr as *mut u8, 0, 4 * XP_HDR);
            std::ptr::write_bytes(ans.ptr as *mut u8, 0, 64);
        }
        let (seq, pb) = (Buf::zeroed(4)?, Buf::new(32)?);
        pb.write(0, &params.words())?;
        let (stop, layers, times) = (Arc::new(AtomicBool::new(false)), Arc::new(Mutex::new(Vec::new())), Arc::new(Mutex::new(Vec::new())));
        let (r, a, s, l, t) = (req.ptr as usize, ans.ptr as usize, stop.clone(), layers.clone(), times.clone());
        let worker = std::thread::Builder::new()
            .name("glyd-device-plan-host".into())
            .spawn(move || serve(r, a, &s, &l, &t, threads, pin))
            .map_err(|e| Error(format!("the device plan's host thread: {e}")))?;
        Ok(DevMail { stop, worker: Some(worker), req, ans, seq, params: pb, set: params, layers, times })
    }

    /// Layer `layer`'s experts for the host's thread.
    pub fn add_layer(&self, layer: usize, h: HostFp4) {
        let mut l = self.layers.lock().unwrap_or_else(|e| e.into_inner());
        if l.len() <= layer {
            l.resize(layer + 1, None);
        }
        l[layer] = Some(h);
    }

    /// The plan's settings from now on (written between steps: no graph runs the plan meanwhile).
    pub fn set_params(&mut self, p: Params) -> Result<()> {
        if p != self.set {
            self.params.write(0, &p.words())?;
            self.set = p;
        }
        Ok(())
    }

    /// The host's time for layer `layer`'s pairs since this was last asked, ms.
    pub fn host_ms(&self, layer: usize) -> f64 {
        let mut t = self.times.lock().unwrap_or_else(|e| e.into_inner());
        t.get_mut(layer).map_or(0.0, std::mem::take)
    }
}

impl Drop for DevMail {
    /// The host's thread stopped before the buffers it reads (and the store's host tier, which the store drops after this) go.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// The host's thread: each request answered as it comes (its number moves), looked for without a pause while requests come, every 200 microseconds after 50 ms without one.
fn serve(req: usize, ans: usize, stop: &AtomicBool, layers: &Mutex<Vec<Option<HostFp4>>>, times: &Mutex<Vec<f64>>, threads: usize, pin: bool) {
    let kernel = crate::cpu_experts::checked_kernel();
    let (rq, an) = (req as *const u32, ans as *mut u32);
    let (mut last, mut idle, mut out) = (0u32, Instant::now(), Vec::new());
    while !stop.load(Ordering::Relaxed) {
        // SAFETY: the request's first word, mapped host memory the GPU writes; read as it is.
        let s = unsafe { std::ptr::read_volatile(rq) };
        if s == last {
            if idle.elapsed() > Duration::from_millis(50) {
                std::thread::sleep(Duration::from_micros(200));
            } else {
                std::hint::spin_loop();
            }
            continue;
        }
        fence(Ordering::Acquire);
        (last, idle) = (s, Instant::now());
        let ls: Vec<Option<HostFp4>> = layers.lock().unwrap_or_else(|e| e.into_inner()).clone();
        // SAFETY: the request was written before its number (xp_plan's fences), the answer's buffer is this thread's until it writes the number, and the layers' addresses live while the store does.
        let r = unsafe { answer(rq, (an as *mut u8).add(64) as *mut f32, &ls, kernel, threads, pin, &mut out) };
        if let Ok((layer, ms)) = r {
            let mut t = times.lock().unwrap_or_else(|e| e.into_inner());
            if t.len() <= layer {
                t.resize(layer + 1, 0.0);
            }
            t[layer] += ms;
        } else if let Err(e) = &r {
            eprintln!("the device plan's host thread: request {s}: {e}");
        }
        // SAFETY: the answer's first two words, mapped host memory the GPU reads; the sums before them.
        unsafe {
            std::ptr::write_volatile(an.add(1), r.is_err() as u32);
            fence(Ordering::SeqCst);
            std::ptr::write_volatile(an, s);
        }
    }
}

/// A request's pairs computed: the layer and its tokens, pairs and z's width read from `req`, each pair's product by the layer's FP4 experts (src/cpu_fp4.rs `experts`, an expert's bytes read once for its
/// pairs), a token's pairs added in the order of the request into `sums` (fp32 [tokens, width], zeros where a token has none), as the store's host path sends them up. The layer and the ms taken.
///
/// # Safety
/// `req` holds a request (`REQ_BYTES`), `sums` room for `CPU_TOKENS` x `XP_WIDTH` values, and the layers' addresses their experts.
pub unsafe fn answer(
    req: *const u32,
    sums: *mut f32,
    layers: &[Option<HostFp4>],
    kernel: crate::cpu_experts::Kernel,
    threads: usize,
    pin: bool,
    out: &mut Vec<f32>,
) -> Result<(usize, f64)> {
    let t0 = Instant::now();
    let w = |i: usize| std::ptr::read_volatile(req.add(i)) as usize;
    let (layer, tokens, n, width) = (w(1), w(2), w(3), w(4));
    if tokens == 0 || tokens > crate::offload::CPU_TOKENS || n > XP_PAIRS || width > XP_WIDTH {
        return err(format!("a request of {tokens} tokens, {n} pairs and rows of {width}"));
    }
    let l = layers.get(layer).copied().flatten().ok_or_else(|| Error(format!("a request for layer {layer}, which the host has no experts of")))?;
    if width != l.latent {
        return err(format!("rows of z of {width} for layer {layer}, whose experts read {}", l.latent));
    }
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let (e, token, wt) = (w(XP_HDR + 3 * i), w(XP_HDR + 3 * i + 1), f32::from_bits(w(XP_HDR + 3 * i + 2) as u32));
        if e >= l.experts || token >= tokens {
            return err(format!("pair {i} of layer {layer}: expert {e} of {}, token {token} of {tokens}", l.experts));
        }
        pairs.push(crate::cpu_fp4::Fp4Pair { expert: l.expert(e), token, weight: wt });
    }
    let z = std::slice::from_raw_parts(req.add(XP_HDR + 3 * XP_PAIRS) as *const u16, tokens * width);
    out.clear();
    out.resize(n * width, 0.0);
    crate::cpu_fp4::experts(kernel, z, &pairs, out, l.act, threads, pin)?;
    let s = std::slice::from_raw_parts_mut(sums, tokens * width);
    s.fill(0.0);
    for (row, p) in out.chunks_exact(width).zip(&pairs) {
        s[p.token * width..][..width].iter_mut().zip(row).for_each(|(a, b)| *a += *b);
    }
    Ok((layer, t0.elapsed().as_secs_f64() * 1e3))
}

/// kernels/moe.cu's xp_plan on the host, its decisions the kernel's step for step (the tests' reference): a layer's table and its clock. `call` plans one call: each pair's place (-1: the host's, or no
/// expert), the copies and promotions ((expert, place)) and the host's pairs (their numbers).
pub struct Policy {
    pub place_of: Vec<i32>,
    pub expert_of: Vec<i32>,
    last: Vec<u32>,
    hostlast: Vec<u32>,
    refb: Vec<bool>,
    step: u32,
    hand: usize,
}

/// One call's plan (`Policy::call`).
#[derive(Debug, Default, PartialEq)]
pub struct Planned {
    pub way: Vec<i64>,
    pub fills: Vec<(u16, u32)>,
    pub promos: Vec<(u16, u32)>,
    pub host: Vec<usize>,
    pub reads: usize,
    pub hits: usize,
}

impl Policy {
    pub fn new(experts: usize, slots: usize) -> Policy {
        Policy {
            place_of: vec![-1; experts],
            expert_of: vec![-1; slots],
            last: vec![0; slots],
            hostlast: vec![0; experts],
            refb: vec![false; slots],
            step: 0,
            hand: 0,
        }
    }

    /// The places as `LayerCache::expert_in` gives them (`DevPlan::upload`'s).
    pub fn set(&mut self, expert_of: &[i32]) {
        self.place_of.iter_mut().for_each(|p| *p = -1);
        for (p, &e) in expert_of.iter().enumerate() {
            self.expert_of[p] = e;
            if e >= 0 {
                self.place_of[e as usize] = p as i32;
            }
        }
        self.last.iter_mut().for_each(|x| *x = 0);
        self.refb.iter_mut().for_each(|x| *x = false);
    }

    fn place(&mut self, e: usize) -> Option<usize> {
        let slots = self.expert_of.len();
        for _ in 0..2 * slots + 2 {
            let c = self.hand;
            self.hand = (self.hand + 1) % slots;
            if self.last[c] == self.step {
                continue;
            }
            if self.refb[c] {
                self.refb[c] = false;
                continue;
            }
            if let Ok(old) = usize::try_from(self.expert_of[c]) {
                self.place_of[old] = -1;
            }
            (self.expert_of[c], self.place_of[e], self.last[c], self.refb[c]) = (e as i32, c as i32, self.step, true);
            return Some(c);
        }
        None
    }

    /// One call of `ids` planned with `p`, as xp_plan plans it (in f32, as the kernel adds its times).
    pub fn call(&mut self, ids: &[i64], p: &Params) -> Planned {
        use crate::offload::Mode;
        const CMAX: usize = 2048;
        let experts = self.place_of.len();
        self.step += 1;
        let step = self.step;
        let (mut cnt, mut st) = (vec![0usize; experts], vec![0u8; experts]);
        ids.iter().filter(|&&e| e >= 0).for_each(|&e| cnt[e as usize] += 1);
        let mut out = Planned::default();
        for &e in ids.iter().filter(|&&e| e >= 0) {
            let e = e as usize;
            if st[e] != 0 {
                continue;
            }
            out.reads += 1;
            match usize::try_from(self.place_of[e]) {
                Ok(q) => {
                    (st[e], self.last[q], self.refb[q]) = (1, step, true);
                    out.hits += 1;
                }
                Err(_) => st[e] = 4,
            }
        }
        let hot = |hl: &[u32], e: usize| hl[e] > 0 && step - hl[e] <= p.window as u32;
        let (mut nb, mut nhot) = (vec![0usize; CMAX + 1], vec![0usize; CMAX + 1]);
        let (mut nmiss, mut cmax, mut left) = (0usize, 0usize, 0f32);
        for &e in ids.iter().filter(|&&e| e >= 0) {
            let e = e as usize;
            if st[e] != 4 {
                continue;
            }
            st[e] = 5;
            let c = cnt[e].min(CMAX);
            nb[c] += 1;
            nhot[c] += hot(&self.hostlast, e) as usize;
            (nmiss, cmax, left) = (nmiss + 1, cmax.max(c), left + p.pair + p.row * (c as f32 - 1.0));
        }
        let room = self.expert_of.len() - out.hits;
        let (mut cstar, mut mstar) = (CMAX + 1, 0usize);
        let (mut copying, mut computing) = (0f32, if nmiss > 0 { p.fixed + left } else { 0.0 });
        let score = |cpy: f32, cmp: f32| if p.mode == Mode::Split { (cpy - cmp).abs() } else { cpy.max(cmp) };
        if p.mode == Mode::Copy {
            cstar = 0;
        } else if p.mode != Mode::Cpu && nmiss > 0 {
            let (mut best, mut taken, mut kept, mut l) = (score(0.0, computing), 0usize, nmiss, left);
            'levels: for c in (1..=cmax).rev() {
                for j in 0..nb[c] {
                    if taken >= room {
                        break 'levels;
                    }
                    (taken, kept, l) = (taken + 1, kept - 1, l - (p.pair + p.row * (c as f32 - 1.0)));
                    let (cpy, cmp) = (p.copy * taken as f32, if kept > 0 { p.fixed + l } else { 0.0 });
                    let now = score(cpy, cmp);
                    if now < best - 1e-3 {
                        (best, cstar, mstar, copying, computing) = (now, c, j + 1, cpy, cmp);
                    }
                }
            }
        }
        let np_max =
            if p.mode == Mode::Copy || computing <= copying || p.copy <= 0.0 { 0 } else { (p.max_promote as i64).min(((computing - copying) / p.copy) as i64) };
        let mut left_at = if cstar <= CMAX { [mstar.min(nhot[cstar]), mstar - mstar.min(nhot[cstar])] } else { [0, 0] };
        for &e in ids.iter().filter(|&&e| e >= 0) {
            let e = e as usize;
            if st[e] != 5 {
                continue;
            }
            let (c, h) = (cnt[e].min(CMAX), hot(&self.hostlast, e));
            let mut want = p.mode == Mode::Copy || c > cstar;
            if !want && c == cstar && left_at[!h as usize] > 0 {
                left_at[!h as usize] -= 1;
                want = true;
            }
            if let Some(q) = want.then(|| self.place(e)).flatten() {
                st[e] = 2;
                out.fills.push((e as u16, q as u32));
                continue;
            }
            st[e] = 3;
            if h && (out.promos.len() as i64) < np_max {
                if let Some(q) = self.place(e) {
                    out.promos.push((e as u16, q as u32));
                }
            }
            self.hostlast[e] = step;
        }
        for (j, &e) in ids.iter().enumerate() {
            let s = if e >= 0 { st[e as usize] } else { 0 };
            out.way.push(if s == 1 || s == 2 { self.place_of[e as usize] as i64 } else { -1 });
            if s == 3 {
                out.host.push(j);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random FP4 matrix in the packed-scale layout (Kimi-K3's; DeepSeek-V4's releases the other), as src/cpu_fp4.rs reads it.
    fn pack(rows: usize, cols: usize, seed: u64) -> glyd_gpu::pack::Tiered {
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let q: Vec<u8> = (0..rows * cols / 2).map(|_| rnd() as u8).collect();
        let e: Vec<u8> = (0..rows * cols / 32).map(|_| 120 + (rnd() % 4) as u8).collect();
        glyd_gpu::pack::pack_fp4r_checked(&q, &e, rows, cols).expect("a pack")
    }

    /// The request a GPU's plan writes (words; z after the pairs), for `pairs` (expert, token, weight) of `tokens` rows of z.
    fn request(layer: usize, tokens: usize, width: usize, pairs: &[(usize, usize, f32)], z: &[u16]) -> Vec<u32> {
        let mut r = vec![0u32; REQ_BYTES / 4];
        r[..6].copy_from_slice(&[7, layer as u32, tokens as u32, pairs.len() as u32, width as u32, 2]);
        for (i, &(e, t, w)) in pairs.iter().enumerate() {
            r[XP_HDR + 3 * i..][..3].copy_from_slice(&[e as u32, t as u32, w.to_bits()]);
        }
        for (i, c) in z.chunks(2).enumerate() {
            r[XP_HDR + 3 * XP_PAIRS + i] = c[0] as u32 | (c.get(1).copied().unwrap_or(0) as u32) << 16;
        }
        r
    }

    /// A request is answered as the store's host path computes and sends up a call's host pairs: each pair's output by the layer's FP4 experts (read where the host tier holds them, an expert's bytes
    /// once for its pairs), a token's pairs added in the request's order, zeros for a token with none; a request the host cannot answer is refused.
    #[test]
    fn a_request_is_answered_with_the_hosts_sums() {
        let (experts, latent, inter, tokens) = (5usize, 128usize, 64usize, 4usize);
        let ups: Vec<_> = (0..experts).map(|e| pack(2 * inter, latent, 11 + e as u64)).collect();
        let downs: Vec<_> = (0..experts).map(|e| pack(latent, inter, 31 + e as u64)).collect();
        let cat =
            |v: &[glyd_gpu::pack::Tiered], blocks: bool| v.iter().flat_map(|p| if blocks { p.blocks.clone() } else { p.data.clone() }).collect::<Vec<u8>>();
        let (ud, ub, dd, db) = (cat(&ups, false), cat(&ups, true), cat(&downs, false), cat(&downs, true));
        let act = crate::cpu_fp4::Act::Situ { beta: 4.0, linear: 25.0 };
        let h = HostFp4 {
            up: [(ud.as_ptr() as usize, ups[0].data.len()), (ub.as_ptr() as usize, ups[0].blocks.len())],
            down: [(dd.as_ptr() as usize, downs[0].data.len()), (db.as_ptr() as usize, downs[0].blocks.len())],
            inter,
            latent,
            packed: true,
            act,
            experts,
        };
        let z: Vec<u16> = (0..tokens * latent).map(|i| (crate::cpu_fp4::bf16(((i * 13 % 29) as f32 - 14.0) / 64.0).to_bits() >> 16) as u16).collect();
        let pairs = [(2usize, 0usize, 0.5f32), (4, 1, 0.25), (2, 2, 0.125), (0, 2, 1.0), (3, 0, 0.75)];
        let req = request(1, tokens, latent, &pairs, &z);
        let layers = vec![None, Some(h)];
        let (mut sums, mut out) = (vec![f32::NAN; crate::offload::CPU_TOKENS * XP_WIDTH], Vec::new());
        let k = crate::cpu_experts::Kernel::Scalar;
        // SAFETY: the request, the sums' room and the experts' bytes are the buffers above.
        let (layer, _) = unsafe { answer(req.as_ptr(), sums.as_mut_ptr(), &layers, k, 2, false, &mut out) }.unwrap();
        assert_eq!(layer, 1);
        // (the store's host path: the pairs through src/cpu_fp4.rs, then each into its token's row, in order)
        // SAFETY: as above.
        let fp: Vec<_> = pairs.iter().map(|&(e, token, weight)| crate::cpu_fp4::Fp4Pair { expert: unsafe { h.expert(e) }, token, weight }).collect();
        let mut want_out = vec![0f32; pairs.len() * latent];
        crate::cpu_fp4::experts(k, &z, &fp, &mut want_out, act, 1, false).unwrap();
        let mut want = vec![0f32; tokens * latent];
        for (row, p) in want_out.chunks_exact(latent).zip(&pairs) {
            want[p.1 * latent..][..latent].iter_mut().zip(row).for_each(|(a, b)| *a += *b);
        }
        assert_eq!(sums[..tokens * latent].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert!(sums[3 * latent..4 * latent].iter().all(|&x| x == 0.0), "a token with no pair: zeros");
        // (refused: a layer the host has no experts of, rows of another width, an expert past the layer's)
        for (layer, width, pairs) in [(0, latent, &pairs[..]), (1, latent / 2, &pairs[..]), (1, latent, &[(9usize, 0usize, 1.0f32)][..])] {
            let r = request(layer, tokens, width, pairs, &z);
            // SAFETY: as above.
            assert!(unsafe { answer(r.as_ptr(), sums.as_mut_ptr(), &layers, k, 1, false, &mut out) }.is_err());
        }
    }

    /// The plan's settings are the store's times and mode as the kernel reads them: four floats, the mode, the window, the most promotions.
    #[test]
    fn the_settings_from_the_mode_and_the_costs() {
        use crate::offload::{Costs, Mode, Offload};
        let c = Costs { copy: 500.0, fixed: 100.0, pair: 200.0, row: 25.0, backlog: 0.0 };
        let p = Params::of(&Offload { mode: Mode::Split, window: 8, promote: 3, ..Offload::default() }, &c);
        assert_eq!(p.words(), [500f32.to_bits(), 100f32.to_bits(), 200f32.to_bits(), 25f32.to_bits(), 2, 8, 3, 0]);
        assert_eq!(Params::of(&Offload::default(), &c).words()[4], 3, "auto");
    }

    /// Over calls of random choices in every mode, the plan keeps its table whole and decides what the store's `plan` decides: each pair's place holds its expert, the host's pairs are the ones with
    /// no place, copies and promotions take places no choice of the call found or took; in auto and split the misses copied are the ones `offload::plan` copies (by their pairs and whether hot), in copy
    /// mode every miss the places hold, in cpu mode none; a promotion is of a hot miss the host keeps, at most the most a call.
    #[test]
    fn the_plan_decides_as_the_stores_plan() {
        use crate::offload::{Costs, Miss, Mode};
        let (experts, slots) = (40usize, 9usize);
        let costs = Costs { copy: 600.0, fixed: 100.0, pair: 400.0, row: 30.0, backlog: 0.0 };
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for mode in [Mode::Auto, Mode::Split, Mode::Copy, Mode::Cpu] {
            let p = Params { copy: 600.0, fixed: 100.0, pair: 400.0, row: 30.0, mode, window: 4, max_promote: 2 };
            let mut pl = Policy::new(experts, slots);
            let mut hostlast = vec![0u32; experts];
            let (mut copied, mut hosted) = (0usize, 0usize);
            for step in 1..=400u32 {
                let tokens = 1 + next(6);
                let ids: Vec<i64> = (0..tokens * 2).map(|_| next(experts) as i64).collect();
                let before = pl.place_of.clone();
                let got = pl.call(&ids, &p);
                // (the store's plan of the same misses: distinct, in the order of the choices, with their pairs and whether hot)
                let mut misses: Vec<Miss> = Vec::new();
                for &e in &ids {
                    let e = e as usize;
                    if before[e] < 0 && !misses.iter().any(|m| m.expert as usize == e) {
                        let hot = hostlast[e] > 0 && step - hostlast[e] <= p.window as u32;
                        misses.push(Miss { expert: e as u16, pairs: ids.iter().filter(|&&i| i as usize == e).count(), hot });
                    }
                }
                let room = slots - got.hits;
                let mut want: Vec<(usize, bool)> = match mode {
                    Mode::Auto | Mode::Split => {
                        crate::offload::plan(mode, &misses, room, &costs).iter().zip(&misses).filter(|(c, _)| **c).map(|(_, m)| (m.pairs, m.hot)).collect()
                    }
                    Mode::Copy => misses.iter().take(room).map(|m| (m.pairs, m.hot)).collect(),
                    Mode::Cpu => Vec::new(),
                };
                let mut have: Vec<(usize, bool)> =
                    got.fills.iter().map(|&(e, _)| misses.iter().find(|m| m.expert == e).map(|m| (m.pairs, m.hot)).expect("a copy of a miss")).collect();
                (want.sort(), have.sort());
                assert_eq!(have, want, "{mode:?} step {step}: the misses copied");
                let mut taken = std::collections::HashSet::new();
                for (j, &w) in got.way.iter().enumerate() {
                    if w >= 0 {
                        assert_eq!(pl.expert_of[w as usize], ids[j] as i32, "{mode:?} step {step}: pair {j}'s place holds its expert");
                        taken.insert(w);
                    } else {
                        assert!(got.host.contains(&j), "{mode:?} step {step}: a pair with no place is the host's");
                    }
                }
                assert!(got.promos.len() <= p.max_promote as usize);
                for &(e, q) in &got.promos {
                    assert!(!taken.contains(&(q as i64)), "{mode:?} step {step}: a promotion takes no place the call uses");
                    assert!(
                        misses.iter().any(|m| m.expert == e && m.hot) && got.host.iter().any(|&j| ids[j] as u16 == e),
                        "{mode:?} step {step}: a promotion of a hot miss the host keeps"
                    );
                }
                for &j in &got.host {
                    hostlast[ids[j] as usize] = step;
                }
                for (q, &e) in pl.expert_of.iter().enumerate() {
                    assert!(e < 0 || pl.place_of[e as usize] == q as i32, "{mode:?} step {step}: the table is whole");
                }
                (copied, hosted) = (copied + got.fills.len(), hosted + got.host.len());
            }
            eprintln!("{mode:?}: {copied} copies, {hosted} host pairs over 400 calls");
            assert!(mode == Mode::Cpu || copied > 0, "{mode:?} copies some misses");
            assert!(mode == Mode::Copy || hosted > 0, "{mode:?} leaves some to the host");
        }
    }
}
