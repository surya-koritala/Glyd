//! Kimi-K3's expert cache with every layer across the GPUs (src/dynamic/tp.rs). The experts are in the computer's memory once, packed by the first GPU's
//! thread in portable pinned memory (each GPU copies from it at its link's rate); a layer's places are the same on every GPU, each holding that GPU's slice of
//! the rows of the expert in it. The first GPU's thread plans every call by src/offload.rs's policy (`serve`, `LayerCache`) and posts the plan; every GPU
//! copies its slices of the experts the plan copies, over its own link, and runs its products; the host's pairs are computed once, on the first GPU's thread,
//! and every GPU adds their sums. A call computes what the expert cache on one GPU computes, slice by slice.

use crate::dev::{copy_d2h_async, copy_h2d_async, memset_async, Buf, Ev, Pinned, Side};
use crate::ffi::{err, Error, Result, Stream};
use crate::offload::{distinct, fill_way, serve, Costs, LayerCache, Offload, CPU_TOKENS};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Host memory every thread may read: pinned once, written before it is shared, never after.
pub struct HostBuf(pub Pinned);
// SAFETY: the bytes are written by one thread before the buffer is shared and only read after (the copies and the host's experts).
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

/// A layer's experts in the computer's memory: each expert's first matrix (gate and up) and second, codes and scales, end to end at their strides.
pub struct HostLayer {
    pub mats: [[HostBuf; 2]; 2],
    pub stride: [[u64; 2]; 2],
    pub packed: bool,
    pub inter: usize,
    pub latent: usize,
    pub act: (f32, f32),
}

impl HostLayer {
    fn view(&self, e: usize) -> crate::cpu_fp4::Fp4Expert<'_> {
        let v = |m: usize, rows: usize, cols: usize| {
            let (d, s) = (self.stride[m][0] as usize, self.stride[m][1] as usize);
            // SAFETY: every expert of the layer end to end (checked where the layer was made), never written after.
            let (data, blocks) = unsafe {
                (
                    std::slice::from_raw_parts((self.mats[m][0].0.ptr as *const u8).add(e * d), d),
                    std::slice::from_raw_parts((self.mats[m][1].0.ptr as *const u8).add(e * s), s),
                )
            };
            crate::cpu_fp4::Fp4View { data, blocks, rows, cols, packed: self.packed }
        };
        crate::cpu_fp4::Fp4Expert { up: v(0, 2 * self.inter, self.latent), down: v(1, self.latent, self.inter) }
    }
}

/// One wave of a call: the experts copied into places before its products (expert, place), and each pair's place (-1: none).
struct Wave {
    loads: Vec<(u16, u32)>,
    way: Vec<i64>,
}

/// The host's sums of a call, [tokens, latent] fp32, once the first GPU's thread has made them.
#[derive(Default)]
struct Sums {
    got: Mutex<Option<std::result::Result<Arc<Vec<f32>>, String>>>,
    cv: Condvar,
}

/// A call's plan, every GPU's: its waves, the experts copied behind it, the host's pairs (their numbers among the call's) and their sums.
struct Plan {
    waves: Vec<Wave>,
    promote: Vec<(u16, u32)>,
    mine: Vec<usize>,
    sums: Option<Arc<Sums>>,
}

/// The plans posted, by (layer, call), with the GPUs yet to take each.
type Mail = HashMap<(usize, u64), (Arc<Plan>, usize)>;

/// What a group's GPUs share: the layers' host tiers and the plans posted for each call.
#[derive(Default)]
pub struct TpCache {
    host: Mutex<HashMap<usize, Arc<HostLayer>>>,
    host_cv: Condvar,
    mail: Mutex<Mail>,
    mail_cv: Condvar,
}

/// How long a GPU's thread waits for the first one's host tier or plan before it takes the group for broken (a load of Kimi-K3 packs a layer in seconds).
const WAIT: Duration = Duration::from_secs(600);

fn poisoned() -> Error {
    Error("the GPUs' expert cache is poisoned".into())
}

impl TpCache {
    /// Layer `layer`'s host tier, the first GPU's.
    pub fn put_host(&self, layer: usize, h: Arc<HostLayer>) -> Result<()> {
        self.host.lock().map_err(|_| poisoned())?.insert(layer, h);
        self.host_cv.notify_all();
        Ok(())
    }

    /// Layer `layer`'s host tier, once the first GPU's thread has made it.
    pub fn host(&self, layer: usize) -> Result<Arc<HostLayer>> {
        let t0 = Instant::now();
        let mut g = self.host.lock().map_err(|_| poisoned())?;
        loop {
            if let Some(h) = g.get(&layer) {
                return Ok(h.clone());
            }
            if t0.elapsed() > WAIT {
                return err(format!("layer {layer}'s experts were not packed by the first GPU's thread in {} s", WAIT.as_secs()));
            }
            g = self.host_cv.wait_timeout(g, Duration::from_secs(1)).map_err(|_| poisoned())?.0;
        }
    }

    fn post(&self, key: (usize, u64), p: Arc<Plan>, readers: usize) -> Result<()> {
        if readers > 0 {
            self.mail.lock().map_err(|_| poisoned())?.insert(key, (p, readers));
            self.mail_cv.notify_all();
        }
        Ok(())
    }

    fn take(&self, key: (usize, u64)) -> Result<Arc<Plan>> {
        let t0 = Instant::now();
        let mut g = self.mail.lock().map_err(|_| poisoned())?;
        loop {
            if let Some((p, left)) = g.get_mut(&key) {
                let p = p.clone();
                *left -= 1;
                if *left == 0 {
                    g.remove(&key);
                }
                return Ok(p);
            }
            if t0.elapsed() > WAIT {
                return err(format!("no plan for call {} of layer {} from the first GPU's thread in {} s", key.1, key.0, WAIT.as_secs()));
            }
            g = self.mail_cv.wait_timeout(g, Duration::from_secs(1)).map_err(|_| poisoned())?.0;
        }
    }
}

/// The copy stream's backlog, in microseconds of copying, past which a call copies nothing behind it (src/offload.rs's).
const BACKLOG_US: f64 = 10_000.0;

/// A layer's policy, the first GPU's: its cache, the call each expert was last left to the host in, its calls.
struct Policy {
    cache: LayerCache,
    last: Vec<u64>,
    calls: u64,
}

/// What the first GPU's thread plans with: each layer's policy, how a miss is served, the times, the copies still running on its stream (start, end,
/// bytes), and the counts since the last report.
struct Leader {
    layers: HashMap<usize, Policy>,
    off: Offload,
    costs: Costs,
    kernel: crate::cpu_experts::Kernel,
    batches: VecDeque<(Ev, Ev, u64)>,
    spare: Vec<Ev>,
    stats: [u64; 6],
    cpu_ms: f64,
}

/// A GPU's share of the cache, every layer's: its copy stream, its pinned buffers (the router's choices, their weights and the experts' input read back on the
/// first GPU; the pairs' places and the host's sums sent up on each) and the event after the last of those uploads, the first GPU's planning.
pub struct Rank {
    cache: Arc<TpCache>,
    rank: usize,
    n: usize,
    copy: Side,
    ids: Pinned,
    wts: Pinned,
    z: Pinned,
    way: Pinned,
    sums: Pinned,
    sent: Ev,
    got: Ev,
    leader: Option<RefCell<Leader>>,
    pub expert_bytes: u64,
    /// This GPU's thread's time in a call's parts (us, since the last report: waiting for the router's choices, planning and posting the plan or taking it,
    /// queuing the copies and products, the host's pairs, the whole call) and the calls: docs/ceiling.md's per-layer terms.
    spans: Cell<[f64; 5]>,
    calls: Cell<u64>,
    /// A development build's GLYD_KIMI_TRACE=FILE: the first GPU's line a call, the layer, the tokens and the router's choices (`replay_bench` reads it).
    trace: Option<RefCell<std::io::LineWriter<std::fs::File>>>,
    /// GLYD_KIMI_STATS on the device path (the first GPU's): each layer's counts (choices, found in a place, copied in) when last read, their sum over
    /// the layers since, and the layer a pass starts at.
    dev_stats: Option<(RefCell<HashMap<usize, [u32; 3]>>, Cell<[u64; 3]>, Cell<Option<usize>>)>,
}

impl Rank {
    /// GPU `rank` of `n`'s share, for calls of up to `chunk` tokens of `k` pairs each and experts of `latent` outputs; the first GPU plans with `off`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(cache: Arc<TpCache>, rank: usize, n: usize, chunk: usize, k: usize, latent: usize, off: Offload, expert_bytes: u64) -> Result<Rank> {
        let pairs = chunk * k;
        let few = chunk.min(CPU_TOKENS);
        let leader = (rank == 0).then(|| {
            RefCell::new(Leader {
                layers: HashMap::new(),
                off,
                costs: Costs::default(),
                kernel: crate::cpu_experts::checked_kernel(),
                batches: VecDeque::new(),
                spare: Vec::new(),
                stats: [0; 6],
                cpu_ms: 0.0,
            })
        });
        let trace = match crate::tune!("GLYD_KIMI_TRACE").filter(|_| rank == 0) {
            Some(f) => Some(RefCell::new(std::io::LineWriter::new(std::fs::File::create(&f).map_err(|e| Error(format!("{f}: {e}")))?))),
            None => None,
        };
        let dev_stats = (rank == 0 && crate::tune!("GLYD_KIMI_STATS").is_some()).then(|| (RefCell::new(HashMap::new()), Cell::new([0; 3]), Cell::new(None)));
        Ok(Rank {
            cache,
            rank,
            n,
            copy: Side::new()?,
            ids: Pinned::new(pairs * 8)?,
            wts: Pinned::new(pairs * 4)?,
            z: Pinned::new(few * latent * 2)?,
            way: Pinned::new(pairs * 8)?,
            sums: Pinned::new(few * latent * 4)?,
            sent: Ev::new()?,
            got: Ev::new()?,
            leader,
            expert_bytes,
            spans: Cell::new([0.0; 5]),
            calls: Cell::new(0),
            trace,
            dev_stats,
        })
    }

    /// The counts since the last call of it, in a line (the first GPU's), and them reset.
    pub fn report(&self) -> String {
        let Some(ld) = &self.leader else { return String::new() };
        let mut l = ld.borrow_mut();
        let [reads, hits, copied, cpu, promoted, waves] = std::mem::take(&mut l.stats);
        let ms = std::mem::take(&mut l.cpu_ms);
        let (sp, calls) = (self.spans.replace([0.0; 5]), self.calls.replace(0).max(1) as f64);
        format!("expert cache across {} GPUs: {reads} reads, {hits} in places ({:.1}%), {copied} copied, {promoted} behind calls, {cpu} pairs on the host ({ms:.0} ms), {waves} waves past the first; a copy {:.0} us, a host pair {:.0} us; the first GPU's thread a call ({calls:.0} calls): the choices {:.0} us, the plan {:.0}, copies and products queued {:.0}, the host's pairs {:.0}, the call {:.0}", self.n, 100.0 * hits as f64 / reads.max(1) as f64, l.costs.copy, l.costs.pair, sp[0] / calls, sp[1] / calls, sp[2] / calls, sp[3] / calls, sp[4] / calls)
    }
}

impl Leader {
    /// Layer `layer`'s places as `expert_in` has them (the device path's table read back): its cache's places and order of use set from them.
    fn set_places(&mut self, layer: usize, experts: usize, slots: usize, expert_in: &[i32]) {
        let p = self.layers.entry(layer).or_insert_with(|| Policy { cache: LayerCache::new(experts, slots), last: vec![0; experts], calls: 0 });
        p.cache.set_places(expert_in);
    }

    /// The copies finished since the last plan counted into the times; the copying still queued, in microseconds.
    fn backlog(&mut self, expert: u64) -> Result<f64> {
        while let Some((a, b, bytes)) = self.batches.front() {
            if !b.done()? {
                break;
            }
            let ms = b.since(a)? as f64;
            self.costs.copied(*bytes, ms, expert);
            let (a, b, _) = self.batches.pop_front().expect("a batch");
            self.spare.extend([a, b]);
        }
        Ok(self.costs.queued(self.batches.iter().map(|x| x.2).sum(), expert))
    }

    /// A call of layer `layer` (`experts` experts in `slots` places) for the router's choices `ids` (k a token) and their weights: on the host's terms where
    /// `hybrid` (one wave: the misses copied where the plan says, the others the host's), else every expert copied, in waves of as many as the places hold.
    #[allow(clippy::too_many_arguments)]
    fn plan(&mut self, layer: usize, ids: &[i64], hybrid: bool, experts: usize, slots: usize, expert: u64) -> Result<Plan> {
        let backlog = self.backlog(expert)?;
        let p = self.layers.entry(layer).or_insert_with(|| Policy { cache: LayerCache::new(experts, slots), last: vec![0; experts], calls: 0 });
        p.calls += 1;
        let needed = distinct(ids, experts);
        let pairs = ids.len();
        self.stats[0] += needed.len() as u64;
        if hybrid {
            let mut pairs_of = vec![0usize; experts];
            ids.iter().filter(|&&e| e >= 0 && (e as usize) < experts).for_each(|&e| pairs_of[e as usize] += 1);
            let costs = Costs { backlog, ..self.costs };
            let sv = serve(&mut p.cache, &needed, &pairs_of, &mut p.last, p.calls, &self.off, &costs)?;
            let mut place = vec![-1i64; experts];
            for &e in &sv.wanted {
                place[e as usize] = p.cache.slot_of(e as usize).ok_or_else(|| Error(format!("expert {e} has no place")))? as i64;
            }
            let mut way = vec![-1i64; pairs];
            fill_way(ids, &place, &mut way);
            let mine: Vec<usize> = (0..pairs).filter(|&j| way[j] < 0 && ids[j] >= 0 && (ids[j] as usize) < experts).collect();
            let promote = if !sv.promote.is_empty() && backlog <= BACKLOG_US { p.cache.hint(&sv.promote)?.loads } else { Vec::new() };
            self.stats[1] += sv.hits as u64;
            self.stats[2] += sv.loads.len() as u64;
            self.stats[3] += mine.len() as u64;
            self.stats[4] += promote.len() as u64;
            let sums = (!mine.is_empty()).then(|| Arc::new(Sums::default()));
            return Ok(Plan { waves: vec![Wave { loads: sv.loads, way }], promote, mine, sums });
        }
        let mut waves = Vec::new();
        for wave in p.cache.waves(&needed) {
            let got = p.cache.request(&wave)?;
            self.stats[1] += got.hits as u64;
            self.stats[2] += got.loads.len() as u64;
            let mut place = vec![-1i64; experts];
            for &e in &wave {
                place[e as usize] = p.cache.slot_of(e as usize).ok_or_else(|| Error(format!("expert {e} has no place")))? as i64;
            }
            let mut way = vec![-1i64; pairs];
            fill_way(ids, &place, &mut way);
            waves.push(Wave { loads: got.loads, way });
        }
        self.stats[5] += waves.len().saturating_sub(1) as u64;
        Ok(Plan { waves, promote: Vec::new(), mine: Vec::new(), sums: None })
    }

    /// The host's pairs `mine` of a call (k a token, `tokens` rows of z) by the layer's host tier: their weighted outputs added up by token, in the pairs' order.
    #[allow(clippy::too_many_arguments)]
    fn compute(&mut self, host: &HostLayer, z: &[u16], ids: &[i64], wts: &[f32], mine: &[usize], k: usize, tokens: usize) -> Result<Vec<f32>> {
        let latent = host.latent;
        let pairs: Vec<crate::cpu_fp4::Fp4Pair> =
            mine.iter().map(|&j| crate::cpu_fp4::Fp4Pair { expert: host.view(ids[j] as usize), token: j / k, weight: wts[j] }).collect();
        let mut out = vec![0f32; mine.len() * latent];
        let t0 = Instant::now();
        crate::cpu_fp4::experts(
            self.kernel,
            z,
            &pairs,
            &mut out,
            crate::cpu_fp4::Act::Situ { beta: host.act.0, linear: host.act.1 },
            self.off.threads,
            self.off.pin,
        )?;
        let us = t0.elapsed().as_secs_f64() * 1e6;
        self.costs.computed(crate::cpu_fp4::by_expert(&pairs).len(), mine.len(), us);
        self.cpu_ms += us / 1e3;
        let mut sums = vec![0f32; tokens * latent];
        for (row, &j) in out.chunks_exact(latent).zip(mine) {
            sums[(j / k) * latent..][..latent].iter_mut().zip(row).for_each(|(a, b)| *a += *b);
        }
        Ok(sums)
    }
}

/// A layer's places on this GPU: their codes and scales ([first, second matrix]), this GPU's bytes of an expert's in each, the events after its last copies
/// (`ready`) and after the last products that read the places (`done`), and its calls so far.
pub struct Places {
    pub base: [[u64; 2]; 2],
    pub slice: [[u64; 2]; 2],
    pub experts: usize,
    pub slots: usize,
    ready: Ev,
    done: Ev,
    calls: Cell<u64>,
    /// The device path's table of the places (`attach_device`): decode steps planned on the GPU; the calls the first GPU's thread plans read it back
    /// before and write their copies to it after.
    pub dev: Option<DevTable>,
}

impl Places {
    pub fn new(base: [[u64; 2]; 2], slice: [[u64; 2]; 2], experts: usize, slots: usize) -> Result<Places> {
        Ok(Places { base, slice, experts, slots, ready: Ev::new()?, done: Ev::new()?, calls: Cell::new(0), dev: None })
    }

    /// The device path's table for these places, every place free, this GPU (`rank`) copying its slices from `host`.
    pub fn attach_device(&mut self, host: &HostLayer, rank: usize) -> Result<()> {
        self.dev = Some(DevTable::new(self, host, rank)?);
        Ok(())
    }
}

/// kernels/kimi.cu's DC_MAX: the most experts and places a layer's table holds.
const DC_MAX: usize = 1024;

/// A layer's places on this GPU as kernels/kimi.cu's dc_plan keeps them (place_of [experts] i32, expert_of [slots] i32, last [slots] u32, the clock's
/// marks [slots] u8, its counters [8] u32), the four regions dc_fill copies (host base, bytes an expert, this GPU's offset in one, device base, bytes a
/// place), and the host's mirror of the first two.
pub struct DevTable {
    tab: Buf,
    pub reg: Buf,
    pub experts: usize,
    pub slots: usize,
    mirror: RefCell<(Vec<i32>, Vec<i32>)>,
}

impl DevTable {
    fn new(pl: &Places, host: &HostLayer, rank: usize) -> Result<DevTable> {
        let (experts, slots) = (pl.experts, pl.slots);
        if experts > DC_MAX || slots > DC_MAX || slots == 0 {
            return err(format!("the device path's table holds {DC_MAX} experts and places, where the layer has {experts} and {slots}"));
        }
        let tab = Buf::zeroed(8 * experts + 8 * slots + slots.next_multiple_of(16) + 32)?;
        tab.write(0, &vec![-1i32; experts + slots])?;
        let mut r = Vec::with_capacity(20);
        for m in 0..2 {
            for k in 0..2 {
                let sl = pl.slice[m][k];
                r.extend([host.mats[m][k].0.ptr as u64, host.stride[m][k], rank as u64 * sl, pl.base[m][k], sl]);
            }
        }
        if r.iter().any(|x| x % 16 != 0) {
            return err(format!("the device path's copies take 16-byte pieces: {r:?}"));
        }
        let reg = Buf::new(20 * 8)?;
        reg.write(0, &r)?;
        Ok(DevTable { tab, reg, experts, slots, mirror: RefCell::new((vec![-1; experts], vec![-1; slots])) })
    }

    /// dc_plan's arguments: place_of, expert_of, last, the marks, the counters.
    pub fn at(&self) -> (u64, u64, u64, u64, u64) {
        let p = self.tab.ptr;
        let e = p + 4 * self.experts as u64;
        let l = e + 4 * self.slots as u64;
        let m = l + 4 * self.slots as u64;
        (p, e, l, m, m + self.slots.next_multiple_of(16) as u64)
    }

    /// The places as the GPU has them, into the mirror (the stream's work before it waited for by the caller).
    fn download(&self) -> Result<()> {
        let mut g = self.mirror.borrow_mut();
        let (place_of, expert_of) = &mut *g;
        self.tab.read(0, place_of)?;
        self.tab.read(4 * self.experts, expert_of)
    }

    /// A plan's copies (expert, place) into the mirror, and the mirror up to the GPU.
    fn apply(&self, loads: &[(u16, u32)]) -> Result<()> {
        let mut g = self.mirror.borrow_mut();
        let (place_of, expert_of) = &mut *g;
        for &(e, p) in loads {
            let (e, p) = (e as usize, p as usize);
            if e >= self.experts || p >= self.slots {
                return err(format!("a copy of expert {e} into place {p} of {}", self.slots));
            }
            let old = expert_of[p];
            if old >= 0 {
                place_of[old as usize] = -1;
            }
            expert_of[p] = e as i32;
            place_of[e] = p as i32;
        }
        self.tab.write(0, place_of)?;
        self.tab.write(4 * self.experts, expert_of)
    }

    /// The counters: reads, hits, copies (the GPU's work before waited for by the caller).
    pub fn counters(&self) -> Result<[u32; 3]> {
        let mut m = [0u32; 8];
        self.tab.read((self.at().4 - self.tab.ptr) as usize, &mut m)?;
        Ok([m[2], m[3], m[4]])
    }
}

/// What a layer's call hands the cache: the router's choices and weights (int64 and fp32 [pairs] on the device), the experts' input z (bf16 [tokens, latent]),
/// the device buffers the pairs' places go to (int64) and the host's sums (fp32), the tokens and pairs.
pub struct Call {
    pub ids: u64,
    pub wts: u64,
    pub z: u64,
    pub way: u64,
    pub part: u64,
    pub tokens: usize,
    pub pairs: usize,
}

/// The stages a call gives its layer in turn on the engine's stream: a wave's products (its number, of how many; whether any pair has a place) and the sum
/// with the host's sums (0: none).
pub enum Stage {
    Products(usize, usize, bool),
    Sum(u64),
}

impl Rank {
    /// The copies of `loads` (this GPU's slices) into layer `pl`'s places on the copy stream, after the products that read the places, `ready` after them
    /// (the layer's products wait for it: a call's own copies, or the last call's behind it, whose experts this call may find in places). The first GPU
    /// times them.
    ///
    /// # Safety
    /// The places are live device buffers of `slots` experts' slices; `host` holds every expert.
    unsafe fn copy(&self, pl: &Places, host: &HostLayer, loads: &[(u16, u32)]) -> Result<()> {
        if loads.is_empty() {
            return Ok(());
        }
        let cs = self.copy.stream;
        pl.done.wait(cs)?;
        let start = match &self.leader {
            Some(ld) => {
                let e = ld.borrow_mut().spare.pop().map_or_else(Ev::new, Ok)?;
                e.record(cs)?;
                Some(e)
            }
            None => None,
        };
        for &(e, place) in loads {
            if e as usize >= pl.experts || place as usize >= pl.slots {
                return err(format!("expert {e} into place {place} of {} for {}", pl.slots, pl.experts));
            }
            for m in 0..2 {
                for part in 0..2 {
                    let (whole, sl) = (host.stride[m][part], pl.slice[m][part]);
                    let src = (host.mats[m][part].0.ptr as *const u8).add((e as u64 * whole + self.rank as u64 * sl) as usize);
                    copy_h2d_async(pl.base[m][part] + place as u64 * sl, src as *const _, sl as usize, cs)?;
                }
            }
        }
        if let (Some(ld), Some(a)) = (&self.leader, start) {
            let mut l = ld.borrow_mut();
            let b = l.spare.pop().map_or_else(Ev::new, Ok)?;
            b.record(cs)?;
            l.batches.push_back((a, b, loads.len() as u64 * self.expert_bytes));
        }
        pl.ready.record(cs)
    }

    /// One call of the layer `index` whose places are `pl`: the router's choices read on the first GPU and planned there, then on every GPU the plan's copies,
    /// the pairs' places sent up and `body`'s products wave by wave, the copies behind the call, and the sum with the host's sums (the first GPU computes them
    /// once its products are queued). `early` is queued while the first GPU's host reads the choices and plans.
    ///
    /// # Safety
    /// The addresses of `call` are live device buffers of the sizes it names on this GPU, its stream `s`.
    pub unsafe fn run(
        &self,
        index: usize,
        pl: &Places,
        call: &Call,
        s: Stream,
        early: &mut dyn FnMut() -> Result<()>,
        body: &mut dyn FnMut(Stage) -> Result<()>,
    ) -> Result<()> {
        let (pairs, tokens) = (call.pairs, call.tokens);
        if pairs * 8 > self.way.bytes || pairs % tokens.max(1) != 0 {
            return err(format!("a call of {tokens} tokens and {pairs} pairs, where the buffers are made for {}", self.way.bytes / 8));
        }
        let k = pairs / tokens;
        let t0 = Instant::now();
        let span = |i: usize, t: Instant| {
            let mut a = self.spans.get();
            a[i] += t.elapsed().as_secs_f64() * 1e6;
            self.spans.set(a);
        };
        // (the host's terms as one GPU's cache has them: a call of a few tokens, in any mode but copies alone)
        let hybrid = (1..=CPU_TOKENS).contains(&tokens) && self.leader.as_ref().is_none_or(|l| l.borrow().off.mode != crate::offload::Mode::Copy);
        let host = self.cache.host(index)?;
        let latent = host.latent;
        let key = (index, pl.calls.get());
        pl.calls.set(key.1 + 1);
        // (the device path's decode steps moved experts: the places as the GPU has them, before this call is planned; this GPU's work waited for)
        if let Some(t) = &pl.dev {
            self.got.record(s)?;
            self.got.wait_host()?;
            t.download()?;
            if let Some(ld) = &self.leader {
                ld.borrow_mut().set_places(index, pl.experts, pl.slots, &t.mirror.borrow().1);
            }
            // (GLYD_KIMI_STATS: the decode steps since the last prompt, every layer's, said at the next prompt's first layer)
            if let Some((last, sum, first)) = &self.dev_stats {
                if first.get().is_none_or(|f| index <= f) {
                    let s = sum.replace([0; 3]);
                    if s[0] > 0 {
                        eprintln!(
                            "device path, decode steps since the last prompt: {} choices, {} found in a place ({:.1}%), {} copied in",
                            s[0],
                            s[1],
                            100.0 * s[1] as f64 / s[0] as f64,
                            s[2]
                        );
                    }
                    first.set(Some(index));
                }
                let c = t.counters()?;
                let was = last.borrow_mut().insert(index, c).unwrap_or([0; 3]);
                let mut s = sum.get();
                for i in 0..3 {
                    s[i] += c[i].wrapping_sub(was[i]) as u64;
                }
                sum.set(s);
            }
        }
        if self.leader.is_some() {
            copy_d2h_async(self.ids.ptr, call.ids, pairs * 8, s)?;
            if hybrid {
                copy_d2h_async(self.wts.ptr, call.wts, pairs * 4, s)?;
                copy_d2h_async(self.z.ptr, call.z, tokens * latent * 2, s)?;
            }
            self.got.record(s)?;
        }
        early()?;
        let plan = match &self.leader {
            Some(ld) => {
                let t = Instant::now();
                self.got.wait_host()?;
                span(0, t);
                let t = Instant::now();
                let ids: Vec<i64> = std::slice::from_raw_parts(self.ids.ptr as *const i64, pairs).to_vec();
                if let Some(f) = &self.trace {
                    use std::io::Write;
                    let line: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
                    writeln!(f.borrow_mut(), "{index} {tokens} {}", line.join(" ")).map_err(|e| Error(format!("GLYD_KIMI_TRACE: {e}")))?;
                }
                let p = Arc::new(ld.borrow_mut().plan(index, &ids, hybrid, pl.experts, pl.slots, self.expert_bytes)?);
                self.cache.post(key, p.clone(), self.n - 1)?;
                span(1, t);
                p
            }
            None => {
                let t = Instant::now();
                let p = self.cache.take(key)?;
                span(1, t);
                p
            }
        };
        let t = Instant::now();
        let waves = plan.waves.len();
        for (w, wave) in plan.waves.iter().enumerate() {
            self.copy(pl, &host, &wave.loads)?;
            pl.ready.wait(s)?;
            // (the pinned places of the last call sent before they are written again)
            self.sent.wait_host()?;
            std::slice::from_raw_parts_mut(self.way.ptr as *mut i64, pairs).copy_from_slice(&wave.way);
            copy_h2d_async(call.way, self.way.ptr, pairs * 8, s)?;
            self.sent.record(s)?;
            body(Stage::Products(w, waves, wave.way.iter().any(|&p| p >= 0)))?;
            pl.done.record(s)?;
        }
        // (the copies behind the call: after its products, as the places the policy gives them may be ones the call read)
        self.copy(pl, &host, &plan.promote)?;
        // (the device path's table told of the call's copies, and the engine's stream after them: a decode step's copies come after)
        if let Some(t) = &pl.dev {
            let loads: Vec<(u16, u32)> = plan.waves.iter().flat_map(|w| w.loads.iter().copied()).chain(plan.promote.iter().copied()).collect();
            t.apply(&loads)?;
            pl.ready.wait(s)?;
        }
        span(2, t);
        self.calls.set(self.calls.get() + 1);
        let Some(sums) = &plan.sums else {
            let r = body(Stage::Sum(0));
            span(4, t0);
            return r;
        };
        if let Some(ld) = &self.leader {
            let t = Instant::now();
            let ids = std::slice::from_raw_parts(self.ids.ptr as *const i64, pairs);
            let wts = std::slice::from_raw_parts(self.wts.ptr as *const f32, pairs);
            let z = std::slice::from_raw_parts(self.z.ptr as *const u16, tokens * latent);
            let r = ld.borrow_mut().compute(&host, z, ids, wts, &plan.mine, k, tokens);
            *sums.got.lock().map_err(|_| poisoned())? = Some(r.as_ref().map(|v| Arc::new(v.clone())).map_err(|e| e.msg.clone()));
            sums.cv.notify_all();
            span(3, t);
            r?;
        }
        let got = {
            let t0 = Instant::now();
            let mut g = sums.got.lock().map_err(|_| poisoned())?;
            loop {
                if let Some(r) = g.as_ref() {
                    break r.clone().map_err(Error)?;
                }
                if t0.elapsed() > WAIT {
                    return err("the host's sums of a call did not come from the first GPU's thread");
                }
                g = sums.cv.wait_timeout(g, Duration::from_secs(1)).map_err(|_| poisoned())?.0;
            }
        };
        self.sent.wait_host()?;
        std::slice::from_raw_parts_mut(self.sums.ptr as *mut f32, tokens * latent).copy_from_slice(&got);
        copy_h2d_async(call.part, self.sums.ptr, tokens * latent * 4, s)?;
        self.sent.record(s)?;
        let r = body(Stage::Sum(call.part));
        span(4, t0);
        r
    }

    /// No pair of a wave has a place: its sum is nothing but the host's.
    ///
    /// # Safety
    /// `y` is a live device buffer of `bytes`.
    pub unsafe fn zero(y: u64, bytes: usize, s: Stream) -> Result<()> {
        memset_async(y, 0, bytes, s)
    }

    /// This GPU's place in the group and the group's size.
    pub fn place(&self) -> (usize, usize) {
        (self.rank, self.n)
    }

    /// The group's shared cache.
    pub fn cache_ref(&self) -> &Arc<TpCache> {
        &self.cache
    }
}

/// The device path's policy (kernels/kimi.cu's dc_plan) for one layer: CLOCK over `slots` places, a place this step's left alone. Places start holding
/// experts 0 to slots - 1.
#[cfg(feature = "dev")]
struct Clock {
    place_of: Vec<i32>,
    expert_of: Vec<i32>,
    last: Vec<u32>,
    refb: Vec<bool>,
    hand: usize,
    step: u32,
}

#[cfg(feature = "dev")]
impl Clock {
    fn new(experts: usize, slots: usize) -> Clock {
        let place_of = (0..experts).map(|e| if e < slots { e as i32 } else { -1 }).collect();
        Clock { place_of, expert_of: (0..slots as i32).collect(), last: vec![0; slots], refb: vec![false; slots], hand: 0, step: 0 }
    }

    /// One decode step's choices (distinct experts): how many were in a place.
    fn call(&mut self, ids: &[usize]) -> usize {
        self.step += 1;
        let (step, slots) = (self.step, self.expert_of.len());
        let mut hits = 0;
        for &e in ids {
            if let Ok(q) = usize::try_from(self.place_of[e]) {
                hits += 1;
                (self.last[q], self.refb[q]) = (step, true);
            }
        }
        for &e in ids {
            if self.place_of[e] >= 0 {
                continue;
            }
            let mut q = None;
            for _ in 0..2 * slots + 2 {
                let c = self.hand;
                self.hand = (self.hand + 1) % slots;
                if self.last[c] == step {
                    continue;
                }
                if self.refb[c] {
                    self.refb[c] = false;
                    continue;
                }
                q = Some(c);
                break;
            }
            let Some(q) = q else { continue };
            if let Ok(old) = usize::try_from(self.expert_of[q]) {
                self.place_of[old] = -1;
            }
            (self.expert_of[q], self.place_of[e], self.last[q], self.refb[q]) = (e as i32, q as i32, step, true);
        }
        hits
    }
}

/// The share of a layer's choices an ideal cache of `slots` places finds (Belady's: a miss takes the place of the expert used again last), steps from
/// `warm` on counted: what no policy can beat.
#[cfg(feature = "dev")]
fn ideal(calls: &[Vec<usize>], experts: usize, slots: usize, warm: usize) -> (usize, usize) {
    // (each expert's steps, in order, and where it is in them)
    let mut uses: Vec<Vec<usize>> = vec![Vec::new(); experts];
    for (t, ids) in calls.iter().enumerate() {
        for &e in ids {
            uses[e].push(t);
        }
    }
    let mut at = vec![0usize; experts];
    let mut held: Vec<usize> = (0..slots.min(experts)).collect();
    let mut in_place = vec![false; experts];
    for &e in &held {
        in_place[e] = true;
    }
    let (mut hits, mut reads) = (0, 0);
    for (t, ids) in calls.iter().enumerate() {
        for &e in ids {
            while at[e] < uses[e].len() && uses[e][at[e]] <= t {
                at[e] += 1;
            }
        }
        let next = |e: usize, at: &[usize]| uses[e].get(at[e]).copied().unwrap_or(usize::MAX);
        for &e in ids {
            if t >= warm {
                reads += 1;
                hits += usize::from(in_place[e]);
            }
        }
        for &e in ids {
            if in_place[e] {
                continue;
            }
            // (the held expert not chosen now that is used again last)
            let (i, _) =
                held.iter().enumerate().filter(|(_, &h)| !ids.contains(&h)).max_by_key(|(_, &h)| next(h, &at)).expect("more places than a step's choices");
            in_place[held[i]] = false;
            held[i] = e;
            in_place[e] = true;
        }
    }
    (hits, reads)
}

/// `dcreplay TRACE [--places 318,430] [--warm 32] [--decode]` (development): a GLYD_KIMI_TRACE file's tokens replayed, one at a time, through the device
/// path's policy and an ideal cache, at each count of places a layer: the share of choices found in a place (the formula's --hit, src/ceiling.rs), over
/// the layers. A prompt's call is its tokens in turn (the host plans a prompt, so a run on the device path traces its prompts: a long one gives
/// thousands of tokens' routing); --decode: decode steps alone.
#[cfg(feature = "dev")]
pub fn replay_bench(a: &crate::args::Args) -> Result<()> {
    let path = a.need(1, "TRACE")?;
    let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{path}: {e}")))?;
    let mut layers: std::collections::BTreeMap<usize, Vec<Vec<usize>>> = std::collections::BTreeMap::new();
    let (mut most, decode) = (0, a.has("decode"));
    for line in text.lines() {
        let v: Vec<usize> = line.split_whitespace().map(|x| x.parse::<usize>().map_err(|_| Error(format!("{path}: {line:?}")))).collect::<Result<_>>()?;
        let (tokens, ids) = (v.get(1).copied().unwrap_or(0), v.get(2..).unwrap_or(&[]));
        if tokens == 0 || ids.is_empty() || ids.len() % tokens != 0 || (decode && tokens > 1) {
            continue;
        }
        most = most.max(ids.iter().copied().max().unwrap_or(0));
        let calls = layers.entry(v[0]).or_default();
        calls.extend(ids.chunks(ids.len() / tokens).map(<[usize]>::to_vec));
    }
    let experts = a.num("experts", most + 1)?;
    let warm: usize = a.num("warm", 32usize)?;
    let places: Vec<usize> = a
        .flag("places")
        .unwrap_or("318,430")
        .split(',')
        .map(|x| x.trim().parse::<usize>().map_err(|_| Error(format!("--places {x}"))))
        .collect::<Result<_>>()?;
    let steps = layers.values().map(Vec::len).max().unwrap_or(0);
    println!("{path}: {} layers, {steps} tokens{}, {experts} experts; the first {warm} not counted", layers.len(), if decode { " (decode steps)" } else { "" });
    println!("{:>7} {:>10} {:>10} {:>18}", "places", "CLOCK", "ideal", "CLOCK's layers");
    for &p in &places {
        let (mut ch, mut ih, mut n) = (0, 0, 0);
        let (mut lo, mut hi) = (1.0f64, 0.0f64);
        for calls in layers.values() {
            if p >= experts || calls.iter().any(|c| c.len() > p) {
                continue;
            }
            let mut c = Clock::new(experts, p);
            let (mut h, mut r) = (0, 0);
            for (t, ids) in calls.iter().enumerate() {
                let x = c.call(ids);
                if t >= warm {
                    (h, r) = (h + x, r + ids.len());
                }
            }
            let (ih_, _) = ideal(calls, experts, p, warm);
            (ch, ih, n) = (ch + h, ih + ih_, n + r);
            let f = h as f64 / r.max(1) as f64;
            (lo, hi) = (lo.min(f), hi.max(f));
        }
        println!("{p:>7} {:>10.3} {:>10.3} {:>8.3} to {:.3}", ch as f64 / n.max(1) as f64, ih as f64 / n.max(1) as f64, lo, hi);
    }
    Ok(())
}

/// `dcfill [--mb M] [--fills F] [--blocks 8,16,32,64]` (development): the device path's copies (kernels/kimi.cu's dc_fill, the GPU reading pinned host memory
/// itself) against the copy engine's (one copy, and a copy a fill), F fills of M MB each from host memory to this GPU: each's GB/s, the median of 10.
#[cfg(feature = "dev")]
pub fn fill_bench(a: &crate::args::Args) -> Result<()> {
    let dev = crate::dev::Dev::new(0)?;
    let k = crate::dynamic::kimi::KernKimi::load(&dev)?;
    let fills: usize = a.num("fills", 16usize)?;
    let len = ((a.num("mb", 2.19f64)? * 1e6) as usize).next_multiple_of(16).max(16);
    let host = Pinned::portable(fills * len)?;
    let devb = Buf::new(fills * len)?;
    let mut r = vec![0u64; 20];
    r[..5].copy_from_slice(&[host.ptr as u64, len as u64, 0, devb.ptr, len as u64]);
    let reg = Buf::new(160)?;
    reg.write(0, &r)?;
    let f: Vec<[i32; 2]> = (0..fills as i32).map(|i| [i, i]).collect();
    let (fb, nb) = (Buf::new(fills * 8)?, Buf::new(4)?);
    fb.write(0, &f)?;
    nb.write(0, &[fills as i32])?;
    let s = dev.stream;
    let time = |run: &dyn Fn() -> Result<()>| -> Result<f64> {
        let mut t = Vec::new();
        for _ in 0..11 {
            let (e0, e1) = (Ev::new()?, Ev::new()?);
            e0.record(s)?;
            run()?;
            e1.record(s)?;
            dev.sync()?;
            t.push(e1.since(&e0)? as f64);
        }
        t.sort_by(f64::total_cmp);
        Ok((fills * len) as f64 / (t[5] * 1e-3) / 1e9)
    };
    println!("{fills} fills of {:.2} MB from pinned host memory to {}", len as f64 / 1e6, dev.name);
    for b in a.flag("blocks").unwrap_or("8,16,32,64").split(',').filter_map(|x| x.trim().parse::<u32>().ok()) {
        // SAFETY: the buffers above, of these sizes; the kernel reads fills [nfill] and the regions' bytes.
        let g = time(&|| unsafe { crate::launch!(k.dc_fill, [b, fills as u32, 1], [256, 1, 1], s; fb.ptr, nb.ptr, reg.ptr) })?;
        println!("  dc_fill, {b:>3} blocks a fill: {g:6.1} GB/s");
    }
    // SAFETY: as above.
    let one = time(&|| unsafe { copy_h2d_async(devb.ptr, host.ptr as *const _, fills * len, s) })?;
    let each = time(&|| {
        for i in 0..fills {
            // SAFETY: as above, fill i's bytes.
            unsafe { copy_h2d_async(devb.ptr + (i * len) as u64, (host.ptr as *const u8).add(i * len) as *const _, len, s)? };
        }
        Ok(())
    })?;
    println!("  the copy engine, one copy: {one:6.1} GB/s; a copy a fill: {each:6.1} GB/s");
    Ok(())
}

#[cfg(all(test, feature = "dev"))]
mod tests {
    use super::*;

    /// Five steps of one choice over three experts and two places, as dc_plan decides them and as the ideal cache does (worked by hand).
    #[test]
    fn the_clock_and_the_ideal_over_a_short_trace() {
        let calls: Vec<Vec<usize>> = [2, 0, 2, 1, 2].iter().map(|&e| vec![e]).collect();
        let mut c = Clock::new(3, 2);
        let hits: usize = calls.iter().map(|ids| c.call(ids)).sum();
        assert_eq!(hits, 1);
        assert_eq!(ideal(&calls, 3, 2, 0), (3, 5));
    }
}
