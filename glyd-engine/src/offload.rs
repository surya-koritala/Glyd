//! The expert cache (Linear GLY-130): a mixture-of-experts model whose routed experts do not all fit in the GPU's memory keeps every expert of every layer in host memory (pinned, in the packed form the
//! grouped products read) and a fixed number of them a layer in the GPU's, in places of one size (`ExpertStack::pooled`), copied in on a copy stream as the router chooses them.
//!
//! What a layer's call does (`ExpertStore::run`): the router's choices are read to the host (the one wait a layer has), `LayerCache` says which of the chosen experts are in a place already and which places the others
//! take (never the place of an expert the call needs), the copies of the others are queued on the copy stream behind the engine's stream's last use of those places, the engine's stream waits for them with an event,
//! each pair's place goes to the device, and the grouped products run on a plan made of the places (`MoeLayer::experts_products`). No kernel is changed: a layer that is all there sorts its pairs by expert and this
//! one by place, and each pair's rows are the same whatever the order of the plan. A call that chooses more experts than a layer has places for is computed in waves of as many as fit, a wave's pairs the ones whose
//! expert it holds and the rest routed nowhere in it; the last wave's sum is the layer's (the earlier waves' rows stay in the products' workspace).
//!
//! The host's half (Linear GLY-130; `Mode`): an expert that is not in a place need not be copied for the call. With the router's weights and the experts' input z read back beside the choices (2 KB a token), the host computes
//! the pairs of those experts from its own memory (src/cpu_experts.rs: the second matrices are the host tier's own bytes, the first ones a copy of them in the row layout, `CpuRows`) while the GPU runs the products of the
//! experts that are in places; the host's weighted sums of a token's pairs (4 KB a token, fp32) go up and the sum adds them to the GPU's before its one rounding (`MoeLayer::experts_sum`), the pairs the host computed being
//! no experts in the sum's ids. `plan` chooses, for each miss, the host or a copy by the times it expects (`Costs`, measured as the run goes); an expert the host computed twice within a few calls gets a place in the
//! background (`Offload::window`), behind the call's products on the copy stream, which the layer's next call waits for as it does for any copy. The wait on the host stays one a layer.
//!
//! A layer whose experts are in groups of different sizes (the Hummingbird level's bit widths, src/moe_hb.rs) has a layer of the cache for each group, and a call of it with the host's half is `ExpertStore::run_groups`: one
//! wait for the layer, the misses of both groups planned together (`plan_groups`: the host's one call takes the pairs of both, a copy costs by its group's bytes), the host reading the experts' places in the host tier where they
//! are (`HostHalf::Hb`: no second copy), the rotated input read back as fp32.
//!
//! The disk tier (the SSD tier; `Tier`, `ExpertStore::add_layer_on_disk`): a computer with 32 GB of memory cannot pin the 39.2 GiB of a Hummingbird model's experts. With a budget for the experts (`HostMemory`, `--host-memory`) smaller
//! than they are, a group of a layer keeps `ram` of its experts in slots of the host tier (pinned, each slot an expert's place: `ram` of the group's experts end to end) and the rest in the packed directory, read when the router
//! asks for one. Which expert is in which slot is a `LayerCache` of its own (a least-recently-used one over the experts a call missed on the GPU); every reader of the host tier's bytes goes through the slot: `fetch` (the copies to
//! places: of a call's misses, of hints and of promotions) and `Resident::hb_expert` (the host's half). An expert that is not in a slot is read before it is used (`stage`: the reads of a call made 16 at a time by the pool of
//! src/disk.rs, bypassing the page cache), into the slot of the least recently used expert the call does not need, and a slot that a queued copy has yet to read is not written (`Tier::busy`: the number of the last batch of copies
//! to read it, against the batches the copy stream is known to have run, `Cuda::retired`; a victim that is still read is waited for by syncing the copy stream). The tier changes where an expert's bytes are and never what is
//! computed: the GPU's cache, its waves, the planner and the host's pairs are what they are with every expert in memory (a call whose experts do not all fit the slots is staged, copied and computed a slot's worth at a time),
//! so that a run with the tier is the run without it to the bit (`hold_costs` fixes the planner's times, which a run measures, so that two runs plan alike).
//!
//! Seams: the policy is `LayerCache` (a least-recently-used cache; `request`, `hint`, and seeding, which is a hint of a profile's experts, least used first); a predictor (`ExpertStore::set_predictor`: `run` asks it, for
//! each call, for the experts the layers ahead will be asked for; or `hint` between steps) has its copies queued beside the engine's work. The bytes of an expert in the host tier are addressed by `HostPack` alone and its places by `PackIn::dev`, so a second form of an expert (another layout, another width) is
//! another matrix of the layer and another call of the grouped products; the pairs a wave computes are the ids `run` hands the products, so experts computed elsewhere are ids routed nowhere. What to do with a miss is
//! `plan` alone.

use crate::cpu_experts::{checked_kernel, default_threads, experts_views, row_stride, ExpertView, Kernel, PairView, RowsView, Z};
use crate::cpu_hb::{self, HbExpert, HbPair};
use crate::dev::{copy_d2h_async, copy_h2d_async, memset_async, Dev, Ev, EvRef, Pinned, Side};
use crate::disk::{self, Job};
use crate::ffi::{err, Error, Result, Stream};
use crate::moe_hb::HbPlace;
use crate::moe_hbg::{HbgExpert, HbgPlace};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

const NONE: u32 = u32::MAX;

/// The sections of the pinned buffer the pairs' places are made in: one for each of the two groups of a layer's experts, and the ids of the sum.
const WAYS: usize = 3;

/// The most tokens a call has some of its experts computed on the host for (its z, its weights and its sums are read back and sent up: a prompt chunk past it is copied and computed on the GPU whole).
pub const CPU_TOKENS: usize = 64;

// ---- one layer's residency and the policy (no GPU) ---------------------------------------------------------------------------------------------------------------------------------------

/// What a request of experts found: how many were in a place already, and the (expert, place) of each one that was not and is now to be copied there.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Loaded {
    pub hits: usize,
    pub loads: Vec<(u16, u32)>,
}

/// Which of a layer's experts are in which of its places, and the policy that decides which one leaves for a new one: the least recently used of the experts the request does not need. The places are in a list
/// by use (`older` and `newer` link them, `head` the most recent); a request's experts are used in the order given, so the last is the most recent.
pub struct LayerCache {
    slot_of: Vec<u32>,
    expert_in: Vec<u32>,
    older: Vec<u32>,
    newer: Vec<u32>,
    head: u32,
    tail: u32,
    free: Vec<u32>,
    // the request each expert was last asked for in (the ones the current request needs are not taken)
    asked: Vec<u64>,
    request: u64,
}

impl LayerCache {
    /// A cache of `slots` places for a layer of `experts` experts, empty.
    pub fn new(experts: usize, slots: usize) -> LayerCache {
        LayerCache {
            slot_of: vec![NONE; experts],
            expert_in: vec![NONE; slots],
            older: vec![NONE; slots],
            newer: vec![NONE; slots],
            head: NONE,
            tail: NONE,
            free: (0..slots as u32).rev().collect(),
            asked: vec![0; experts],
            request: 0,
        }
    }

    pub fn slots(&self) -> usize {
        self.expert_in.len()
    }

    pub fn experts(&self) -> usize {
        self.slot_of.len()
    }

    /// The place expert `e` is in.
    pub fn slot_of(&self, e: usize) -> Option<usize> {
        self.slot_of.get(e).copied().filter(|&s| s != NONE).map(|s| s as usize)
    }

    /// Each place's expert (-1 where free): what `set_places` takes.
    pub fn expert_in(&self) -> Vec<i32> {
        self.expert_in.iter().map(|&e| if e == NONE { -1 } else { e as i32 }).collect()
    }

    /// The experts in places now.
    pub fn resident(&self) -> usize {
        self.expert_in.len() - self.free.len()
    }

    /// Place `p` taken out of the list by use.
    fn unlink(&mut self, p: u32) {
        let (o, n) = (self.older[p as usize], self.newer[p as usize]);
        match o {
            NONE => self.tail = n,
            o => self.newer[o as usize] = n,
        }
        match n {
            NONE => self.head = o,
            n => self.older[n as usize] = o,
        }
    }

    /// Place `p` (not in the list) made the most recently used.
    fn push(&mut self, p: u32) {
        self.older[p as usize] = self.head;
        self.newer[p as usize] = NONE;
        match self.head {
            NONE => self.tail = p,
            h => self.newer[h as usize] = p,
        }
        self.head = p;
    }

    /// The place a miss takes: a free one, else the one least recently used whose expert the current request does not need.
    fn victim(&mut self) -> Option<u32> {
        if let Some(p) = self.free.pop() {
            return Some(p);
        }
        let mut p = self.tail;
        while p != NONE && self.asked[self.expert_in[p as usize] as usize] == self.request {
            p = self.newer[p as usize];
        }
        (p != NONE).then_some(p)
    }

    /// Experts `wanted` (distinct, at most `slots`) used in order: the ones in a place are touched, the others take a place and are listed to be copied there.
    fn take(&mut self, wanted: &[u16]) -> Result<Loaded> {
        if wanted.len() > self.slots() {
            return err(format!("{} experts asked of a cache of {}", wanted.len(), self.slots()));
        }
        self.request += 1;
        for &e in wanted {
            let n = self.slot_of.len();
            let a = self.asked.get_mut(e as usize).ok_or_else(|| Error(format!("expert {e} of {n}")))?;
            if *a == self.request {
                return err(format!("expert {e} asked for twice"));
            }
            *a = self.request;
        }
        let mut got = Loaded::default();
        for &e in wanted {
            let p = match self.slot_of[e as usize] {
                NONE => {
                    let p = self.victim().ok_or_else(|| Error("no place to take".into()))?;
                    let old = std::mem::replace(&mut self.expert_in[p as usize], e as u32);
                    if old != NONE {
                        self.slot_of[old as usize] = NONE;
                        self.unlink(p);
                    }
                    self.slot_of[e as usize] = p;
                    got.loads.push((e, p));
                    p
                }
                p => {
                    got.hits += 1;
                    self.unlink(p);
                    p
                }
            };
            self.push(p);
        }
        Ok(got)
    }

    /// A call's experts (distinct, at most `slots`), used in order: `hits` are in places, `loads` are to be copied.
    pub fn request(&mut self, needed: &[u16]) -> Result<Loaded> {
        self.take(needed)
    }

    /// Experts a predictor expects the layer to be asked for, least expected first (repeats and ids that are no expert left out; of more than `slots`, the last): placed and touched as a request's are, so the last is the most
    /// recent. On an empty cache this is seeding: a static profile's experts, the most used last.
    pub fn hint(&mut self, experts: &[u16]) -> Result<Loaded> {
        let mut want = distinct(&experts.iter().map(|&e| e as i64).collect::<Vec<_>>(), self.experts());
        want.drain(..want.len().saturating_sub(self.slots()));
        self.take(&want)
    }

    /// The places as `expert_in` has them (an expert's number a place, negative where free: a GPU's own table read back, src/dynamic/tp_cache.rs): each
    /// expert's place from it, the places in their numbers' order as the order of use, the others free. An expert named twice keeps its first place.
    pub fn set_places(&mut self, expert_in: &[i32]) {
        let slots = self.expert_in.len();
        self.slot_of.iter_mut().for_each(|x| *x = NONE);
        (self.older, self.newer, self.head, self.tail) = (vec![NONE; slots], vec![NONE; slots], NONE, NONE);
        for p in 0..slots {
            let e = expert_in.get(p).copied().unwrap_or(-1);
            self.expert_in[p] = NONE;
            if e >= 0 && (e as usize) < self.slot_of.len() && self.slot_of[e as usize] == NONE {
                self.expert_in[p] = e as u32;
                self.slot_of[e as usize] = p as u32;
                self.push(p as u32);
            }
        }
        self.free = (0..slots as u32).rev().filter(|&p| self.expert_in[p as usize] == NONE).collect();
    }

    /// The cache made `slots` places: the experts in places past them leave it, the others keep their places and their order of use; places added are free.
    pub fn resize(&mut self, slots: usize) {
        let mut order = Vec::new();
        let mut p = self.tail;
        while p != NONE {
            order.push(p);
            p = self.newer[p as usize];
        }
        for &e in self.expert_in.iter().skip(slots) {
            if e != NONE {
                self.slot_of[e as usize] = NONE;
            }
        }
        self.expert_in.resize(slots, NONE);
        (self.older, self.newer, self.head, self.tail) = (vec![NONE; slots], vec![NONE; slots], NONE, NONE);
        for p in order.into_iter().filter(|&p| (p as usize) < slots) {
            self.push(p);
        }
        self.free = (0..slots as u32).rev().filter(|&p| self.expert_in[p as usize] == NONE).collect();
    }

    /// A call's experts as the waves it is computed in: one (as given) where they fit in the places, else waves of as many as fit with the ones in places first, so that each expert is copied once.
    pub fn waves(&self, needed: &[u16]) -> Vec<Vec<u16>> {
        if needed.len() <= self.slots() {
            return vec![needed.to_vec()];
        }
        let (there, others): (Vec<u16>, Vec<u16>) = needed.iter().partition(|&&e| self.slot_of[e as usize] != NONE);
        there.into_iter().chain(others).collect::<Vec<u16>>().chunks(self.slots()).map(<[u16]>::to_vec).collect()
    }
}

/// The distinct experts among the router's choices `ids` of a call, in the order they first appear (an id outside the layer's experts is no expert).
pub fn distinct(ids: &[i64], experts: usize) -> Vec<u16> {
    let mut seen = vec![false; experts];
    let mut out = Vec::new();
    for &i in ids {
        if i >= 0 && (i as usize) < experts && !seen[i as usize] {
            seen[i as usize] = true;
            out.push(i as u16);
        }
    }
    out
}

/// A call's distinct experts (`needed`) in the order of their pairs among `ids`, the fewest first (the order given among equals): asked for in this order, the busiest are the most recently used, and where a call
/// is computed in waves (`LayerCache::waves`) its last wave, whose experts stay in places for the next call, holds the busiest of its misses. On the Nemotron 3 Super decode traces at 16 and 20 rows a step this
/// finds 1 to 7 points more of the pairs in places than the order the experts first appear in, and never fewer (the test below).
pub fn by_rows(mut needed: Vec<u16>, ids: &[i64], experts: usize) -> Vec<u16> {
    let mut n = vec![0u32; experts];
    ids.iter().filter(|&&e| e >= 0 && (e as usize) < experts).for_each(|&e| n[e as usize] += 1);
    needed.sort_by_key(|&e| n[e as usize]);
    needed
}

/// A layer's experts ranked by routing counts, the most used first and the lower expert first among equals (numpy's stable argsort of the negated counts): the first `k` are a static profile of the layer.
pub fn profile(counts: &[u64], k: usize) -> Vec<u16> {
    let mut ranked: Vec<u16> = (0..counts.len() as u16).collect();
    ranked.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
    ranked.truncate(k);
    ranked
}

/// Each pair's place for a wave (`place[expert]`: the expert's place if it is in the wave, else -1), -1 for a pair whose id is no expert: what the plan of the grouped products is made of.
pub fn fill_way(ids: &[i64], place: &[i64], out: &mut [i64]) {
    for (o, &i) in out.iter_mut().zip(ids) {
        *o = if i >= 0 && (i as usize) < place.len() { place[i as usize] } else { -1 };
    }
}

// ---- what to do with a miss (no GPU) ----------------------------------------------------------------------------------------------------------------------------------------------------

/// What happens to an expert a call needs that is not in a place: `Copy` copies it before the call's products (the cache's first version, and the only way for a call of more than `CPU_TOKENS` tokens); `Cpu` has the host
/// compute its pairs; `Split` does both, a share of the misses each so that the copies and the host's work take the same time; `Auto` does what the times expected say is quickest, the host's alone, copies alone or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Copy,
    Cpu,
    Split,
    Auto,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        [Mode::Copy, Mode::Cpu, Mode::Split, Mode::Auto].into_iter().find(|m| m.name() == s)
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Copy => "copy",
            Mode::Cpu => "cpu",
            Mode::Split => "split",
            Mode::Auto => "auto",
        }
    }
}

/// How the cache serves a miss: the `Mode`, the threads of the host's crew (the thread that calls is one; `default_threads`) and whether each keeps to a core, the calls within which a second miss of an expert is
/// worth a place (`window`: 16 by default), and the places a call starts copying in behind it at most (`promote`: 4). Over the record's decode traces (`serving_the_traces_with_the_host_and_promotions`) a window of 16
/// calls finds as many reads in places as one of 32 or 128 with fewer copies, and 4 a call finds 5 to 6 points more than 2 where the copy stream keeps up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Offload {
    pub mode: Mode,
    pub threads: usize,
    pub pin: bool,
    pub window: u64,
    pub promote: usize,
    /// A pair that would be computed by the host and whose routing weight is under this is left out of the sum (0: none is): the cold tail of a token's experts, for the speed of the host's half.
    pub skip: f32,
    /// A missed expert that every row of the call holding it ranks among its `skip_rank` smallest weights (of the row's k) is left out of the sum, neither copied nor computed (0: none is): the tail of a verification's
    /// rows, whose union of experts grows with the rows.
    pub skip_rank: usize,
    /// `skip_rank` for the rows after the first of a verification in parts (its guesses' rows; 0: `skip_rank` for them too).
    pub skip_rank_later: usize,
    /// Whether the copies of experts into places are made by a thread of their own: a copy from memory that is not pinned (a mapped file's pages) holds the thread that makes it for the time of the transfer, which is the
    /// time of the call's host work; with a copy thread the call goes on to the host's pairs at once, and a promotion's copies cost the call nothing.
    pub async_copies: bool,
    /// Whether a layer whose host half the plan on the GPU computes (FP4 experts) plans its decode calls on the GPU (src/devplan.rs): no host waits inside such a call, and a step of them is one graph.
    pub device: bool,
}

impl Default for Offload {
    fn default() -> Offload {
        Offload {
            mode: Mode::Auto,
            threads: default_threads(),
            pin: false,
            window: 16,
            promote: 4,
            skip: 0.0,
            skip_rank: 0,
            skip_rank_later: 0,
            async_copies: false,
            device: false,
        }
    }
}

/// The host's pairs `mine` (numbers into a call's pairs, `k` a row) less those of the experts that every row holding them ranks among its `bottom` smallest weights (ties: the later pair the smaller).
fn tail_skip(mine: &mut Vec<usize>, ids: &[i64], wts: &[f32], k: usize, bottom: usize) {
    tail_skip_rows(mine, ids, wts, k, bottom, bottom);
}

/// `tail_skip` with the rows after the first ranked against a tail of their own (`later`: a verification's guesses, whose rows are kept only where the guesses are right).
fn tail_skip_rows(mine: &mut Vec<usize>, ids: &[i64], wts: &[f32], k: usize, first: usize, later: usize) {
    if (first == 0 && later == 0) || k == 0 {
        return;
    }
    let cut = |j: usize| k.saturating_sub(if j < k { first } else { later });
    let rank = |j: usize| {
        let r0 = j / k * k;
        (r0..r0 + k).filter(|&i| wts[i] > wts[j] || (wts[i] == wts[j] && i < j)).count()
    };
    let head: Vec<i64> = mine.iter().filter(|&&j| rank(j) < cut(j)).map(|&j| ids[j]).collect();
    mine.retain(|&j| head.contains(&ids[j]));
}

/// The offload settings of a command line: `--cpu-threads N` (the threads of the host's crew, the thread that calls included; the computer's cores less two where none is given), and in a development build `--offload-mode
/// copy|cpu|split|auto`, `--cpu-pin` (each thread of the crew on a core of its own), `--offload-window N` and `--offload-promote N`.
pub fn offload_args(a: &crate::args::Args) -> Result<Offload> {
    #[allow(unused_mut)] // (a release build sets nothing more)
    let mut o = Offload { threads: a.num("cpu-threads", Offload::default().threads)?.max(1), ..Offload::default() };
    #[cfg(feature = "dev")]
    {
        if let Some(m) = a.flag("offload-mode") {
            o.mode = Mode::parse(m).ok_or_else(|| Error(format!("--offload-mode {m}: copy, cpu, split or auto")))?;
        }
        o.pin = a.has("cpu-pin");
        o.window = a.num("offload-window", o.window)?;
        o.promote = a.num("offload-promote", o.promote)?;
        o.device = a.has("device-plan");
    }
    Ok(o)
}

/// One variant of a measurement's sweep, `MODE[:THREADS][:pin]` (`cpu`, `cpu:16`, `split:8:pin`), on the settings `base`; None for text that is not one.
#[cfg(feature = "dev")]
pub fn parse_variant(text: &str, base: &Offload) -> Option<Offload> {
    let mut parts = text.split(':');
    let mut o = Offload { mode: Mode::parse(parts.next()?)?, ..*base };
    for p in parts {
        match p {
            "pin" => o.pin = true,
            n => o.threads = n.parse::<usize>().ok().filter(|&n| n > 0)?,
        }
    }
    Some(o)
}

/// The copy stream's backlog, in microseconds of copying, past which a call starts no copies behind it (the next call of a layer waits for its layer's, and the copies of a token must not outlast the token).
const BACKLOG_US: f64 = 10_000.0;

/// The times `plan` weighs, in microseconds: copying one expert (its bytes over the link), what a call of the host's has to start with (waking the crew), the host's time for an expert's first row (its bytes read from the
/// computer's memory, and the row's products) and for each row of it after the first (`row`: the products alone, the expert's bytes in the core's cache by then), and the copying already queued on the copy stream ahead
/// of the call's own (`backlog`: the stream is one queue, so a copy the call asks for starts after it and the call's products wait for the copy). The first three figures are the records' (an expert of 3.1 MB at 12.4 GB/s,
/// benchmarks/cpu/cpu-experts-epyc-2026-10-10: a layer's call of 2.5 to 10 pairs took 253 to 610 microseconds); `row` starts as `pair` (each row as dear as the first, as a host that reads an expert for each of its
/// pairs pays); `copied` and `computed` move them to what this computer does. The backlog is not learned: the store sets it on the copy of the costs a call is planned with (`queued`), and it is 0 where the stream is
/// idle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub copy: f64,
    pub fixed: f64,
    pub pair: f64,
    pub row: f64,
    pub backlog: f64,
}

impl Default for Costs {
    fn default() -> Costs {
        Costs { copy: 250.0, fixed: 100.0, pair: 50.0, row: 50.0, backlog: 0.0 }
    }
}

impl Costs {
    const SMOOTH: f64 = 0.25;

    /// A batch of copies of `bytes` took `ms`: an expert of `expert` bytes takes its share of that.
    pub fn copied(&mut self, bytes: u64, ms: f64, expert: u64) {
        if bytes > 0 && ms > 0.0 {
            let sample = (ms * 1e3 * expert as f64 / bytes as f64).clamp(1.0, 1e5);
            self.copy += Costs::SMOOTH * (sample - self.copy);
        }
    }

    /// The time `bytes` of copies still queued will take: an expert of `expert` bytes takes `copy`.
    pub fn queued(&self, bytes: u64, expert: u64) -> f64 {
        bytes as f64 / expert.max(1) as f64 * self.copy
    }

    /// A call of the host's took `us` for `pairs` pairs of `experts` experts (each read once for its pairs): the call's time past the fixed part is `experts` first rows and `pairs - experts` further ones, and the
    /// two times move toward what makes it so (a normalized least-mean-squares step: a call of a row an expert moves `pair` alone, as the store's first model did; one of many rows an expert mostly `row`).
    pub fn computed(&mut self, experts: usize, pairs: usize, us: f64) {
        if pairs == 0 || experts == 0 || experts > pairs {
            return;
        }
        let (e, r) = (experts as f64, (pairs - experts) as f64);
        let miss = (us - self.fixed).max(0.0) - (e * self.pair + r * self.row);
        let step = Costs::SMOOTH * miss / (e * e + r * r);
        self.pair = (self.pair + step * e).clamp(2.0, 1e4);
        self.row = (self.row + step * r).clamp(0.1, 1e4);
    }

    /// The host's time for a miss of `rows` rows (its expert read once, then each row's products).
    pub fn host(&self, rows: usize) -> f64 {
        self.pair + self.row * rows.saturating_sub(1) as f64
    }
}

/// An expert a call needs that is not in a place: the pairs of the call that are its (tokens that chose it), and whether the host computed it a short while ago (a second miss soon after the first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Miss {
    pub expert: u16,
    pub pairs: usize,
    pub hot: bool,
}

/// Which of the misses are copied to a place (true) and which are computed on the host (false); `room` places are free for copies (the call's other experts keep theirs). The host's time is by the pair and a copy costs the
/// same whatever the pairs, so the copies go to the misses with the most pairs, the hot ones first among equals, and `k` of them are copied: `Split` takes the `k` at which the copies and the host's work take the same
/// time, `Auto` the `k` at which the longer of the two is shortest (none, all or some), both by the times in `costs`. `Copy` copies all and `Cpu` none. The copy stream is one queue and the call's products wait for the
/// copies, so for `Auto` the copies take the stream's `costs.backlog` and their own (none: no wait); a decode step's misses of a pair each are the host's while the stream is behind, and the heavy misses of a prefill
/// are copied while the backlog is shorter than the host's work they save (`Split` keeps its definition: the copies and the host's work take the same time, backlog or none).
pub fn plan(mode: Mode, misses: &[Miss], room: usize, costs: &Costs) -> Vec<bool> {
    plan_groups(mode, &[misses.to_vec()], &[room], &[1.0], costs).remove(0)
}

/// `plan` for a call whose experts are in groups that are caches of their own (the Hummingbird level's two bit widths): `misses[g]` those of group g, `rooms[g]` its free places, and `sizes[g]` the time of one of its
/// copies as a share of `costs.copy` (the bytes of its experts over the bytes `costs` counts a copy in). The host computes the pairs of the misses of all the groups that are not copied, in one call, so the copies go to the misses
/// with the most pairs of any group that has a place left, and `k` of them are copied as `plan` says.
pub fn plan_groups(mode: Mode, misses: &[Vec<Miss>], rooms: &[usize], sizes: &[f64], costs: &Costs) -> Vec<Vec<bool>> {
    match mode {
        Mode::Copy => return misses.iter().map(|m| vec![true; m.len()]).collect(),
        Mode::Cpu => return misses.iter().map(|m| vec![false; m.len()]).collect(),
        Mode::Split | Mode::Auto => {}
    }
    let mut order: Vec<(usize, usize)> = misses.iter().enumerate().flat_map(|(g, m)| (0..m.len()).map(move |i| (g, i))).collect();
    order.sort_by_key(|&(g, i)| (std::cmp::Reverse(misses[g][i].pairs), std::cmp::Reverse(misses[g][i].hot), g, i));
    // (the host's time is the misses' it keeps: each expert read once, then its rows, `Costs::host`)
    let (mut left, mut kept): (f64, usize) = (misses.iter().flatten().map(|m| costs.host(m.pairs)).sum(), misses.iter().map(Vec::len).sum());
    let (mut free, mut taken) = (rooms.to_vec(), vec![0usize; misses.len()]);
    let score = |left: f64, kept: usize, taken: &[usize]| {
        let copying: f64 = sizes.iter().zip(taken).map(|(size, &n)| costs.copy * size * n as f64).sum();
        let computing = if kept > 0 { costs.fixed + left } else { 0.0 };
        if mode == Mode::Split {
            (copying - computing).abs()
        } else {
            let waiting = if taken.iter().any(|&n| n > 0) { costs.backlog + copying } else { 0.0 };
            waiting.max(computing)
        }
    };
    let (mut best, mut chosen) = (score(left, kept, &taken), 0usize);
    let mut copies = Vec::new();
    for &(g, i) in &order {
        if free[g] == 0 {
            continue;
        }
        free[g] -= 1;
        taken[g] += 1;
        left -= costs.host(misses[g][i].pairs);
        kept -= 1;
        copies.push((g, i));
        let now = score(left, kept, &taken);
        if now < best - 1e-9 {
            (best, chosen) = (now, copies.len());
        }
    }
    let mut copy: Vec<Vec<bool>> = misses.iter().map(|m| vec![false; m.len()]).collect();
    copies.iter().take(chosen).for_each(|&(g, i)| copy[g][i] = true);
    copy
}

/// Whether a call whose misses are more than the places its hits leave (`room`) is quicker with every miss copied, in waves of as many as the places hold (`LayerCache::waves`), than as `plan` serves it (at most
/// `room` copies, the host the rest), by the times expected in `costs`. Only `Auto` decides so: with many rows a miss has many pairs, and the host's time for them can pass the copies' (DeepSeek-V4 at 32 rows on the
/// 4080 SUPER, 32 places for 64 experts: 334 ms a step with the host against 280 copying all, `storebench`).
pub fn waves_quicker(mode: Mode, misses: &[Miss], room: usize, costs: &Costs) -> bool {
    if mode != Mode::Auto || misses.len() <= room {
        return false;
    }
    let (mut copying, mut host, mut kept) = (0.0, 0.0, 0usize);
    for (m, c) in misses.iter().zip(plan(mode, misses, room, costs)) {
        if c {
            copying += costs.copy;
        } else {
            (host, kept) = (host + costs.host(m.pairs), kept + 1);
        }
    }
    let waiting = if copying > 0.0 { costs.backlog + copying } else { 0.0 };
    let planned = waiting.max(if kept > 0 { costs.fixed + host } else { 0.0 });
    costs.backlog + costs.copy * (misses.len() as f64) < planned
}

/// What a call did with its experts (`serve`): the experts that got a place now in the order the call asked for them, the copies to queue ((expert, place) of the misses `plan` copies), the reads found in a place (`hits`) and
/// not (`misses`), the misses left to the host (`kept`), and the ones of those worth a place behind the call (`promote`).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Served {
    pub wanted: Vec<u16>,
    pub loads: Vec<(u16, u32)>,
    pub hits: usize,
    pub misses: usize,
    pub kept: Vec<u16>,
    pub promote: Vec<u16>,
}

/// A layer's call for the experts `needed` (distinct, in the router's order; `pairs_of[e]` the pairs of the call that are expert e's) at its `now`-th call: the hits keep their places, the misses are copied or kept for
/// the host as `plan` says (`off.mode`, the times `costs`; never more copies than the places the hits leave), the copies are given places (`LayerCache::request`: the experts of the call that are in places or get one are
/// the ones it needs, so nothing it needs is taken), and the misses the host keeps are noted in `last` (the call each expert was last kept in, 0: never): one that was kept within `off.window` calls before is worth a place
/// behind this call (`off.promote` of them at most).
pub fn serve(cache: &mut LayerCache, needed: &[u16], pairs_of: &[usize], last: &mut [u64], now: u64, off: &Offload, costs: &Costs) -> Result<Served> {
    let mut parts = [Part { cache, needed, pairs_of, last, now, size: 1.0 }];
    Ok(serve_groups(&mut parts, off, costs)?.remove(0))
}

/// A group's share of a call (`serve_groups`): its cache, the experts the call needs of it and the pairs of each, the calls it was kept in, its call's number, and the time of one of its copies as a share of `costs.copy`.
pub struct Part<'a> {
    pub cache: &'a mut LayerCache,
    pub needed: &'a [u16],
    pub pairs_of: &'a [usize],
    pub last: &'a mut [u64],
    pub now: u64,
    pub size: f64,
}

/// `serve` for a call of several groups (caches of their own), the misses of all planned together (`plan_groups`: one call of the host's computes the pairs of the ones that are kept), and `off.promote` places behind
/// the call at most for all of them.
pub fn serve_groups(parts: &mut [Part], off: &Offload, costs: &Costs) -> Result<Vec<Served>> {
    let mut found: Vec<Vec<Miss>> = Vec::new();
    for p in parts.iter() {
        let mut misses = Vec::new();
        for &e in p.needed {
            if p.cache.slot_of(e as usize).is_none() {
                let seen = p.last[e as usize];
                misses.push(Miss { expert: e, pairs: p.pairs_of[e as usize], hot: seen != 0 && p.now - seen <= off.window });
            }
        }
        found.push(misses);
    }
    // (the places the hits leave)
    let rooms: Vec<usize> = parts.iter().zip(&found).map(|(p, m)| p.cache.slots() - (p.needed.len() - m.len())).collect();
    let sizes: Vec<f64> = parts.iter().map(|p| p.size).collect();
    let copies = plan_groups(off.mode, &found, &rooms, &sizes, costs);
    let (mut served, mut promote) = (Vec::new(), off.promote);
    for ((p, misses), (mut copy, room)) in parts.iter_mut().zip(&found).zip(copies.into_iter().zip(rooms)) {
        let mut left = room;
        for c in copy.iter_mut().filter(|c| **c) {
            *c = left > 0;
            left = left.saturating_sub(1);
        }
        let mut take = vec![false; p.cache.experts()];
        for (m, &c) in misses.iter().zip(&copy) {
            take[m.expert as usize] = c;
        }
        let wanted: Vec<u16> = p.needed.iter().copied().filter(|&e| take[e as usize] || p.cache.slot_of(e as usize).is_some()).collect();
        let got = p.cache.request(&wanted)?;
        let (mut kept, mut hot) = (Vec::new(), Vec::new());
        for (m, &c) in misses.iter().zip(&copy) {
            if !c {
                kept.push(m.expert);
                if m.hot && hot.len() < promote {
                    hot.push(m.expert);
                }
                p.last[m.expert as usize] = p.now;
            }
        }
        promote -= hot.len();
        served.push(Served { wanted, loads: got.loads, hits: got.hits, misses: misses.len(), kept, promote: hot });
    }
    Ok(served)
}

/// A group of a layer's call (`place_groups`): its cache, the calls its experts were kept in and its call's number, and the time of one of its copies as a share of `costs.copy`.
pub struct Slice<'a> {
    pub cache: &'a mut LayerCache,
    pub last: &'a mut [u64],
    pub now: u64,
    pub size: f64,
}

/// What the host decides for a call of a layer in groups (`place_groups`): for each group the experts the call needs of it (in its numbers, in the router's order), what became of them (`Served`), each pair's number in
/// the group (-1: a pair of another group) and its place there (-1: no place), and the pairs the host computes (group, pair), in the pairs' order.
#[derive(Debug, Default)]
pub struct Placed {
    pub needed: [Vec<u16>; 2],
    pub served: [Option<Served>; 2],
    pub local: [Vec<i64>; 2],
    pub ways: [Vec<i64>; 2],
    pub mine: Vec<(usize, usize)>,
}

/// A call of a layer in groups, on the host alone: the router's choices `host_ids` (the model's expert numbers; `group_of` and `local_of` say each one's group and its number there) are sorted into the groups that are
/// in the store (a `Slice` each, None: not), the groups' misses are planned together (`serve_groups`), and each pair of a group in the store is left with a place in its group's cache or to the host. A pair of a group that
/// is not in the store has neither: the layer runs it elsewhere.
pub fn place_groups(groups: &mut [Option<Slice>; 2], group_of: &[u8], local_of: &[u16], host_ids: &[i64], off: &Offload, costs: &Costs) -> Result<Placed> {
    let (pairs, experts) = (host_ids.len(), group_of.len().min(local_of.len()));
    let mut out = Placed::default();
    for (g, local) in out.local.iter_mut().enumerate() {
        *local =
            host_ids
                .iter()
                .map(|&e| {
                    if groups[g].is_some() && e >= 0 && (e as usize) < experts && group_of[e as usize] as usize == g {
                        local_of[e as usize] as i64
                    } else {
                        -1
                    }
                })
                .collect();
    }
    let mut pairs_of: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for g in 0..2 {
        if let Some(sl) = &groups[g] {
            out.needed[g] = distinct(&out.local[g], sl.cache.experts());
            pairs_of[g] = vec![0; sl.cache.experts()];
            out.local[g].iter().filter(|&&l| l >= 0).for_each(|&l| pairs_of[g][l as usize] += 1);
        }
    }
    let present: Vec<usize> = (0..2).filter(|&g| groups[g].is_some()).collect();
    let served = {
        let mut parts: Vec<Part> = Vec::new();
        for (g, sl) in groups.iter_mut().enumerate() {
            if let Some(sl) = sl.as_mut() {
                parts.push(Part { cache: &mut *sl.cache, needed: &out.needed[g], pairs_of: &pairs_of[g], last: &mut *sl.last, now: sl.now, size: sl.size });
            }
        }
        serve_groups(&mut parts, off, costs)?
    };
    for (g, sv) in present.iter().copied().zip(served) {
        let cache = &groups[g].as_ref().expect("a group").cache;
        let mut place = vec![-1i64; cache.experts()];
        for &e in &sv.wanted {
            place[e as usize] = cache.slot_of(e as usize).expect("an expert in a place") as i64;
        }
        out.ways[g] = vec![-1; pairs];
        fill_way(&out.local[g], &place, &mut out.ways[g]);
        out.served[g] = Some(sv);
    }
    for g in 0..2 {
        out.mine.extend((0..pairs).filter(|&j| out.local[g][j] >= 0 && out.ways[g][j] < 0).map(|j| (g, j)));
    }
    out.mine.sort_by_key(|&(_, j)| j);
    Ok(out)
}

// ---- what the cache costs --------------------------------------------------------------------------------------------------------------------------------------------------------------

/// An expert cache's size: places a layer, over `layers` layers of `experts` experts of `expert_bytes` (the codes and scales of all its matrices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachePlan {
    pub layers: usize,
    pub experts: usize,
    pub slots: usize,
    pub expert_bytes: u64,
    /// Whether the host computes an expert from its place in the host tier as it is (the Hummingbird level's codebook rows), and needs no second copy of any matrix.
    pub in_place: bool,
    /// The bytes of the experts the computer's memory holds (pinned): all of them, `host_bytes`, unless the Hummingbird level is given less (`--host-memory`: the rest stay in the packed directory and are read as the router
    /// asks for them, `Tier`).
    pub ram: u64,
}

impl CachePlan {
    /// The GPU's memory the cache takes.
    pub fn gpu_bytes(&self) -> u64 {
        (self.layers * self.slots) as u64 * self.expert_bytes
    }

    /// The bytes of the experts of all the layers.
    pub fn host_bytes(&self) -> u64 {
        (self.layers * self.experts) as u64 * self.expert_bytes
    }

    /// The bytes of the experts that stay in the packed directory (none where the computer's memory holds them all).
    pub fn disk_bytes(&self) -> u64 {
        self.host_bytes().saturating_sub(self.ram)
    }

    /// The host memory (not pinned) the host's half takes besides: the first matrices again, in the row layout (an expert's two matrices are the same bytes in either layout, so half the experts'); none where the host computes from the host tier in place.
    pub fn host_half_bytes(&self) -> u64 {
        if self.in_place {
            0
        } else {
            self.host_bytes() / 2
        }
    }

    /// Whether a computer with `host` bytes of memory to give (None: it does not say) holds the experts it is to hold (`ram`), the host's half and the `margin` the system and the engine keep. A host half that takes nothing
    /// besides the experts always fits: the plan gave the experts the computer's memory less the margin, and asking again a moment later (the memory free has moved a little) must not take it back.
    pub fn host_half_fits(&self, host: Option<u64>, margin: u64) -> bool {
        self.host_half_bytes() == 0 || host.is_none_or(|h| h >= self.ram + self.host_half_bytes() + margin)
    }
}

/// The places a layer gets out of a budget: the planner's (src/plan.rs `sizing`).
pub use crate::plan::sizing::places_for;

/// `MemAvailable` of a /proc/meminfo's text, in bytes.
pub fn parse_meminfo(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok()?.checked_mul(1024)
}

/// What a control group has free for a process: its limit less what it holds besides the file cache the kernel gives back when asked (`stat`, its memory.stat: `inactive_file` and `active_file`; version 1 lists these
/// for the group alone and, as `total_inactive_file` and `total_active_file`, with its children, which are the ones taken where they are there). A container that has read and written a few hundred GB holds that much
/// cache and no room, to a reader of `current` alone.
pub fn cgroup_free(limit: u64, current: u64, stat: &str) -> u64 {
    let get = |k: &str| stat.lines().find_map(|l| l.strip_prefix(k).and_then(|r| r.strip_prefix(' ')).and_then(|r| r.trim().parse::<u64>().ok()));
    let cache = match (get("total_inactive_file"), get("total_active_file")) {
        (None, None) => get("inactive_file").unwrap_or(0) + get("active_file").unwrap_or(0),
        (i, a) => i.unwrap_or(0) + a.unwrap_or(0),
    };
    limit.saturating_sub(current.saturating_sub(cache))
}

/// The host memory a process can still take: what /proc/meminfo says is available, or less where a control group (v2 or v1) limits it below that (`cgroup_free`). None where the system does not say (not Linux).
/// The one probe of it (#137: Condor's server had its own): every path that plans memory reads it here. GLYD_MEMAVAIL_GB (a development build's): a test's stand-in for a smaller computer.
pub fn host_memory() -> Option<u64> {
    if let Some(g) = crate::tune!("GLYD_MEMAVAIL_GB").and_then(|v| v.parse::<f64>().ok()) {
        return Some((g * (1u64 << 30) as f64) as u64);
    }
    let read = |p: &str| std::fs::read_to_string(p).ok();
    let mut free = parse_meminfo(&crate::os::meminfo()?)?;
    let number = |p: &str| read(p).and_then(|t| t.trim().parse::<u64>().ok());
    let (v2, v1) = ("/sys/fs/cgroup", "/sys/fs/cgroup/memory");
    // (v2: this process's own cgroup, /proc/self/cgroup's "0::/path", and its parents, the nearest with a limit: a systemd unit's MemoryMax, a CI runner's, is below the
    // root; in a container the root is the container's. "max", no limit, does not parse)
    let own = read("/proc/self/cgroup").and_then(|t| t.lines().find_map(|l| l.strip_prefix("0::").map(str::to_string))).unwrap_or_default();
    let limited = std::iter::successors(Some(std::path::Path::new(v2).join(own.trim_start_matches('/'))), |d| {
        d.parent().filter(|p| p.starts_with(v2)).map(std::path::Path::to_path_buf)
    })
    .find_map(|d| number(&d.join("memory.max").to_string_lossy()).map(|l| (l, d)));
    let group = match (limited, number(&format!("{v1}/memory.limit_in_bytes"))) {
        (Some((l, d)), _) => Some((l, number(&d.join("memory.current").to_string_lossy()), read(&d.join("memory.stat").to_string_lossy()))),
        (None, Some(l)) => Some((l, number(&format!("{v1}/memory.usage_in_bytes")), read(&format!("{v1}/memory.stat")))),
        _ => None,
    };
    if let Some((limit, Some(current), stat)) = group {
        free = free.min(cgroup_free(limit, current, stat.as_deref().unwrap_or("")));
    }
    Some(free)
}

// ---- the store: the host tier, the places on the GPU, the copy stream ----------------------------------------------------------------------------------------------------------------------------

/// What the cache has done: reads of experts (a call reads each expert it needs once), the ones found in a place (`hits`) and not (`misses`: copied in before the call's products, `copied`, or computed by the host,
/// `cpu`, for `cpu_pairs` pairs in `cpu_ms` of its calls), experts copied for a predictor (`hinted`) or behind the host's (`promoted`), the bytes copied and the time the copy stream spent copying them, the waves a call
/// needed beyond its first, and the time the host waited for the GPU at the start of a call (`wait_ms`). With a disk tier: the experts the host tier's slots held when they were wanted (`ram_hits`), and those read from
/// the packed directory (`disk_reads`, `disk_bytes`, in `disk_ms` of the host's waiting for the reads of its calls).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub reads: u64,
    pub hits: u64,
    pub misses: u64,
    pub copied: u64,
    pub cpu: u64,
    pub cpu_pairs: u64,
    pub cpu_ms: f64,
    pub hinted: u64,
    pub promoted: u64,
    pub bytes: u64,
    pub copy_ms: f64,
    pub waves: u64,
    pub wait_ms: f64,
    pub ram_hits: u64,
    pub disk_reads: u64,
    pub disk_bytes: u64,
    pub disk_ms: f64,
    /// Pairs of the host's left out for their small routing weights (`Offload::skip`).
    pub skipped: u64,
}

impl Stats {
    pub fn hit_rate(&self) -> f64 {
        if self.reads == 0 {
            0.0
        } else {
            self.hits as f64 / self.reads as f64
        }
    }

    /// Gigabytes a second the copy stream moved the bytes at, while it copied.
    pub fn gbs(&self) -> f64 {
        if self.copy_ms <= 0.0 {
            0.0
        } else {
            self.bytes as f64 / 1e9 / (self.copy_ms / 1e3)
        }
    }

    /// Gigabytes a second the experts were read from the disk at, while the host waited for the reads.
    pub fn disk_gbs(&self) -> f64 {
        if self.disk_ms <= 0.0 {
            0.0
        } else {
            self.disk_bytes as f64 / 1e9 / (self.disk_ms / 1e3)
        }
    }
}

/// What the store did for one of its layers over a step (`ExpertStore::end_step`) or over a run (`ExpertStore::totals`; docs/design/runtime.md §6): the layer's calls and their rows and pairs, the experts they asked
/// for (each once a call), those found in a place, the misses copied in before the products (`copies`) and those whose pairs the host computed (`host` experts, `host_pairs` pairs), the pairs left out for their small
/// weights, the waves past a call's first; the bytes read from the computer's memory for the misses (`copy_bytes` over the link, `host_bytes` by the host's kernels: an expert's bytes once for all its rows where the
/// kernel groups its pairs by expert, once a pair where it does not), and for the experts copied behind the calls for later ones (promotions, hints: `ahead`, `ahead_bytes`); the copy stream's time for the layer's
/// copies, the host's for its pairs, the host's waits for the GPU (the router's choices, a wave's end), and the call's own time on the host besides (`plan_ms`: the plan, the copies and products queued, the
/// host's sums sent). `whole` of the calls had no host half (every miss copied before the products, in waves where they outnumber the places: copy mode, or `waves_quicker`'s choice) and `whole_bytes` of the copies
/// were theirs: a layer whose calls are some of each is not their average (`ceiling --stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerStats {
    pub calls: u64,
    pub whole: u64,
    pub whole_bytes: u64,
    pub rows: u64,
    pub pairs: u64,
    pub reads: u64,
    pub hits: u64,
    pub copies: u64,
    pub host: u64,
    pub host_pairs: u64,
    pub skipped: u64,
    pub waves: u64,
    pub copy_bytes: u64,
    pub host_bytes: u64,
    pub missed_bytes: u64,
    pub ahead: u64,
    pub ahead_bytes: u64,
    pub copy_ms: f64,
    pub host_ms: f64,
    pub wait_ms: f64,
    pub plan_ms: f64,
}

impl LayerStats {
    pub fn add(&mut self, o: &LayerStats) {
        let LayerStats {
            calls,
            whole,
            whole_bytes,
            rows,
            pairs,
            reads,
            hits,
            copies,
            host,
            host_pairs,
            skipped,
            waves,
            copy_bytes,
            host_bytes,
            missed_bytes,
            ahead,
            ahead_bytes,
            copy_ms,
            host_ms,
            wait_ms,
            plan_ms,
        } = *o;
        self.calls += calls;
        self.whole += whole;
        self.whole_bytes += whole_bytes;
        self.rows += rows;
        self.pairs += pairs;
        self.reads += reads;
        self.hits += hits;
        self.copies += copies;
        self.host += host;
        self.host_pairs += host_pairs;
        self.skipped += skipped;
        self.waves += waves;
        self.copy_bytes += copy_bytes;
        self.host_bytes += host_bytes;
        self.missed_bytes += missed_bytes;
        self.ahead += ahead;
        self.ahead_bytes += ahead_bytes;
        self.copy_ms += copy_ms;
        self.host_ms += host_ms;
        self.wait_ms += wait_ms;
        self.plan_ms += plan_ms;
    }

    /// A call of `tokens` rows and `pairs` pairs, whose choices the host waited `waited` ms for.
    fn called(&mut self, tokens: usize, pairs: usize, waited: f64) {
        self.calls += 1;
        self.rows += tokens as u64;
        self.pairs += pairs as u64;
        self.wait_ms += waited;
    }

    /// `n` experts of `bytes` copied in behind a call, for later ones (promotions, hints).
    fn ahead(&mut self, n: usize, bytes: u64) {
        self.ahead += n as u64;
        self.ahead_bytes += bytes;
    }

    /// What a call's plan did (`serve`'s): the experts it asked for, those in places, the misses copied before its products (`bytes` of them) and those the host keeps; `missed` the distinct missed experts' own bytes.
    fn served(&mut self, sv: &Served, bytes: u64, missed: u64) {
        self.reads += (sv.hits + sv.misses) as u64;
        self.hits += sv.hits as u64;
        self.copies += sv.loads.len() as u64;
        self.copy_bytes += bytes;
        self.missed_bytes += missed;
    }

    /// The host's part of a call: `experts` experts for `pairs` pairs (`skipped` left out), `bytes` read by its kernels, in `ms`.
    fn hosted(&mut self, experts: usize, pairs: usize, skipped: usize, bytes: u64, ms: f64) {
        self.host += experts as u64;
        self.host_pairs += pairs as u64;
        self.skipped += skipped as u64;
        self.host_bytes += bytes;
        self.host_ms += ms;
    }

    /// The experts asked for that were in no place (each once a call).
    pub fn misses(&self) -> u64 {
        self.reads - self.hits
    }

    /// The bytes read from the computer's memory for the calls' misses (copied, or read by the host's kernels): `missed_bytes` (the distinct missed experts' own bytes) where each is read once.
    pub fn miss_bytes(&self) -> u64 {
        self.copy_bytes + self.host_bytes
    }

    /// The stats summed over layers.
    pub fn sum<'a>(all: impl IntoIterator<Item = &'a LayerStats>) -> LayerStats {
        let mut t = LayerStats::default();
        all.into_iter().for_each(|l| t.add(l));
        t
    }

    /// The stats as a JSON object (the layer's number among the store's first).
    pub fn json(&self, layer: usize) -> String {
        format!(
            "{{\"layer\":{layer},\"calls\":{},\"whole\":{},\"whole_bytes\":{},\"rows\":{},\"pairs\":{},\"reads\":{},\"hits\":{},\"copies\":{},\"host\":{},\"host_pairs\":{},\"skipped\":{},\"waves\":{},\"copy_bytes\":{},\"host_bytes\":{},\"missed_bytes\":{},\"ahead\":{},\"ahead_bytes\":{},\"copy_ms\":{:.3},\"host_ms\":{:.3},\"wait_ms\":{:.3},\"plan_ms\":{:.3}}}",
            self.calls,
            self.whole,
            self.whole_bytes,
            self.rows,
            self.pairs,
            self.reads,
            self.hits,
            self.copies,
            self.host,
            self.host_pairs,
            self.skipped,
            self.waves,
            self.copy_bytes,
            self.host_bytes,
            self.missed_bytes,
            self.ahead,
            self.ahead_bytes,
            self.copy_ms,
            self.host_ms,
            self.wait_ms,
            self.plan_ms
        )
    }
}

/// Every store's layers' stats since the process started, by the layer's number in its store (`ExpertStore::end_step` adds a step's): what the server's /stats reports. One model's stores a process (two models
/// with experts in a cache would add up layer by layer).
static TOTALS: Mutex<Vec<LayerStats>> = Mutex::new(Vec::new());

/// The layers' stats of every store since the process started (`TOTALS`).
pub fn process_totals() -> Vec<LayerStats> {
    TOTALS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// `process_totals` as /stats's field: `"experts":[...]` (a JSON object a layer, `LayerStats::json`), or nothing where no layer has run a call.
pub fn stats_field() -> String {
    let t = process_totals();
    if t.iter().all(|l| l.calls == 0) {
        return String::new();
    }
    format!(",\"experts\":[{}]", t.iter().enumerate().map(|(l, s)| s.json(l)).collect::<Vec<_>>().join(","))
}

/// A matrix of a layer's experts to hand to the store: the bytes of every expert end to end (the codes, the scales), the bytes of one expert in each, and the addresses of the places for it on the GPU (the first
/// place's codes and scales).
pub struct PackIn<'a> {
    pub bytes: [&'a [u8]; 2],
    pub stride: [u64; 2],
    pub dev: [u64; 2],
}

/// A layer's first matrices in the row layout of `bits` (`cpu_experts::rows_of_experts`), every expert's `inter` neurons of `hidden` weights end to end: with the second matrices (the layer's second pack, which must be
/// in the row layout too) the bytes the host computes the layer's experts from.
pub struct CpuRows {
    pub bits: u32,
    pub inter: usize,
    pub hidden: usize,
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
}

/// A layer's matrix the caller packed in pinned memory (`ExpertStore::add_layer_pinned`): its codes and its scales, every expert end to end at `stride`, and the
/// places' device addresses (as `PackIn`'s).
pub struct PinnedPack {
    pub data: Pinned,
    pub scales: Pinned,
    pub stride: [u64; 2],
    pub dev: [u64; 2],
}

/// Host memory the experts are read from: pinned memory the store owns, or the bytes of a mapped file (not pinned: the pages are the page cache's, which the system takes back under pressure, and a copy from them is
/// the driver's staged one).
pub struct HostMem {
    pub ptr: *mut std::ffi::c_void,
    _own: HostOwner,
}

#[allow(dead_code)]
enum HostOwner {
    None,
    Pinned(Pinned),
    Mapped(Arc<crate::mapped::Mapping>, usize),
}

impl From<Pinned> for HostMem {
    fn from(p: Pinned) -> HostMem {
        HostMem { ptr: p.ptr, _own: HostOwner::Pinned(p) }
    }
}

impl HostMem {
    /// The bytes of `map` from `offset` on.
    /// No memory of its own: the host tier of a layer whose placement holds its experts' addresses (`Placement::maps`).
    pub fn none() -> HostMem {
        HostMem { ptr: std::ptr::null_mut(), _own: HostOwner::None }
    }

    pub fn mapped(map: &Arc<crate::mapped::Mapping>, offset: usize) -> HostMem {
        // SAFETY: the offset is inside the mapping (the caller's), which the value keeps alive.
        HostMem { ptr: unsafe { (map.ptr as *mut u8).add(offset) as *mut std::ffi::c_void }, _own: HostOwner::Mapped(map.clone(), offset) }
    }

    /// Asks the system to read `len` bytes from `at` (from `ptr` on) ahead of their use, in large requests, where they are a mapped file's (one call for an expert: the pages are then read together instead of one fault at a time).
    pub fn will_need(&self, at: usize, len: usize) {
        if let HostOwner::Mapped(map, base) = &self._own {
            map.will_need(base + at, len);
        }
    }
}

/// The same in the host tier: pinned or mapped, an expert's at `expert * stride` and the place `p`'s at `dev + p * stride`.
struct HostPack {
    stride: [u64; 2],
    dev: [u64; 2],
    host: [HostMem; 2],
    /// Where each expert's bytes are, where they are not `expert * stride` apart (a layer of mixed formats).
    place: Option<Placement>,
}

/// The experts of a layer whose formats differ (GLM-5.3-Flash's `glyd-hbr-codes/2` packs: cb2, cb3, cb4 and ternary rows, a matrix's format of its own): for each expert the offset and the length of its codes and of its
/// codebooks in the host tier (the stride of the layer's places is the biggest expert's: an expert copies only its own length, to the start of its place), the formats of its up and its down for the host's kernels, and
/// the byte the kernels read the formats from, which is copied to the last byte of the place's codebooks.
pub struct Placement {
    pub at: [Vec<u64>; 2],
    pub len: [Vec<u32>; 2],
    pub fmts: Vec<(crate::cpu_cb::Fmt, crate::cpu_cb::Fmt)>,
    pub marks: Vec<u8>,
    /// Where `at` holds addresses (the experts in more than one file: a pack and its sidecar), the files they are in (kept mapped as long as the layer is); none: `at` holds offsets from the host tier's start.
    pub maps: Vec<Arc<crate::mapped::Mapping>>,
}

impl HostPack {
    /// The address of expert `e`'s bytes of stream `k`.
    fn addr(&self, k: usize, e: usize) -> *const u8 {
        match &self.place {
            Some(p) if !p.maps.is_empty() => p.at[k][e] as usize as *const u8,
            _ => (self.host[k].ptr as usize + self.at(k, e)) as *const u8,
        }
    }

    /// Asks the system to read expert `e`'s bytes of stream `k` ahead of their use (where they are a mapped file's).
    fn will_need(&self, k: usize, e: usize) {
        match &self.place {
            Some(p) if !p.maps.is_empty() => {
                let a = p.at[k][e] as usize;
                if let Some(m) = p.maps.iter().find(|m| (m.ptr as usize..m.ptr as usize + m.len).contains(&a)) {
                    m.will_need(a - m.ptr as usize, p.len[k][e] as usize);
                }
            }
            _ => self.host[k].will_need(self.at(k, e), self.len(k, e)),
        }
    }

    /// Where expert `e`'s bytes of stream `k` start in the host tier, from `host[k]`.
    fn at(&self, k: usize, e: usize) -> usize {
        match &self.place {
            Some(p) => p.at[k][e] as usize,
            None => e * self.stride[k] as usize,
        }
    }

    /// How many bytes expert `e` has in stream `k`.
    fn len(&self, k: usize, e: usize) -> usize {
        match &self.place {
            Some(p) => p.len[k][e] as usize,
            None => self.stride[k] as usize,
        }
    }

    /// The bytes of expert `e` in both streams.
    fn bytes_of(&self, e: usize) -> u64 {
        (self.len(0, e) + self.len(1, e)) as u64
    }
}

/// What the host computes a layer's experts from: the first matrices in the row layout (`CpuRows`; the second ones are the host tier's own bytes), or the places of a Hummingbird group as the host tier holds them (`HbPlace`: both
/// matrices are neuron rows there, and the host reads them where they are).
pub enum HostHalf {
    Rows(CpuRows),
    Hb(HbPlace),
    /// Experts as codebook rows (src/cpu_cb.rs; the clamp of the SwiGLU, the width and the neurons: GLM-5.3-Flash's 4,096 and 2,048, Qwen3.8-Flash-Next's 2,560 and 640): the host tier holds, for each expert, the codes
    /// and then the codebooks of its downT and its up as the pack's files do.
    Cb2(f32, usize, usize),
    /// Experts in the exact FP4 block layouts (Kimi-K3's, DeepSeek-V4's as released), read where the host tier holds them (src/cpu_fp4.rs): the layer's two packs, the first [2 inter,
    /// latent] (gate then up), the second [latent, inter]; `packed` the Fp4R layout's scales; `act` the activation and where the weight goes, as the layer's GPU kernels have them.
    Fp4 {
        inter: usize,
        latent: usize,
        packed: bool,
        act: crate::cpu_fp4::Act,
    },
    /// Hummingbird's experts of every mixture of experts (src/moe_hbg.rs: a group's places, `HbgPlace`, as the host tier holds them, read in place): each pair's two products with the family's activation `act`, finished as
    /// `finish` says (src/cpu_hbg.rs `store_pairs`: weighted in the rotated basis, or normed in the model's for Muse Spark).
    Hbg {
        place: HbgPlace,
        act: crate::moe::Act,
        finish: crate::cpu_hbg::Finish,
    },
}

/// The GPU a call of the store runs on: its number in the driver's list and the engine's stream there (what the store needs of a `Dev`; a layer that holds no `Dev`, `moe::MoeLayer`, gives these).
#[derive(Clone, Copy, Debug)]
pub struct At {
    pub ordinal: i32,
    pub stream: Stream,
}

impl At {
    pub fn of(dev: &Dev) -> At {
        At { ordinal: dev.ordinal, stream: dev.stream }
    }

    fn sync(&self) -> Result<()> {
        crate::dev::sync_stream(self.stream)
    }
}

/// The host's half of a layer, and the call each expert was last computed in (0: never).
struct CpuSide {
    half: HostHalf,
    last: Vec<u64>,
}

/// Reads expert `e` of a group (its number in the group) from the packed directory into the bytes of a slot of the host tier, `data` and `meta` as a place holds an expert (`HbPlace`: the codes, then the scales and the
/// codebooks); called on the threads of the disk pool (src/disk.rs).
pub type Reader = Arc<dyn Fn(usize, &mut [u8], &mut [u8]) -> Result<()> + Send + Sync>;

/// Where the slots of a host tier are: the memory of the codes and of the scales and codebooks, and the bytes of a slot in each.
#[derive(Clone, Copy)]
struct Slots {
    base: [*mut u8; 2],
    stride: [usize; 2],
}

/// A slot's address, handed to a thread of the disk pool: the caller waits for every read before it returns, so the memory outlives them, and no two reads are of one slot.
struct Raw(*mut u8);

// SAFETY: see above; the pointer is used by the one read it was made for.
unsafe impl Send for Raw {}

impl Raw {
    fn get(&self) -> *mut u8 {
        self.0
    }
}

/// What a request of experts of the disk tier found.
#[derive(Debug)]
struct Staged {
    hits: usize,
    reads: usize,
    ms: f64,
}

/// Reads under way into a tier's slots (`Tier::begin`); dropped, it waits for them (the memory they write may be about to go).
struct Staging {
    hits: usize,
    reads: usize,
    t0: Instant,
    pending: disk::Pending,
}

/// A group's experts when the computer's memory holds only some of them: `ram` says which expert is in which slot of the host tier (the same policy as the GPU's places, over the experts the GPU missed), `read` brings
/// an expert in from the packed directory, `busy` is for each slot the number of the last batch of copies queued that reads it (`Cuda::issued` then; 0: none), and `seed` the experts the slots start with.
struct Tier {
    ram: LayerCache,
    read: Reader,
    busy: Vec<u64>,
    seed: Vec<u16>,
}

impl Tier {
    /// A tier of `slots` slots for a group of `experts`, empty; its first experts are what `seed` puts in.
    fn new(experts: usize, slots: usize, read: Reader) -> Tier {
        Tier { ram: LayerCache::new(experts, slots), read, busy: vec![0; slots], seed: (0..slots as u16).collect() }
    }

    /// The slots start with the group's first experts, read from the packed directory (the group's order is the pack's; there is no record of which are most used within a width).
    fn seed(&mut self, mem: Slots) -> Result<()> {
        let got = self.ram.request(&self.seed)?;
        self.read_all(&got.loads, mem)
    }

    /// The slots as `seed` left them: an expert that left is read again, one in the slot it started in is not (a cold start of a measurement).
    fn reseed(&mut self, mem: Slots) -> Result<()> {
        let fresh = LayerCache::new(self.ram.experts(), self.ram.slots());
        let old = std::mem::replace(&mut self.ram, fresh);
        let got = self.ram.request(&self.seed)?;
        let loads: Vec<(u16, u32)> = got.loads.into_iter().filter(|&(e, s)| old.slot_of(e as usize) != Some(s as usize)).collect();
        self.read_all(&loads, mem)
    }

    /// Experts `wanted` (distinct, at most the slots) in slots, the most recently used: the ones that are not are read from the packed directory into the slots of the least recently used experts that the request does not
    /// need. A slot that a queued copy has yet to read (`busy` past `retired`, the batches the copy stream is known to have run) is not written before `wait` has had the copy stream run them (it returns the batches run then).
    /// A read that fails leaves the tier empty: what its slots hold is not known.
    fn stage(&mut self, wanted: &[u16], mem: Slots, retired: u64, wait: &mut dyn FnMut() -> Result<u64>) -> Result<Staged> {
        let s = self.begin(wanted, mem, retired, wait)?;
        self.finish(s)
    }

    /// `stage` started: the experts are given their slots and the reads are under way (on the disk pool's threads) while the caller goes on; `finish` waits for them. Two tiers' reads can be under way at once.
    fn begin(&mut self, wanted: &[u16], mem: Slots, retired: u64, wait: &mut dyn FnMut() -> Result<u64>) -> Result<Staging> {
        let got = self.ram.request(wanted)?;
        let t0 = Instant::now();
        let started = (|| {
            if got.loads.iter().any(|&(_, s)| self.busy[s as usize] > retired) {
                wait()?;
            }
            self.start_all(&got.loads, mem)
        })();
        match started {
            Ok(pending) => Ok(Staging { hits: got.hits, reads: got.loads.len(), t0, pending }),
            Err(e) => {
                self.forget();
                Err(e)
            }
        }
    }

    /// The reads of `begin` done: what was found and read, and the time since the start.
    fn finish(&mut self, s: Staging) -> Result<Staged> {
        let Staging { hits, reads, t0, pending } = s;
        if let Err(e) = pending.wait().map_err(|e| Error(format!("reading experts from the packed directory: {e}"))) {
            self.forget();
            return Err(e);
        }
        Ok(Staged { hits, reads, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    /// The tier empty: what its slots hold is not known (a read failed).
    fn forget(&mut self) {
        self.ram = LayerCache::new(self.ram.experts(), self.ram.slots());
    }

    /// The experts of `loads` ((expert, slot)) read into their slots, 16 at a time, all done when this returns.
    fn read_all(&self, loads: &[(u16, u32)], mem: Slots) -> Result<()> {
        self.start_all(loads, mem)?.wait().map_err(|e| Error(format!("reading experts from the packed directory: {e}")))
    }

    /// The reads of `loads` ((expert, slot)) started on the disk pool.
    fn start_all(&self, loads: &[(u16, u32)], mem: Slots) -> Result<disk::Pending> {
        let jobs: Vec<Job> = loads
            .iter()
            .map(|&(e, s)| {
                // SAFETY: the slot is one of the tier's (the cache has as many slots as the memory holds), so both ranges are inside the memory.
                let (d, m) = unsafe { (Raw(mem.base[0].add(s as usize * mem.stride[0])), Raw(mem.base[1].add(s as usize * mem.stride[1]))) };
                let (read, lens) = (self.read.clone(), mem.stride);
                Box::new(move || {
                    // SAFETY: the slot's bytes, which nothing else reads or writes until the reads are done (`Staging` and `Pending` wait for them before the memory can go).
                    let (data, meta) = unsafe { (std::slice::from_raw_parts_mut(d.get(), lens[0]), std::slice::from_raw_parts_mut(m.get(), lens[1])) };
                    read(e as usize, data, meta).map_err(|x| std::io::Error::other(x.msg))
                }) as Job
            })
            .collect();
        disk::pool().start(jobs).map_err(|e| Error(format!("reading experts from the packed directory: {e}")))
    }
}

struct Resident {
    cache: LayerCache,
    packs: Vec<HostPack>,
    // the copies of the layer's last call or hint are waited for by the engine's stream before the products read the places (`ready`), and the engine's stream's last use of the places is waited for by the copy
    // stream before it writes them (`done`)
    ready: Ev,
    done: Ev,
    pending: bool,
    // the copy thread's last job for the layer, until the engine's stream has been told to wait for it
    job: Option<Arc<JobState>>,
    // the place of each expert in the wave being run, -1 for the others
    place: Vec<i64>,
    cpu: Option<CpuSide>,
    // the layer's calls so far
    calls: u64,
    // the disk tier, where the host tier holds some of the experts only (its `host` buffers then hold the slots, not every expert)
    tier: Option<Tier>,
    // the layer's number in the store, and what it has done since the step began (`ExpertStore::end_step`)
    index: usize,
    stats: LayerStats,
    // the layer's places as the GPU plans its decode calls (src/devplan.rs; `Offload::device`), None where the host plans every call
    dplan: Option<crate::devplan::DevPlan>,
}

struct Cuda {
    copy: Side,
    ids: Pinned,
    // the places of the pairs: a section of `pairs` int64 for each group of a layer's call and one more for the sum's ids (`WAYS`)
    way: Pinned,
    // the router's weights, the experts' input and the host's sums of a call, where the host computes some of its experts
    wts: Pinned,
    z: Pinned,
    part: Pinned,
    out: Vec<f32>,
    // (start, stop, bytes, the copy thread's job, the layer) of the batches of copies not yet counted (a job's events are recorded when it is done submitting), and events to make more of
    timing: VecDeque<(Ev, Ev, u64, Option<Arc<JobState>>, usize)>,
    spare: Vec<Ev>,
    // the copy stream's time for each layer's batches counted since the step began
    layer_ms: Vec<f64>,
    // the batches of copies queued so far, and the ones counted (known to have run): a slot of a disk tier read by batch n is free to write once `retired` is n or more
    issued: u64,
    retired: u64,
    // the planner's times are held at their starting values (`ExpertStore::hold_costs`)
    frozen: bool,
    // the copy thread (made at the first copy that is to go through it) and whether copies go through it now; the device it binds to, known from the first call
    worker: Option<Worker>,
    use_worker: bool,
    ordinal: Option<i32>,
    // the thread the host's pairs of a call are computed from (it takes part in the crew's job; kept, not made for each call)
    hands: Option<Sender<Box<dyn FnOnce() + Send>>>,
    // the host's buffers are named by graphs (`parts_host`): they may not be made again
    fixed: bool,
}

// ---- the copy thread -------------------------------------------------------------------------------------------------------------------------------------------------------------

/// How a job of the copy thread stands: not yet submitted (None), or submitted with its error, if any.
struct JobState {
    done: Mutex<Option<std::result::Result<(), String>>>,
    cv: Condvar,
}

impl JobState {
    fn new() -> Arc<JobState> {
        Arc::new(JobState { done: Mutex::new(None), cv: Condvar::new() })
    }

    fn finish(&self, r: std::result::Result<(), String>) {
        *self.done.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        self.cv.notify_all();
    }

    fn submitted(&self) -> bool {
        self.done.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Waits until the copies are queued and the events recorded (the copies themselves run on the copy stream).
    fn wait(&self) -> Result<()> {
        let mut g = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while g.is_none() {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        match g.as_ref() {
            Some(Ok(())) => Ok(()),
            Some(Err(e)) => err(e.clone()),
            None => Ok(()),
        }
    }
}

/// A batch of an layer's copies for the copy thread: (device address, host address, bytes) of each, the copy stream, the event of the engine's last use of the places that the stream waits for first, the batch's two
/// timing events and the layer's `ready` to record after it.
struct CopyJob {
    copies: Vec<(u64, usize, usize)>,
    stream: Stream,
    wait: EvRef,
    start: EvRef,
    stop: EvRef,
    ready: EvRef,
    state: Arc<JobState>,
}

// SAFETY: the stream and the events are handles the driver takes from any thread, and the host addresses are of memory that outlives the job (the host tier's mapped file or pinned slots, which the owner of the jobs
// waits for before it frees them: `Worker::drain`).
unsafe impl Send for CopyJob {}

struct Worker {
    tx: Option<Sender<CopyJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
    // the jobs sent and the ones done
    left: Arc<(Mutex<usize>, Condvar)>,
}

/// The next closure for `hands`: looked for for a short while (a call comes every millisecond or two in a decode), then slept for.
fn hands_thread(rx: Receiver<Box<dyn FnOnce() + Send>>) {
    use std::sync::mpsc::TryRecvError;
    loop {
        let t0 = Instant::now();
        let job = loop {
            match rx.try_recv() {
                Ok(j) => break Some(j),
                Err(TryRecvError::Disconnected) => break None,
                Err(TryRecvError::Empty) => {
                    if t0.elapsed().as_micros() > 2000 {
                        break rx.recv().ok();
                    }
                    std::hint::spin_loop();
                }
            }
        };
        match job {
            Some(f) => f(),
            None => return,
        }
    }
}

/// The result of a closure sent to `hands`: looked for for a short while, then slept for.
fn recv_spin<T>(rx: &Receiver<T>) -> Result<T> {
    use std::sync::mpsc::TryRecvError;
    let t0 = Instant::now();
    loop {
        match rx.try_recv() {
            Ok(v) => return Ok(v),
            Err(TryRecvError::Disconnected) => return err("the host's pairs were lost"),
            Err(TryRecvError::Empty) => {
                if t0.elapsed().as_micros() > 1500 {
                    return rx.recv().map_err(|_| Error("the host's pairs were lost".into()));
                }
                std::hint::spin_loop();
            }
        }
    }
}

impl Cuda {
    /// The thread the host's pairs are computed on, made at the first call.
    fn hands(&mut self) -> Result<&Sender<Box<dyn FnOnce() + Send>>> {
        if self.hands.is_none() {
            let (tx, rx) = channel::<Box<dyn FnOnce() + Send>>();
            std::thread::Builder::new().name("glyd-host-pairs".into()).spawn(move || hands_thread(rx)).map_err(|e| Error(format!("the host's thread: {e}")))?;
            self.hands = Some(tx);
        }
        Ok(self.hands.as_ref().expect("made above"))
    }
}

fn copy_worker(ordinal: i32, rx: Receiver<CopyJob>, left: Arc<(Mutex<usize>, Condvar)>) {
    let bound = crate::dev::bind_thread(ordinal);
    while let Ok(job) = rx.recv() {
        let r = match &bound {
            Err(e) => Err(e.msg.clone()),
            Ok(()) => (|| -> Result<()> {
                job.wait.wait(job.stream)?;
                job.start.record(job.stream)?;
                for &(dst, src, len) in &job.copies {
                    // SAFETY: the host range is the host tier's, kept alive until the job is done (see `CopyJob`); the device range a place of the layer, which the stream waited for the engine's last use of.
                    unsafe { copy_h2d_async(dst, src as *const std::ffi::c_void, len, job.stream)? };
                }
                job.stop.record(job.stream)?;
                job.ready.record(job.stream)?;
                Ok(())
            })()
            .map_err(|e| e.msg),
        };
        job.state.finish(r);
        let (m, cv) = &*left;
        *m.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        cv.notify_all();
    }
}

impl Worker {
    fn new(ordinal: i32) -> Result<Worker> {
        let (tx, rx) = channel::<CopyJob>();
        let left = Arc::new((Mutex::new(0usize), Condvar::new()));
        let l2 = left.clone();
        let handle = std::thread::Builder::new()
            .name("glyd-expert-copies".into())
            .spawn(move || copy_worker(ordinal, rx, l2))
            .map_err(|e| Error(format!("the copy thread: {e}")))?;
        Ok(Worker { tx: Some(tx), handle: Some(handle), left })
    }

    fn submit(&self, job: CopyJob) -> Result<()> {
        *self.left.0.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        self.tx.as_ref().ok_or_else(|| Error("the copy thread is gone".into()))?.send(job).map_err(|_| Error("the copy thread is gone".into()))
    }

    /// Returns when every job sent has been submitted to the copy stream.
    fn drain(&self) {
        let (m, cv) = &*self.left;
        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
        while *g > 0 {
            g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// What a predictor is: given a layer and the experts its call asked for (distinct, in the router's order), the (layer, experts) it expects the layers ahead to be asked for, least expected first.
pub type Predictor = Box<dyn FnMut(usize, &[u16]) -> Vec<(usize, Vec<u16>)>>;

struct Inner {
    // the plan on the GPU's host thread and buffers (src/devplan.rs), dropped first: its thread reads the layers' host tiers
    mail: Option<crate::devplan::DevMail>,
    cuda: Option<Cuda>,
    layers: Vec<Option<Resident>>,
    stats: Stats,
    predict: Option<Predictor>,
    offload: Offload,
    costs: Costs,
    kernel: Option<Kernel>,
    // the bytes of an expert (the codes and scales of its matrices)
    expert_bytes: u64,
    // the memory the experts may take in the computer (None: every expert is in it), and what the disk tiers have been given of it
    host: Option<HostMemory>,
    pinned: u64,
    frozen: bool,
    // the next `run_parts`' wait: a mapped host word the front's last kernel moves to the value given (`arm_ready`), looked at instead of the stream's synchronize
    ready: Option<(usize, u32)>,
    // each layer's stats of the last step ended (`end_step`) and of every step since the store was made or made cold
    last: Vec<LayerStats>,
    totals: Vec<LayerStats>,
}

/// The computer's memory for the experts: `bytes` of the `total` bytes of all the experts of all the layers. Where it is less than the total, a group of a layer is given its share of it (`ExpertStore::host_slots`) and holds
/// that many of its experts in memory, the rest on the disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    pub bytes: u64,
    pub total: u64,
}

/// A group of Hummingbird experts that is not all in the computer's memory (`ExpertStore::add_layer_on_disk`): its place, the addresses of its places on the GPU (`HbStack::places`), the slots it has in memory, and the
/// reader of its experts from the packed directory.
pub struct DiskIn {
    pub place: HbPlace,
    pub dev: [u64; 2],
    pub ram: usize,
    pub read: Reader,
}

/// `DiskIn` for a group of Hummingbird experts of any mixture of experts (src/moe_hbg.rs, `HbgStack::places`): its place, the family's activation and how the host finishes a pair, the places on the GPU, the slots in memory, the reader.
pub struct DiskInG {
    pub place: HbgPlace,
    pub act: crate::moe::Act,
    pub finish: crate::cpu_hbg::Finish,
    pub dev: [u64; 2],
    pub ram: usize,
    pub read: Reader,
}

/// The parts of a group on the disk that `add_on_disk` takes whatever its format.
struct DiskSlots {
    dev: [u64; 2],
    ram: usize,
    read: Reader,
}

/// The expert cache of a model: a `LayerCache`, a host tier and a set of places for each mixture-of-experts layer (by its number among them), and one copy stream. The layers' blocks share it (`Rc`); it is not for threads.
pub struct ExpertStore {
    inner: RefCell<Inner>,
}

fn pinned_copy(bytes: &[u8]) -> Result<Pinned> {
    let p = Pinned::new(bytes.len()).map_err(|e| {
        Error(format!(
            "{} MiB of the computer's memory cannot be pinned for the experts: {}",
            bytes.len() >> 20,
            e.msg.replace("out of memory", "no memory left")
        ))
    })?;
    // SAFETY: the allocation holds bytes.len() bytes and does not overlap the slice.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.ptr as *mut u8, bytes.len()) };
    Ok(p)
}

impl Drop for Cuda {
    /// The copies still queued (a hint's, a predictor's) have run before the pinned buffers, the host tier and the stream go.
    fn drop(&mut self) {
        if let Some(w) = &self.worker {
            w.drain();
        }
        let _ = self.copy.sync();
    }
}

impl Cuda {
    fn new() -> Result<Cuda> {
        Ok(Cuda {
            copy: Side::new()?,
            ids: Pinned::new(8)?,
            way: Pinned::new(8)?,
            wts: Pinned::new(8)?,
            z: Pinned::new(8)?,
            part: Pinned::new(8)?,
            out: Vec::new(),
            timing: VecDeque::new(),
            spare: Vec::new(),
            layer_ms: Vec::new(),
            issued: 0,
            retired: 0,
            frozen: false,
            worker: None,
            use_worker: false,
            ordinal: None,
            hands: None,
            fixed: false,
        })
    }

    fn event(&mut self) -> Result<Ev> {
        match self.spare.pop() {
            Some(e) => Ok(e),
            None => Ev::new(),
        }
    }

    /// The batches of copies that have finished, added to the time and to what `costs` expects a copy to take (all of them, once the copy stream is waited for).
    fn count(&mut self, stats: &mut Stats, costs: &mut Costs, expert: u64, all: bool) -> Result<()> {
        if all {
            if let Some(w) = &self.worker {
                w.drain();
            }
            self.copy.sync()?;
        }
        while let Some((a, b, bytes, job, layer)) = self.timing.front() {
            // (a batch of the copy thread's has its events recorded once the thread has submitted it)
            if job.as_ref().is_some_and(|j| !j.submitted()) || !b.done()? {
                break;
            }
            let ms = b.since(a)? as f64;
            stats.copy_ms += ms;
            if self.layer_ms.len() <= *layer {
                self.layer_ms.resize(*layer + 1, 0.0);
            }
            self.layer_ms[*layer] += ms;
            if !self.frozen {
                costs.copied(*bytes, ms, expert);
            }
            let (a, b, _, _, _) = self.timing.pop_front().expect("a batch");
            self.spare.extend([a, b]);
            self.retired += 1;
        }
        Ok(())
    }

    /// The time the batches of copies not yet counted will take, as `costs` expects an expert of `expert` bytes to: the copy stream's backlog (none where the planner's times are held: they do not follow the run).
    fn backlog(&self, expert: u64, costs: &Costs) -> f64 {
        if self.frozen {
            return 0.0;
        }
        costs.queued(self.timing.iter().map(|t| t.2).sum(), expert)
    }

    /// Room for `pairs` choices in the host buffers the ids come to and the places go from, and for `z` bytes of the experts' input and `part` bytes of the host's sums: each buffer that is too small is made larger (the
    /// engine's stream is waited for before the old ones are freed: a copy queued from them may not have run); none is made smaller.
    fn reserve(&mut self, pairs: usize, z: usize, part: usize, at: At) -> Result<()> {
        self.ordinal = Some(at.ordinal);
        let short = |b: &Pinned, need: usize| b.bytes < need;
        if short(&self.ids, pairs * 8) || short(&self.way, pairs * 8 * WAYS) || short(&self.wts, pairs * 4) || short(&self.z, z) || short(&self.part, part) {
            if self.fixed {
                return err(format!("a call of {pairs} pairs, where the host's buffers that graphs name were made for fewer"));
            }
            at.sync()?;
            // (the choices, their weights and the input mapped for the device: a kernel of the step's front writes them there, with no copy: `parts_host`)
            for (b, need, mapped) in [
                (&mut self.ids, pairs * 8, true),
                (&mut self.way, pairs * 8 * WAYS, false),
                (&mut self.wts, pairs * 4, true),
                (&mut self.z, z, true),
                (&mut self.part, part, false),
            ] {
                if short(b, need) {
                    *b = if mapped { Pinned::mapped(need.max(8))? } else { Pinned::new(need.max(8))? };
                }
            }
        }
        Ok(())
    }
}

impl Resident {
    /// The cache emptied: after a copy that failed part way, the places hold what is not known, and an empty cache copies again whatever is asked for.
    fn forget(&mut self) {
        self.cache = LayerCache::new(self.cache.experts(), self.cache.slots());
        if let Some(j) = self.job.take() {
            let _ = j.wait();
        }
        self.pending = false;
    }

    /// Waits until the copy thread has queued the layer's last batch (its `ready` is then recorded).
    fn join_job(&mut self) -> Result<()> {
        match self.job.take() {
            Some(j) => j.wait(),
            None => Ok(()),
        }
    }

    /// The engine's stream `s` told to wait for the copies of the layer that are pending (the copy thread's batch is joined first, so that the event it records is the one waited for).
    fn wait_ready(&mut self, s: Stream) -> Result<()> {
        self.join_job()?;
        if self.pending {
            self.ready.wait(s)?;
            self.pending = false;
        }
        Ok(())
    }

    /// The layer's hidden size, as the host's half has it (0: none).
    fn hidden(&self) -> usize {
        match self.cpu.as_ref().map(|c| &c.half) {
            Some(HostHalf::Rows(r)) => r.hidden,
            Some(HostHalf::Hb(p)) => p.cols,
            Some(HostHalf::Cb2(_, hidden, _)) => *hidden,
            Some(HostHalf::Fp4 { latent, .. }) => *latent,
            Some(HostHalf::Hbg { place, .. }) => place.width,
            None => 0,
        }
    }

    /// The bytes of an expert in the host tier (and in a place).
    fn expert_bytes(&self) -> u64 {
        self.packs.iter().map(|p| p.stride[0] + p.stride[1]).sum()
    }

    /// Expert `e`'s own bytes in the host tier (a layer of mixed formats: its own lengths), where it is.
    fn bytes_of(&self, e: usize) -> u64 {
        let slot = self.host_slot(e).unwrap_or(e);
        self.packs.iter().map(|p| p.bytes_of(slot)).sum()
    }

    /// The own bytes of the experts a call missed (`serve`'s: copied, or kept for the host), each once.
    fn missed_bytes(&self, sv: &Served) -> u64 {
        sv.loads.iter().map(|l| l.0).chain(sv.kept.iter().copied()).map(|e| self.bytes_of(e as usize)).sum()
    }

    /// `host_reads` of the layer's host half: its FP4 kernel and the hummingbird places' of every mixture of experts take an expert's pairs together (src/cpu_fp4.rs `by_expert`, src/cpu_hbg.rs `store_pairs`), the
    /// others a pair at a time.
    fn host_reads(&self, ids: &[i64], mine: &[usize]) -> (usize, u64) {
        let grouped = matches!(self.cpu.as_ref().map(|c| &c.half), Some(HostHalf::Fp4 { .. } | HostHalf::Hbg { .. }));
        host_reads(ids, mine, self.cache.experts(), grouped, |e| self.bytes_of(e))
    }

    /// Where expert `e`'s bytes are in the host tier: its own place where every expert is there, else its slot (an expert that is in none was not read before it was used: a fault).
    fn host_slot(&self, e: usize) -> Result<usize> {
        match &self.tier {
            None => Ok(e),
            Some(t) => t
                .ram
                .slot_of(e)
                .ok_or_else(|| Error(format!("expert {e} is not in the computer's memory: it was not read from the packed directory before it was used"))),
        }
    }

    /// Expert `e` of a Hummingbird group as the host's kernels read it: its place in the host tier, where it is.
    fn hb_expert(&self, e: usize) -> Result<HbExpert<'_>> {
        let Some(CpuSide { half: HostHalf::Hb(place), .. }) = &self.cpu else { return err("a layer without a Hummingbird host half".to_string()) };
        let (data, meta) = (&self.packs[0].host[0], &self.packs[0].host[1]);
        // SAFETY: the host tier holds every expert of the layer end to end, or its slots (`add_layer` and `add_layer_on_disk` checked the sizes), and nothing writes the slot read: a slot is written by `stage` on the thread that
        // calls, before the reads of its call, and the reads of a call are done before the next.
        unsafe { hb_view(place, data.ptr as *const u8, meta.ptr as *const u8, self.host_slot(e)?) }
    }

    /// Expert `e` of a group of Hummingbird experts of any mixture of experts (`HostHalf::Hbg`) as the host reads it: its place in the host tier, where it is.
    fn hbg_expert(&self, e: usize) -> Result<HbgExpert<'_>> {
        let Some(CpuSide { half: HostHalf::Hbg { place, .. }, .. }) = &self.cpu else { return err("a layer without a Hummingbird host half".to_string()) };
        let (data, meta, at) = (&self.packs[0].host[0], &self.packs[0].host[1], self.host_slot(e)?);
        // SAFETY: as `hb_expert`'s: the host tier holds the layer's experts end to end, or its slots, and the slot read is not written during the call.
        unsafe {
            place.views(
                std::slice::from_raw_parts((data.ptr as *const u8).add(at * place.data), place.data),
                std::slice::from_raw_parts((meta.ptr as *const u8).add(at * place.meta), place.meta),
            )
        }
    }

    /// Expert `e` of a layer of this format as the host's kernels read it: its place in the host tier, where it is (the mapped file's bytes: the codes of its downT then its up, then the codebooks of both), the formats of
    /// its two matrices those of its entry in the layer's placement (two-bit both without one).
    fn glm_expert(&self, e: usize, limit: f32) -> Result<crate::cpu_cb::GlmExpert<'_>> {
        use crate::cpu_cb::{Fmt, GlmExpert, GlmMat};
        let p = &self.packs[0];
        let slot = self.host_slot(e)?;
        let (fu, fd) = p.place.as_ref().map_or((Fmt::Cb2, Fmt::Cb2), |pl| pl.fmts[slot]);
        let (cl, bl) = (p.len(0, slot), p.len(1, slot));
        // SAFETY: the host tier holds every expert of the layer (`add_layer_mapped` checked the offsets and the lengths) and nothing writes it.
        let (codes, books) = unsafe { (std::slice::from_raw_parts(p.addr(0, slot), cl), std::slice::from_raw_parts(p.addr(1, slot), bl)) };
        let Some(CpuSide { half: HostHalf::Cb2(_, hid, inter), .. }) = &self.cpu else { return err("a layer without a host half".to_string()) };
        let (hid, inter) = (*hid, *inter);
        let (dc, uc) = codes.split_at(inter * fd.row_bytes(hid));
        let (db, ub) = books.split_at(inter * fd.entries() * 2);
        GlmExpert::mixed(
            GlmMat { fmt: fu, rows: 2 * inter, cols: hid, codes: uc, book: ub },
            GlmMat { fmt: fd, rows: inter, cols: hid, codes: dc, book: db },
            limit,
        )
    }

    /// Expert `e` as the host's kernels read it: the first matrices of `cpu`, the second ones the host tier's.
    fn cpu_expert(&self, e: usize) -> ExpertView<'_> {
        let Some(CpuSide { half: HostHalf::Rows(c), .. }) = &self.cpu else { panic!("a layer with a host half of rows") };
        let (ud, us) = (c.inter * row_stride(c.bits, c.hidden), c.inter * (c.hidden / 32) * 2);
        let down = &self.packs[1];
        let (dd, ds) = (down.stride[0] as usize, down.stride[1] as usize);
        // SAFETY: the host tier holds every expert of the layer end to end (`add_layer` checked the sizes), and nothing writes it.
        let (data, scales) = unsafe {
            (
                std::slice::from_raw_parts((down.host[0].ptr as *const u8).add(e * dd), dd),
                std::slice::from_raw_parts((down.host[1].ptr as *const u8).add(e * ds), ds),
            )
        };
        ExpertView {
            up: RowsView { bits: c.bits, rows: c.inter, cols: c.hidden, data: &c.data[e * ud..(e + 1) * ud], scales: &c.scales[e * us..(e + 1) * us] },
            down: RowsView { bits: c.bits, rows: c.inter, cols: c.hidden, data, scales },
        }
    }
}

impl Resident {
    /// Expert `e` of a layer whose host half is `HostHalf::Fp4`, where the host tier holds it.
    fn fp4_expert(&self, e: usize) -> crate::cpu_fp4::Fp4Expert<'_> {
        let Some(CpuSide { half: HostHalf::Fp4 { inter, latent, packed, .. }, .. }) = &self.cpu else { panic!("a layer with a host half of FP4 experts") };
        let view = |p: &HostPack, rows: usize, cols: usize| {
            let (d, s) = (p.stride[0] as usize, p.stride[1] as usize);
            // SAFETY: the host tier holds every expert of the layer end to end (`add_layer` checked the sizes), and nothing writes it.
            let (data, blocks) = unsafe {
                (std::slice::from_raw_parts((p.host[0].ptr as *const u8).add(e * d), d), std::slice::from_raw_parts((p.host[1].ptr as *const u8).add(e * s), s))
            };
            crate::cpu_fp4::Fp4View { data, blocks, rows, cols, packed: *packed }
        };
        crate::cpu_fp4::Fp4Expert { up: view(&self.packs[0], 2 * inter, *latent), down: view(&self.packs[1], *latent, *inter) }
    }

    /// The layer's FP4 experts as the plan on the GPU's host thread reads them (src/devplan.rs), where its host half is FP4 experts, every one of them in the host tier, in two packs the kernels copy in
    /// 16-byte pieces; None for any other layer (the host plans its calls).
    fn host_fp4(&self) -> Option<crate::devplan::HostFp4> {
        let Some(CpuSide { half: HostHalf::Fp4 { inter, latent, packed, act }, .. }) = &self.cpu else { return None };
        let fits = self.tier.is_none()
            && self.packs.len() == 2
            && self
                .packs
                .iter()
                .all(|p| p.place.is_none() && (0..2).all(|k| p.stride[k] % 16 == 0 && p.dev[k] % 16 == 0 && (p.host[k].ptr as usize) % 16 == 0))
            && *latent <= crate::devplan::XP_WIDTH
            && self.cache.experts() <= crate::devplan::XP_MAX;
        let m = |p: &HostPack| [(p.host[0].ptr as usize, p.stride[0] as usize), (p.host[1].ptr as usize, p.stride[1] as usize)];
        fits.then(|| crate::devplan::HostFp4 {
            up: m(&self.packs[0]),
            down: m(&self.packs[1]),
            inter: *inter,
            latent: *latent,
            packed: *packed,
            act: *act,
            experts: self.cache.experts(),
        })
    }
}

/// Expert (or slot) number `at` of a Hummingbird group's host tier as the host's kernels read it: `place`'s view over its `data` and `meta` bytes where they are.
///
/// # Safety
/// `data` and `meta` hold at least `at + 1` experts' bytes (`place.data` and `place.meta` each) and are not written while the view lives.
unsafe fn hb_view<'a>(place: &HbPlace, data: *const u8, meta: *const u8, at: usize) -> Result<HbExpert<'a>> {
    place.expert(std::slice::from_raw_parts(data.add(at * place.data), place.data), std::slice::from_raw_parts(meta.add(at * place.meta), place.meta))
}

/// The link's time for one expert of layer `ly` (its bytes in the host tier copied into its first place, which is free: the plan on the GPU has placed nothing yet), the least of three, microseconds.
fn link_us(cuda: &Cuda, ly: &Resident) -> Result<f64> {
    let s = cuda.copy.stream;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        for p in &ly.packs {
            for k in 0..2 {
                // SAFETY: the host tier holds expert 0 (its first `stride` bytes) and place 0 has room for an expert; no call has run, so nothing reads the place.
                unsafe { copy_h2d_async(p.dev[k], p.host[k].ptr as *const _, p.stride[k] as usize, s)? };
            }
        }
        cuda.copy.sync()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e6);
    }
    Ok(best)
}

/// The copies of `loads` ((expert, place)) queued on the copy stream, behind the engine's stream's last use of the layer's places; the layer's `ready` is recorded after them. Returns the bytes queued.
fn fetch(cuda: &mut Cuda, ly: &mut Resident, loads: &[(u16, u32)], stats: &mut Stats) -> Result<u64> {
    if loads.is_empty() {
        return Ok(0);
    }
    // (where each expert's bytes are in the host tier: its slot, with a disk tier; all found before a copy is queued)
    let at = loads.iter().map(|&(e, _)| ly.host_slot(e as usize)).collect::<Result<Vec<_>>>()?;
    ly.join_job()?;
    for &slot in &at {
        for p in &ly.packs {
            for k in 0..2 {
                p.will_need(k, slot);
            }
        }
    }
    let s = cuda.copy.stream;
    if cuda.use_worker {
        if let Some(ord) = cuda.ordinal {
            if cuda.worker.is_none() {
                cuda.worker = Some(Worker::new(ord)?);
            }
            // (one batch of a layer after another: the earlier one's events are recorded before this one's)
            ly.join_job()?;
            let (a, b) = (cuda.event()?, cuda.event()?);
            let (mut copies, mut bytes) = (Vec::new(), 0u64);
            for (&(_, place), &slot) in loads.iter().zip(&at) {
                for p in &ly.packs {
                    for k in 0..2 {
                        // SAFETY: as below: the host tier holds the expert's bytes (or its slot's) and the places have room for as many experts as the cache has.
                        let src = p.addr(k, slot) as usize;
                        copies.push((p.dev[k] + place as u64 * p.stride[k], src, p.len(k, slot)));
                        bytes += p.len(k, slot) as u64;
                    }
                    if let Some(pl) = &p.place {
                        // (the formats' byte, to the last byte of the place's codebooks)
                        copies.push((p.dev[1] + place as u64 * p.stride[1] + p.stride[1] - 1, pl.marks.as_ptr() as usize + slot, 1));
                    }
                }
            }
            let state = JobState::new();
            // SAFETY: the layer's events (done, ready) live as long as the cache, and the timing events a and b stay in `cuda.timing` until the
            // job is seen done: each outlives the copy thread's use of its handle.
            let job = unsafe {
                CopyJob { copies, stream: s, wait: ly.done.as_ref(), start: a.as_ref(), stop: b.as_ref(), ready: ly.ready.as_ref(), state: state.clone() }
            };
            cuda.worker.as_ref().expect("the copy thread").submit(job)?;
            ly.pending = true;
            ly.job = Some(state.clone());
            stats.bytes += bytes;
            cuda.timing.push_back((a, b, bytes, Some(state), ly.index));
            cuda.issued += 1;
            if let Some(t) = ly.tier.as_mut() {
                at.iter().for_each(|&slot| t.busy[slot] = cuda.issued);
            }
            return Ok(bytes);
        }
    }
    ly.done.wait(s)?;
    let (a, b) = (cuda.event()?, cuda.event()?);
    a.record(s)?;
    let mut bytes = 0u64;
    for (&(_, place), &slot) in loads.iter().zip(&at) {
        for p in &ly.packs {
            for k in 0..2 {
                // SAFETY: the host tier holds every expert of the layer end to end (or its slots, and the expert is in one: `host_slot`) and the places have room for as many as the cache has; the copy stream runs the copies in order.
                unsafe {
                    let src = p.addr(k, slot);
                    copy_h2d_async(p.dev[k] + place as u64 * p.stride[k], src as *const _, p.len(k, slot), s)?;
                }
                bytes += p.len(k, slot) as u64;
            }
            if let Some(pl) = &p.place {
                // SAFETY: the byte is the layer's own table's, which outlives the store's layer.
                unsafe { copy_h2d_async(p.dev[1] + place as u64 * p.stride[1] + p.stride[1] - 1, pl.marks.as_ptr().add(slot) as *const _, 1, s)? };
            }
        }
    }
    b.record(s)?;
    ly.ready.record(s)?;
    ly.pending = true;
    stats.bytes += bytes;
    cuda.timing.push_back((a, b, bytes, None, ly.index));
    // (the slots these copies read are not to be written before the batch has run)
    cuda.issued += 1;
    if let Some(t) = ly.tier.as_mut() {
        at.iter().for_each(|&slot| t.busy[slot] = cuda.issued);
    }
    Ok(bytes)
}

/// Experts `wanted` (distinct, at most the slots of the layer's disk tier) in the host tier's slots, read from the packed directory where they are not (see `Tier::stage`); nothing for a layer with every expert in memory. The
/// time and the counts are the stats'.
fn stage(cuda: &mut Cuda, ly: &mut Resident, wanted: &[u16], stats: &mut Stats, costs: &mut Costs, expert: u64) -> Result<()> {
    let Resident { tier: Some(t), packs, .. } = ly else { return Ok(()) };
    if wanted.is_empty() {
        return Ok(());
    }
    let mem =
        Slots { base: [packs[0].host[0].ptr as *mut u8, packs[0].host[1].ptr as *mut u8], stride: [packs[0].stride[0] as usize, packs[0].stride[1] as usize] };
    // (the batches that have run by now are counted, so that a slot is waited for only if its copies are really still queued)
    cuda.count(stats, costs, expert, false)?;
    let retired = cuda.retired;
    let got = t.stage(wanted, mem, retired, &mut || {
        cuda.count(stats, costs, expert, true)?;
        Ok(cuda.retired)
    })?;
    stats.ram_hits += got.hits as u64;
    stats.disk_reads += got.reads as u64;
    stats.disk_bytes += got.reads as u64 * (mem.stride[0] + mem.stride[1]) as u64;
    stats.disk_ms += got.ms;
    Ok(())
}

/// `stage` for the two groups of a layer together (`wanted[g]` for the layer `ly[g]`, None where the group is not in the store): the reads of both are under way at once, and done when this returns. The queue of a drive wants
/// the reads of a call all at once.
fn stage_pair(cuda: &mut Cuda, ly: [Option<&mut Resident>; 2], wanted: [&[u16]; 2], stats: &mut Stats, costs: &mut Costs, expert: u64) -> Result<()> {
    cuda.count(stats, costs, expert, false)?;
    let retired = cuda.retired;
    let (mut work, mut failed): (Vec<(&mut Tier, usize)>, Option<Error>) = (Vec::new(), None);
    let mut staging: Vec<Staging> = Vec::new();
    for (ly, wanted) in ly.into_iter().zip(wanted) {
        let Some(Resident { tier: Some(t), packs, .. }) = ly else { continue };
        if wanted.is_empty() {
            continue;
        }
        let mem = Slots {
            base: [packs[0].host[0].ptr as *mut u8, packs[0].host[1].ptr as *mut u8],
            stride: [packs[0].stride[0] as usize, packs[0].stride[1] as usize],
        };
        match t.begin(wanted, mem, retired, &mut || {
            cuda.count(stats, costs, expert, true)?;
            Ok(cuda.retired)
        }) {
            Ok(s) => {
                staging.push(s);
                work.push((t, mem.stride[0] + mem.stride[1]));
            }
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    // (every read started is waited for, and a tier whose reads failed is emptied, whatever became of the other)
    for ((t, bytes), s) in work.into_iter().zip(staging) {
        match t.finish(s) {
            Ok(got) => {
                stats.ram_hits += got.hits as u64;
                stats.disk_reads += got.reads as u64;
                stats.disk_bytes += (got.reads * bytes) as u64;
                stats.disk_ms += got.ms;
            }
            Err(e) => failed = failed.or(Some(e)),
        }
    }
    failed.map_or(Ok(()), Err)
}

/// The pairs the host computes (`mine`: (group, pair), `local[g][pair]` the pair's expert's number in its group) cut into rounds of consecutive pairs whose experts the `slots[g]` slots of each group's host tier hold at once:
/// (the first pair, one past the last, the distinct experts of each group in the round in the order they first appear). One round where the tiers hold every expert (`usize::MAX` slots: no experts listed, nothing is read) or the call has no more experts.
fn host_rounds(mine: &[(usize, usize)], local: &[Vec<i64>; 2], slots: [usize; 2]) -> Vec<(usize, usize, [Vec<u16>; 2])> {
    // (no tier: the call's pairs are one round, and nothing is read)
    if slots == [usize::MAX; 2] {
        return vec![(0, mine.len(), [Vec::new(), Vec::new()])];
    }
    let (mut rounds, mut start) = (Vec::new(), 0usize);
    while start < mine.len() {
        let mut seen: [Vec<u16>; 2] = [Vec::new(), Vec::new()];
        let mut end = start;
        while let Some(&(g, j)) = mine.get(end) {
            let expert = local[g][j] as u16;
            if !seen[g].contains(&expert) {
                if seen[g].len() == slots[g] {
                    break;
                }
                seen[g].push(expert);
            }
            end += 1;
        }
        rounds.push((start, end, seen));
        start = end;
    }
    rounds
}

/// The experts of the host's pairs `mine` (numbers into a call's choices `ids`, of a layer of `experts`), each once, and the bytes the host's kernel reads of them (`bytes_of` an expert's): an expert's once for all
/// its pairs where the kernel takes them together (`grouped`), else once a pair.
pub fn host_reads(ids: &[i64], mine: &[usize], experts: usize, grouped: bool, bytes_of: impl Fn(usize) -> u64) -> (usize, u64) {
    let set = distinct(&mine.iter().map(|&j| ids[j]).collect::<Vec<_>>(), experts);
    let bytes = if grouped { set.iter().map(|&e| bytes_of(e as usize)).sum() } else { mine.iter().map(|&j| bytes_of(ids[j] as usize)).sum() };
    (set.len(), bytes)
}

/// The host's time `ms` for the pairs `mine` ((group, pair)) of a call of a layer in groups (`layers`: each group's layer of the store), shared between the groups' layers by their pairs.
fn host_time(all: &mut [Option<Resident>], layers: [Option<usize>; 2], mine: &[(usize, usize)], ms: f64) {
    for (g, l) in layers.iter().enumerate() {
        let n = mine.iter().filter(|&&(h, _)| h == g).count();
        if let Some(ly) = l.and_then(|l| all.get_mut(l)).and_then(Option::as_mut).filter(|_| n > 0) {
            ly.stats.host_ms += ms * n as f64 / mine.len() as f64;
        }
    }
}

/// Layers `a` and `b` of the store (either may be none), both at once.
fn two_layers(layers: &mut [Option<Resident>], a: Option<usize>, b: Option<usize>) -> [Option<&mut Resident>; 2] {
    match (a, b) {
        (Some(x), Some(y)) => {
            let [p, q] = layers.get_disjoint_mut([x, y]).expect("two layers of the store");
            [p.as_mut(), q.as_mut()]
        }
        (Some(x), None) => [layers[x].as_mut(), None],
        (None, Some(y)) => [None, layers[y].as_mut()],
        (None, None) => [None, None],
    }
}

/// The copies of `loads` as `fetch` queues them, after their experts are in the host tier's slots: with a disk tier a slot's worth at a time (`stage`, then the copies of those; a later slot's worth may take the slots of an
/// earlier one once the copies have read them), else all at once. Returns the bytes queued.
fn place(cuda: &mut Cuda, ly: &mut Resident, loads: &[(u16, u32)], stats: &mut Stats, costs: &mut Costs, expert: u64) -> Result<u64> {
    let Some(size) = ly.tier.as_ref().map(|t| t.ram.slots()) else { return fetch(cuda, ly, loads, stats) };
    let mut bytes = 0;
    for chunk in loads.chunks(size) {
        let wanted: Vec<u16> = chunk.iter().map(|l| l.0).collect();
        stage(cuda, ly, &wanted, stats, costs, expert)?;
        bytes += fetch(cuda, ly, chunk, stats)?;
    }
    Ok(bytes)
}

/// What a layer's call hands the store: the device buffers of the router's choices (int64 `pairs`), their weights (fp32), the experts' input z (bf16, `tokens` rows), the buffer the pairs' places go to (int64) and the one
/// the host's sums come to (fp32, `tokens` rows of the layer's hidden size, `CPU_TOKENS` rows at most).
pub struct Call {
    pub ids: u64,
    pub wts: u64,
    pub z: u64,
    pub way: u64,
    pub part: u64,
    pub tokens: usize,
    pub pairs: usize,
}

/// The two stages of a call the engine's stream is given in turn: the experts' products, and the sum of their results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Products,
    Sum,
}

/// What a stage is told: `gpu`, whether any pair has a place (else the products are not run), and for the sum the device address of the ids it is of (the pairs' places: -1 for a pair the host computed; 0: the router's
/// choices) and of the host's sums (0: none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hybrid {
    pub gpu: bool,
    pub ids: u64,
    pub extra: u64,
}

/// A model layer whose experts are in groups, each a layer of the store (the Hummingbird level's two bit widths): the store's layer of each group (None: the group is all on the GPU, and not the store's), and each
/// expert's group and its number there.
pub struct Groups<'a> {
    pub layers: [Option<usize>; 2],
    pub group_of: &'a [u8],
    pub local_of: &'a [u16],
}

/// What a call of a layer in groups hands the store (`run_groups`): the device buffers of the router's choices (int64 `pairs`, the model's expert numbers), their weights (fp32), the experts' input as the host reads it
/// (fp32, `tokens` rows: the Hummingbird layer's rotated z'), the buffers the pairs' places go to for each group (int64) and the ids of the sum (int64), and the one the host's sums come to (fp32, `tokens` rows of the
/// layer's hidden size, `CPU_TOKENS` at most).
pub struct GroupsCall {
    pub ids: u64,
    pub wts: u64,
    pub z: u64,
    pub ways: [u64; 2],
    pub sum_ids: u64,
    pub part: u64,
    pub tokens: usize,
    pub pairs: usize,
}

/// The host's rows (`cuda.out`: `hidden` fp32 for each of `tokens_of`, a pair's token) added up by token into the pinned sums of `tokens` rows and sent to the device at `part` on `s`.
///
/// # Safety
/// `part` is a live device buffer of `tokens * hidden` floats, and the engine's stream does not read the pinned sums now.
unsafe fn send_sums(cuda: &mut Cuda, tokens_of: impl Iterator<Item = usize>, hidden: usize, tokens: usize, part: u64, s: Stream) -> Result<()> {
    let sums = std::slice::from_raw_parts_mut(cuda.part.ptr as *mut f32, tokens * hidden);
    sums.fill(0.0);
    for (row, t) in cuda.out.chunks_exact(hidden).zip(tokens_of) {
        sums[t * hidden..][..hidden].iter_mut().zip(row).for_each(|(a, b)| *a += *b);
    }
    copy_h2d_async(part, cuda.part.ptr, tokens * hidden * 4, s)
}

impl ExpertStore {
    /// An empty store. Its GPU side (the copy stream) is made with the first layer, in the context the engine made current.
    pub fn new() -> Rc<ExpertStore> {
        Rc::new(ExpertStore {
            inner: RefCell::new(Inner {
                mail: None,
                cuda: None,
                layers: Vec::new(),
                stats: Stats::default(),
                predict: None,
                offload: Offload::default(),
                costs: Costs::default(),
                kernel: None,
                expert_bytes: 0,
                host: None,
                pinned: 0,
                frozen: false,
                ready: None,
                last: Vec::new(),
                totals: Vec::new(),
            }),
        })
    }

    /// How a miss is served from now on (the layers loaded with a host half serve them by the mode; the others copy).
    pub fn set_offload(&self, o: Offload) {
        self.inner.borrow_mut().offload = o;
    }

    pub fn offload(&self) -> Offload {
        self.inner.borrow().offload
    }

    /// The host's pairs of this store's calls computed on `other`'s thread for them (a model split over GPUs has a store a GPU, and their calls never overlap: one thread looks for them, where a thread of each
    /// would look for its store's next call while the other store's calls run). Both have their first layer.
    pub fn share_host_thread(&self, other: &ExpertStore) -> Result<()> {
        let none = || Error("the expert cache has no layers".into());
        let tx = other.inner.borrow_mut().cuda.as_mut().ok_or_else(none)?.hands()?.clone();
        self.inner.borrow_mut().cuda.as_mut().ok_or_else(none)?.hands = Some(tx);
        Ok(())
    }

    /// Whether the layers to come are given a host half (`add_layer`'s `cpu`): any mode but `Copy`.
    pub fn wants_cpu(&self) -> bool {
        self.inner.borrow().offload.mode != Mode::Copy
    }

    /// The computer's memory for the experts of the layers to come (None: every expert in it, as the store has always held them). With less than all of them, the loader gives each group of a layer its share (`host_slots`)
    /// and `add_layer_on_disk`.
    pub fn set_host_memory(&self, m: Option<HostMemory>) {
        let mut inner = self.inner.borrow_mut();
        inner.host = m.filter(|m| m.bytes < m.total);
        inner.pinned = 0;
    }

    /// The memory for the experts when it is less than all of them: the layers to come are loaded with a disk tier.
    pub fn host_memory(&self) -> Option<HostMemory> {
        self.inner.borrow().host
    }

    /// The slots a group of `experts` experts of `bytes` each is given of the computer's memory: its share by the bytes of the budget over the bytes of all the experts (at least one slot, at most the group), and no more than
    /// the budget has left (an error where not even one slot is left). Every expert where the memory holds them all.
    pub fn host_slots(&self, experts: usize, bytes: u64) -> Result<usize> {
        let mut inner = self.inner.borrow_mut();
        let Some(m) = inner.host else { return Ok(experts) };
        let share = (experts as u128 * m.bytes as u128 / m.total.max(1) as u128) as usize;
        let slots = share.clamp(1, experts).min((m.bytes.saturating_sub(inner.pinned) / bytes.max(1)) as usize);
        if slots == 0 {
            return err(format!(
                "{} MiB of the computer's memory for the experts are spent: a group of {experts} experts of {bytes} bytes has none left",
                m.bytes >> 20
            ));
        }
        inner.pinned += slots as u64 * bytes;
        Ok(slots)
    }

    /// The planner's times held at the values they start with (a run neither measures the copy stream nor the host's pairs, and the copy stream's backlog is none): two runs of one mode, from one state, plan alike
    /// whatever their timings, which is how a test holds a run to another to the bit.
    pub fn hold_costs(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.frozen = true;
        if let Some(c) = inner.cuda.as_mut() {
            c.frozen = true;
        }
    }

    /// Layer `layer` (its number among the mixture-of-experts layers), of `experts` experts, with `slots` places: its experts' bytes copied into pinned host memory, the places' addresses noted. The cache starts empty.
    /// `cpu`: what the host computes the layer's experts from (`HostHalf`): its first matrices in the row layout (the second pack must be in the row layout then), or, for a Hummingbird group, nothing more than its place
    /// (the one pack must be the place's: the host reads the experts in the host tier).
    pub fn add_layer(&self, layer: usize, experts: usize, slots: usize, packs: &[PackIn], cpu: Option<HostHalf>) -> Result<()> {
        self.check_layer(experts, slots, packs, &cpu)?;
        let host = packs
            .iter()
            .map(|p| Ok(HostPack { stride: p.stride, dev: p.dev, host: [pinned_copy(p.bytes[0])?.into(), pinned_copy(p.bytes[1])?.into()], place: None }))
            .collect::<Result<Vec<_>>>()?;
        self.install(layer, experts, slots, host, cpu.map(|half| CpuSide { half, last: vec![0; experts] }), None)
    }

    /// `add_layer` of packs the caller made in pinned memory itself (`PinnedPack`: every expert's codes and scales end to end at its strides, packed there): the
    /// host tier is those buffers, no copy is made (a load of Kimi-K3's 1.35 TB of experts had copied each layer's 15 GB once more, and freed it).
    pub fn add_layer_pinned(&self, layer: usize, experts: usize, slots: usize, packs: Vec<PinnedPack>, cpu: Option<HostHalf>) -> Result<()> {
        {
            // SAFETY: each buffer holds `bytes` bytes, written by the caller, and lives while the views do.
            let views: Vec<PackIn> = packs
                .iter()
                .map(|p| unsafe {
                    PackIn {
                        bytes: [
                            std::slice::from_raw_parts(p.data.ptr as *const u8, p.data.bytes),
                            std::slice::from_raw_parts(p.scales.ptr as *const u8, p.scales.bytes),
                        ],
                        stride: p.stride,
                        dev: p.dev,
                    }
                })
                .collect();
            self.check_layer(experts, slots, &views, &cpu)?;
        }
        let host = packs.into_iter().map(|p| HostPack { stride: p.stride, dev: p.dev, host: [p.data.into(), p.scales.into()], place: None }).collect();
        self.install(layer, experts, slots, host, cpu.map(|half| CpuSide { half, last: vec![0; experts] }), None)
    }

    /// `add_layer`'s checks: the places, each pack's bytes for the experts, the host half's sizes.
    fn check_layer(&self, experts: usize, slots: usize, packs: &[PackIn], cpu: &Option<HostHalf>) -> Result<()> {
        if slots == 0 || slots > experts {
            return err(format!("a cache of {slots} places for {experts} experts"));
        }
        for p in packs {
            if (0..2).any(|k| p.bytes[k].len() as u64 != p.stride[k] * experts as u64) {
                return err(format!(
                    "a matrix of {} + {} bytes for {experts} experts of {} + {}",
                    p.bytes[0].len(),
                    p.bytes[1].len(),
                    p.stride[0],
                    p.stride[1]
                ));
            }
        }
        // (the host reads the second matrices where the host tier holds them: they must be the rows' size)
        match &cpu {
            Some(HostHalf::Rows(rows)) => {
                let (ud, us) = (rows.inter * row_stride(rows.bits, rows.hidden), rows.inter * (rows.hidden / 32) * 2);
                let down = packs.get(1).map(|p| p.stride);
                if !(4..=6).contains(&rows.bits)
                    || rows.hidden == 0
                    || rows.hidden % 32 != 0
                    || rows.data.len() != experts * ud
                    || rows.scales.len() != experts * us
                    || down != Some([ud as u64, us as u64])
                {
                    return err(format!(
                        "a host half of {} + {} bytes for {experts} experts of {} neurons of {} outputs at {} bits, with second matrices of {:?} bytes",
                        rows.data.len(),
                        rows.scales.len(),
                        rows.inter,
                        rows.hidden,
                        rows.bits,
                        down
                    ));
                }
            }
            Some(HostHalf::Hb(place)) if packs.len() != 1 || packs[0].stride != [place.data as u64, place.meta as u64] => {
                return err(format!(
                    "a host half of {} x {} rows at format {} ({} + {} bytes an expert), with packs of {:?} bytes",
                    place.rows,
                    place.cols,
                    place.bits,
                    place.data,
                    place.meta,
                    packs.iter().map(|p| p.stride).collect::<Vec<_>>()
                ));
            }
            Some(HostHalf::Fp4 { inter, latent, packed, .. }) => {
                let ok = packs.len() == 2
                    && [(2 * inter, *latent), (*latent, *inter)].iter().zip(packs).all(|(&(r, c), p)| {
                        crate::cpu_fp4::Fp4View {
                            data: p.bytes[0].get(..p.stride[0] as usize).unwrap_or(&[]),
                            blocks: p.bytes[1].get(..p.stride[1] as usize).unwrap_or(&[]),
                            rows: r,
                            cols: c,
                            packed: *packed,
                        }
                        .is_valid()
                    });
                if !ok {
                    return err(format!(
                        "a host half of FP4 experts of {inter} neurons and {latent} outputs, with packs of {:?} bytes",
                        packs.iter().map(|p| p.stride).collect::<Vec<_>>()
                    ));
                }
            }
            Some(HostHalf::Hbg { place, .. }) if packs.len() != 1 || packs[0].stride != [place.data as u64, place.meta as u64] => {
                return err(format!(
                    "a host half of experts of {} and {} rows of {} at {} bits ({} + {} bytes an expert), with packs of {:?} bytes",
                    place.up_rows,
                    place.down_rows,
                    place.width,
                    place.bits,
                    place.data,
                    place.meta,
                    packs.iter().map(|p| p.stride).collect::<Vec<_>>()
                ));
            }
            Some(HostHalf::Hb(_)) | Some(HostHalf::Hbg { .. }) | Some(HostHalf::Cb2(..)) | None => {}
        }
        Ok(())
    }

    /// Layer `layer`, a Hummingbird group of `experts` experts with `slots` places on the GPU, whose experts are not all in the computer's memory: `disk.ram` slots of the host tier (pinned, the group's first experts in them,
    /// read from the packed directory by `disk.read`), the others read into a slot as the router asks for them (see `Tier`). The host's half is the places of the host tier as they are (`HostHalf::Hb`).
    pub fn add_layer_on_disk(&self, layer: usize, experts: usize, slots: usize, disk: DiskIn) -> Result<()> {
        let place = disk.place;
        let stride = [place.data, place.meta];
        self.add_on_disk(layer, experts, slots, stride, disk.dev, disk.ram, disk.read, HostHalf::Hb(place))
    }

    /// `add_layer_on_disk` for a group of Hummingbird experts of any mixture of experts (src/moe_hbg.rs): its place, the family's activation and how the host finishes a pair (`HostHalf::Hbg`).
    pub fn add_layer_on_disk_g(&self, layer: usize, experts: usize, slots: usize, disk: DiskInG) -> Result<()> {
        let place = disk.place;
        let stride = [place.data, place.meta];
        self.add_on_disk(layer, experts, slots, stride, disk.dev, disk.ram, disk.read, HostHalf::Hbg { place, act: disk.act, finish: disk.finish })
    }

    /// A group whose experts are not all in the computer's memory: `ram` slots of the host tier of `stride` bytes an expert (its codes; its scales and codebooks), the places at `dev`, the reader, and the host's half.
    #[allow(clippy::too_many_arguments)]
    fn add_on_disk(
        &self,
        layer: usize,
        experts: usize,
        slots: usize,
        stride: [usize; 2],
        dev: [u64; 2],
        ram: usize,
        read: Reader,
        half: HostHalf,
    ) -> Result<()> {
        let disk = DiskSlots { dev, ram, read };
        if slots == 0 || slots > experts || disk.ram == 0 || disk.ram > experts {
            return err(format!("a cache of {slots} places and room in the computer's memory for {} experts, for {experts} experts", disk.ram));
        }
        let place = (stride[0], stride[1]);
        let stride = [stride[0] as u64, stride[1] as u64];
        let alloc = |k: usize| {
            Pinned::new(disk.ram * stride[k] as usize).map_err(|e| {
                Error(format!(
                    "{} MiB of the computer's memory cannot be pinned for the experts: {}",
                    (disk.ram as u64 * stride[k]) >> 20,
                    e.msg.replace("out of memory", "no memory left")
                ))
            })
        };
        let pack = HostPack { stride, dev: disk.dev, host: [alloc(0)?.into(), alloc(1)?.into()], place: None };
        let mem = Slots { base: [pack.host[0].ptr as *mut u8, pack.host[1].ptr as *mut u8], stride: [place.0, place.1] };
        let mut tier = Tier::new(experts, disk.ram, disk.read);
        tier.seed(mem)?;
        self.install(layer, experts, slots, vec![pack], Some(CpuSide { half, last: vec![0; experts] }), Some(tier))
    }

    /// Layer `layer` of `experts` experts with `slots` places on the GPU, whose experts are read from host memory the caller gives (`HostMem`: the codes and the codebooks of every expert end to end, `stride` bytes an
    /// expert in each, copied to the places at `dev + place * stride`): no pinned copy is made, so a mapped file serves as the host tier. Copies only (no host half).
    pub fn add_layer_mapped(
        &self,
        layer: usize,
        experts: usize,
        slots: usize,
        host: [HostMem; 2],
        place: Option<Placement>,
        stride: [u64; 2],
        dev: [u64; 2],
        cb2: Option<(f32, usize, usize)>,
    ) -> Result<()> {
        if slots == 0 || slots > experts {
            return err(format!("a cache of {slots} places for {experts} experts"));
        }
        match &place {
            None if cb2.is_some_and(|(_, h, i)| stride != [(3 * i * h / 4) as u64, (3 * i * 8) as u64]) => {
                return err(format!("a host half of the smallest format for experts of {stride:?} bytes"))
            }
            Some(pl) => {
                let ok = (0..2).all(|k| pl.at[k].len() == experts && pl.len[k].len() == experts && pl.len[k].iter().all(|&l| l as u64 <= stride[k]))
                    && pl.fmts.len() == experts
                    && pl.marks.len() == experts;
                if !ok || stride[1] < 16 + pl.len[1].iter().copied().max().unwrap_or(0) as u64 {
                    return err(format!("the placement of {experts} experts in places of {stride:?} bytes does not hold (every expert within a place)"));
                }
            }
            None => {}
        }
        let cpu = cb2.map(|(l, hidden, inter)| CpuSide { half: HostHalf::Cb2(l, hidden, inter), last: vec![0; experts] });
        self.install(layer, experts, slots, vec![HostPack { stride, dev, host, place }], cpu, None)
    }

    /// The layer's state made and put among the store's.
    fn install(&self, layer: usize, experts: usize, slots: usize, host: Vec<HostPack>, cpu: Option<CpuSide>, tier: Option<Tier>) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if inner.cuda.is_none() {
            let mut c = Cuda::new()?;
            c.frozen = inner.frozen;
            inner.cuda = Some(c);
        }
        if inner.layers.len() <= layer {
            inner.layers.resize_with(layer + 1, || None);
        }
        inner.expert_bytes = host.iter().map(|p| p.stride[0] + p.stride[1]).sum();
        inner.layers[layer] = Some(Resident {
            cache: LayerCache::new(experts, slots),
            packs: host,
            ready: Ev::new()?,
            done: Ev::new()?,
            pending: false,
            job: None,
            place: vec![-1; experts],
            cpu,
            calls: 0,
            tier,
            index: layer,
            stats: LayerStats::default(),
            dplan: None,
        });
        let inner = &mut *inner;
        if inner.offload.device {
            if let Some(h) = inner.layers[layer].as_ref().and_then(Resident::host_fp4) {
                let ly = inner.layers[layer].as_mut().expect("installed above");
                let regions: Vec<[u64; 4]> =
                    ly.packs.iter().flat_map(|p| (0..2).map(move |k| [p.host[k].ptr as u64, p.stride[k], p.dev[k], p.stride[k]])).collect();
                ly.dplan = Some(crate::devplan::DevPlan::new(experts, slots, &regions)?);
                if inner.mail.is_none() {
                    // (the plan on the GPU weighs a copy at the link's time for an expert, measured here once: its copies are the GPU's own, which the copy stream does not time)
                    let cuda = inner.cuda.as_mut().expect("made above");
                    let us = link_us(cuda, ly)?;
                    if !inner.frozen && us > 0.0 {
                        inner.costs.copy = us;
                    }
                    let params = crate::devplan::Params::of(&inner.offload, &inner.costs);
                    inner.mail = Some(crate::devplan::DevMail::new(inner.offload.threads, inner.offload.pin, params)?);
                }
                inner.mail.as_ref().expect("made above").add_layer(layer, h);
            }
        }
        Ok(())
    }

    /// Whether layer `layer`'s decode calls are planned on the GPU (`Offload::device`, a host half of FP4 experts with every expert in the computer's memory): a step of them needs no host and is one graph.
    pub fn device_on(&self, layer: usize) -> bool {
        self.inner.borrow().layers.get(layer).and_then(Option::as_ref).is_some_and(|l| l.dplan.is_some())
    }

    /// The layers whose decode calls are planned on the GPU.
    pub fn device_layers(&self) -> usize {
        self.inner.borrow().layers.iter().flatten().filter(|l| l.dplan.is_some()).count()
    }

    /// What a call of `tokens` tokens and `pairs` pairs of layer `layer` hands kernels/moe.cu's plan on the GPU (src/devplan.rs), where the layer is planned there and the call is one of a decode's
    /// (`devplan::DEV_ROWS` rows at most); None: the store plans it on the host (`run`).
    pub fn device_args(&self, layer: usize, tokens: usize, pairs: usize) -> Option<crate::devplan::DevArgs> {
        let inner = self.inner.borrow();
        let (ly, mail) = (inner.layers.get(layer)?.as_ref()?, inner.mail.as_ref()?);
        let dp = ly.dplan.as_ref()?;
        ((1..=crate::devplan::DEV_ROWS).contains(&tokens) && pairs <= crate::devplan::XP_PAIRS && pairs.is_multiple_of(tokens)).then(|| dp.args(mail, layer))
    }

    /// Layer `layer`'s places emptied and their copies done (the copy thread's batch joined and the copy stream past it), so that their memory can be freed (`QxMoeCb::release_places`); the places at `dev` from now on
    /// (the same strides and number), where a layer's places are made again (`restore_places`): its cache starts empty.
    /// Every layer's copies still queued done (the host waits): `hint`s behind a prompt finished before the decode's first step, not beside it.
    pub fn drain(&self) -> Result<()> {
        let mut guard = self.inner.borrow_mut();
        for ly in guard.layers.iter_mut().flatten() {
            ly.join_job()?;
            if ly.pending {
                ly.ready.wait_host()?;
            }
        }
        Ok(())
    }

    pub fn reset_places(&self, layer: usize, dev: [u64; 2]) -> Result<()> {
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let ly = inner.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
        if ly.dplan.is_some() {
            return err(format!("layer {layer}'s places are planned on the GPU, whose copies name them: they are not moved"));
        }
        ly.join_job()?;
        if ly.pending {
            ly.ready.wait_host()?;
        }
        ly.forget();
        ly.place.fill(-1);
        for p in &mut ly.packs {
            p.dev = dev;
        }
        Ok(())
    }

    /// (development) Layer `layer`'s places emptied where they are (the copies queued waited for): its next call finds no expert in one (`moe::PAIRS`: every pair the host's in
    /// cpu mode). Not for a layer the GPU plans.
    #[cfg(feature = "dev")]
    pub fn empty_places(&self, layer: usize) -> Result<()> {
        let mut guard = self.inner.borrow_mut();
        let ly = guard.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
        if ly.dplan.is_some() {
            return err(format!("layer {layer}'s places are planned on the GPU"));
        }
        ly.join_job()?;
        if ly.pending {
            ly.ready.wait_host()?;
        }
        ly.forget();
        ly.place.fill(-1);
        Ok(())
    }

    /// Layer `layer`'s places from now on `slots` at `dev` (made again in another number, the first ones' bytes moved there by the caller after `drain`): the experts in places past them leave the cache
    /// (`LayerCache::resize`), the others stay.
    pub fn resize_places(&self, layer: usize, slots: usize, dev: [u64; 2]) -> Result<()> {
        let mut guard = self.inner.borrow_mut();
        let ly = guard.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
        if ly.dplan.is_some() {
            return err(format!("layer {layer}'s places are planned on the GPU, whose copies name them: they are not moved"));
        }
        ly.join_job()?;
        if ly.pending {
            ly.ready.wait_host()?;
            ly.pending = false;
        }
        ly.cache.resize(slots);
        ly.place.fill(-1);
        for p in &mut ly.packs {
            p.dev = dev;
        }
        Ok(())
    }

    /// The place each expert of `layer` is in on the GPU (-1: none), after the engine's stream `s` has been told to wait for the copies of the layer that are pending.
    pub fn resident_places(&self, layer: usize, s: Stream) -> Result<Vec<i64>> {
        let mut guard = self.inner.borrow_mut();
        let ly = guard.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
        ly.wait_ready(s)?;
        Ok((0..ly.cache.experts()).map(|e| ly.cache.slot_of(e).map_or(-1, |p| p as i64)).collect())
    }

    /// Every layer's places empty, the counts and the planner's times as they were at the start (a measurement's own beginning): the work queued is waited for first.
    pub fn cold(&self, dev: &Dev) -> Result<()> {
        dev.sync()?;
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        if let Some(c) = inner.cuda.as_mut() {
            c.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, true)?;
        }
        for ly in inner.layers.iter_mut().flatten() {
            ly.forget();
            ly.calls = 0;
            if let Some(dp) = ly.dplan.as_mut() {
                dp.upload(&ly.cache.expert_in())?;
                dp.counters()?;
            }
            if let Some(c) = ly.cpu.as_mut() {
                c.last.fill(0);
            }
            // (a disk tier starts as it was loaded: the copy stream has run every copy by now, so no slot is read by one)
            let Resident { tier: Some(t), packs, .. } = ly else { continue };
            t.reseed(Slots {
                base: [packs[0].host[0].ptr as *mut u8, packs[0].host[1].ptr as *mut u8],
                stride: [packs[0].stride[0] as usize, packs[0].stride[1] as usize],
            })?;
        }
        inner.stats = Stats::default();
        inner.costs = Costs::default();
        for ly in inner.layers.iter_mut().flatten() {
            ly.stats = LayerStats::default();
        }
        if let Some(c) = inner.cuda.as_mut() {
            c.layer_ms.clear();
        }
        (inner.last, inner.totals) = (Vec::new(), Vec::new());
        Ok(())
    }

    /// Experts of layer `layer` a predictor expects it to be asked for, least expected first (`LayerCache::hint`): they are placed and copied beside the engine's work, and the layer's next call waits for the copies.
    /// For a layer other than the one being run, or between calls: the copies are ordered after the layer's last call, not after one being queued.
    pub fn hint(&self, layer: usize, experts: &[u16]) -> Result<()> {
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
        let ly = inner.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
        // (a layer the GPU plans keeps its places in the GPU's table: a hint between its calls is not taken)
        if ly.dplan.is_some() {
            return Ok(());
        }
        let got = ly.cache.hint(experts)?;
        inner.stats.hinted += got.loads.len() as u64;
        let bytes = place(cuda, ly, &got.loads, &mut inner.stats, &mut inner.costs, inner.expert_bytes).inspect_err(|_| ly.forget())?;
        ly.stats.ahead(got.loads.len(), bytes);
        cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)
    }

    /// A guess of the next call of layer `layer` (`experts`, the most expected first): the ones in places are touched and at most `copies` of the others (the most
    /// expected) placed and copied, as `hint` does, beside the engine's work until the call.
    pub fn guess(&self, layer: usize, experts: &[u16], copies: usize) -> Result<()> {
        let want: Vec<u16> = {
            let inner = self.inner.borrow();
            let ly = inner.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
            let mut misses = 0;
            experts
                .iter()
                .copied()
                .filter(|&x| {
                    ly.cache.slot_of(x as usize).is_some()
                        || (misses < copies && {
                            misses += 1;
                            true
                        })
                })
                .rev()
                .collect()
        };
        self.hint(layer, &want)
    }

    /// A predictor `run` asks, with each call's layer and experts, for the experts the layers ahead will be asked for; they are placed and copied (`hint`) at once, beside the engine's work on this layer and the ones
    /// between. The seam for a prediction from the router's choices, from the tokens before, or from a table; none is set by default.
    pub fn set_predictor(&self, p: impl FnMut(usize, &[u16]) -> Vec<(usize, Vec<u16>)> + 'static) {
        self.inner.borrow_mut().predict = Some(Box::new(p));
    }

    /// The predictor's hints for a call of `layer` that asked for `needed`: those for layers the cache holds other than the call's (`call`: `layer` and, for a layer in groups, the other group's, whose places the
    /// call has already sent and waited for: a hint there would copy over an expert the call is about to read) are made (the predictor is not held while it runs, so it may call `hint` itself).
    fn predicted(&self, layer: usize, needed: &[u16], call: [Option<usize>; 2]) -> Result<()> {
        let Some(mut p) = self.inner.borrow_mut().predict.take() else { return Ok(()) };
        let hints = p(layer, needed);
        self.inner.borrow_mut().predict = Some(p);
        for (l, experts) in hints {
            if !call.contains(&Some(l)) && self.inner.borrow().layers.get(l).is_some_and(Option::is_some) {
                self.hint(l, &experts)?;
            }
        }
        Ok(())
    }

    /// One call of layer `layer`: the router's `pairs` choices (int64 on the device at `call.ids`) are read to the host; the experts they name are made resident, in waves where there are more than the layer has places
    /// for; for each wave the pairs' places go to the device at `call.way` (int64: the pair's place, or -1) and `body` is given the stages of the call to queue on the engine's stream, after the copies: `Stage::Products`
    /// (the experts' products) and `Stage::Sum`.
    ///
    /// Where the layer has a host half, the mode is not `Copy` and the call has at most `CPU_TOKENS` tokens, the weights and z are read back with the choices and the misses are served as `plan` says: the copies are
    /// queued, `Stage::Products` is run for the pairs that have a place (one wave; a miss computed by the host is a pair with no place), the host computes the pairs of the experts it keeps and their sums go up to
    /// `call.part`, and `Stage::Sum` is run with them (`Hybrid::extra`) and the ids that leave the host's pairs out (`Hybrid::ids`).
    ///
    /// # Safety
    /// The addresses of `call` are live device buffers of the sizes it names; the engine's stream is `dev`'s, and `body` queues work on it alone.
    pub unsafe fn run(&self, layer: usize, dev: &Dev, call: &Call, body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>) -> Result<()> {
        self.run_early_at(layer, At::of(dev), call, &mut || Ok(()), body)
    }

    /// `run` on the GPU `at` (a layer that holds no `Dev`: `moe::MoeLayer`).
    ///
    /// # Safety
    /// As `run`.
    pub unsafe fn run_at(&self, layer: usize, at: At, call: &Call, body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>) -> Result<()> {
        self.run_early_at(layer, at, call, &mut || Ok(()), body)
    }

    /// `run_early_at` on a `Dev`.
    ///
    /// # Safety
    /// As `run`.
    pub unsafe fn run_early(
        &self,
        layer: usize,
        dev: &Dev,
        call: &Call,
        early: &mut dyn FnMut() -> Result<()>,
        body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>,
    ) -> Result<()> {
        self.run_early_at(layer, At::of(dev), call, early, body)
    }

    /// `run`, with `early` given the engine's stream right after the router's choices have been asked for and before the host waits for them: the GPU work of the layer that does not need the choices (the shared expert)
    /// is queued there and runs while the host reads the choices and plans, instead of before the host's wait.
    ///
    /// # Safety
    /// As `run`; `early` queues work on the engine's stream alone and does not use the store.
    pub unsafe fn run_early_at(
        &self,
        layer: usize,
        at: At,
        call: &Call,
        early: &mut dyn FnMut() -> Result<()>,
        body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>,
    ) -> Result<()> {
        // (the call's own time on the host: what it took less its waits for the GPU and the host's pairs: `LayerStats::plan_ms`)
        let spent = |st: &Self| st.inner.borrow().layers.get(layer).and_then(Option::as_ref).map_or(0.0, |l| l.stats.wait_ms + l.stats.host_ms);
        let (t0, before) = (Instant::now(), spent(self));
        // (a layer the GPU plans decode calls of: its places as the GPU has them before the host plans this call, and the host's after it, once the copies it queued have run for the engine's stream)
        let dev = self.device_on(layer);
        if dev {
            at.sync()?;
            let mut inner = self.inner.borrow_mut();
            let ly = inner.layers[layer].as_mut().expect("a layer");
            let places = ly.dplan.as_ref().expect("planned on the GPU").download()?;
            ly.cache.set_places(&places);
        }
        let mut r = self.run_call(layer, at, call, early, body);
        if dev && r.is_ok() {
            let mut inner = self.inner.borrow_mut();
            let ly = inner.layers[layer].as_mut().expect("a layer");
            r = ly.wait_ready(at.stream).and_then(|_| ly.dplan.as_ref().expect("planned on the GPU").upload(&ly.cache.expert_in()));
        }
        let own = t0.elapsed().as_secs_f64() * 1e3 - (spent(self) - before);
        if let Some(ly) = self.inner.borrow_mut().layers.get_mut(layer).and_then(Option::as_mut) {
            ly.stats.plan_ms += own.max(0.0);
        }
        r
    }

    /// `run_early_at`'s call.
    ///
    /// # Safety
    /// As `run`.
    unsafe fn run_call(
        &self,
        layer: usize,
        at: At,
        call: &Call,
        early: &mut dyn FnMut() -> Result<()>,
        body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>,
    ) -> Result<()> {
        let s = at.stream;
        let pairs = call.pairs;
        let (needed, host_ids, host_wts, hybrid) = {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let mode = inner.offload.mode;
            let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
            cuda.use_worker = inner.offload.async_copies;
            let ly = inner.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
            let hidden = ly.hidden();
            let hybrid = mode != Mode::Copy
                && ly.cpu.as_ref().is_some_and(|c| matches!(c.half, HostHalf::Rows(_) | HostHalf::Cb2(..) | HostHalf::Fp4 { .. }))
                && (1..=CPU_TOKENS).contains(&call.tokens)
                && pairs.is_multiple_of(call.tokens);
            cuda.reserve(pairs, if hybrid { call.tokens * hidden * 2 } else { 0 }, if hybrid { call.tokens * hidden * 4 } else { 0 }, at)?;
            crate::tmark!(s, "store: choices requested");
            copy_d2h_async(cuda.ids.ptr, call.ids, pairs * 8, s)?;
            if hybrid {
                copy_d2h_async(cuda.wts.ptr, call.wts, pairs * 4, s)?;
                copy_d2h_async(cuda.z.ptr, call.z, call.tokens * hidden * 2, s)?;
            }
            // (the host waits for the copies of the choices alone, an event after them: what `early` queues runs on while the host plans)
            let got = cuda.event()?;
            got.record(s)?;
            early()?;
            let t0 = Instant::now();
            got.wait_host()?;
            cuda.spare.push(got);
            let waited = t0.elapsed().as_secs_f64() * 1e3;
            inner.stats.wait_ms += waited;
            ly.stats.called(call.tokens, call.pairs, waited);
            crate::tmark!(s, "store: choices read");
            // SAFETY: the pinned buffers hold `pairs` int64 and fp32, written by the copies that have completed.
            let host_ids: Vec<i64> = std::slice::from_raw_parts(cuda.ids.ptr as *const i64, pairs).to_vec();
            let host_wts: Vec<f32> = if hybrid { std::slice::from_raw_parts(cuda.wts.ptr as *const f32, pairs).to_vec() } else { Vec::new() };
            let needed = by_rows(distinct(&host_ids, ly.cache.experts()), &host_ids, ly.cache.experts());
            // (a call whose misses are more than the places its hits leave, copied whole in waves where that is quicker than the host's share of them)
            let hybrid = hybrid && {
                let experts = ly.cache.experts();
                let mut pairs_of = vec![0usize; experts];
                host_ids.iter().filter(|&&e| e >= 0 && (e as usize) < experts).for_each(|&e| pairs_of[e as usize] += 1);
                let misses: Vec<Miss> = needed
                    .iter()
                    .filter(|&&e| ly.cache.slot_of(e as usize).is_none())
                    .map(|&e| Miss { expert: e, pairs: pairs_of[e as usize], hot: false })
                    .collect();
                let room = ly.cache.slots() - (needed.len() - misses.len());
                !waves_quicker(mode, &misses, room, &Costs { backlog: cuda.backlog(inner.expert_bytes, &inner.costs), ..inner.costs })
            };
            (needed, host_ids, host_wts, hybrid)
        };
        if hybrid {
            let overlap = {
                let inner = self.inner.borrow();
                let ly = inner.layers[layer].as_ref().expect("a layer");
                inner.offload.async_copies
                    && inner.predict.is_none()
                    && ly.tier.is_none()
                    && matches!(ly.cpu.as_ref().map(|c| &c.half), Some(HostHalf::Cb2(..)))
            };
            if overlap {
                return self.run_hybrid_overlap(layer, at, call, &needed, &host_ids, &host_wts, false, body);
            }
            return self.run_hybrid(layer, at, call, &needed, &host_ids, &host_wts, body);
        }
        let waves = self.inner.borrow().layers[layer].as_ref().expect("a layer").cache.waves(&needed);
        for (w, wave) in waves.iter().enumerate() {
            {
                let mut guard = self.inner.borrow_mut();
                let inner = &mut *guard;
                if w > 0 {
                    // (the last wave's places and the host buffer of its pairs' places are free once the engine's stream has run it: a wait for the GPU)
                    let t0 = Instant::now();
                    at.sync()?;
                    if let Some(ly) = inner.layers[layer].as_mut() {
                        ly.stats.wait_ms += t0.elapsed().as_secs_f64() * 1e3;
                    }
                }
                let cuda = inner.cuda.as_mut().expect("the copy stream");
                let ly = inner.layers[layer].as_mut().expect("a layer");
                let got = ly.cache.request(wave)?;
                inner.stats.reads += wave.len() as u64;
                inner.stats.hits += got.hits as u64;
                inner.stats.misses += got.loads.len() as u64;
                inner.stats.copied += got.loads.len() as u64;
                inner.stats.waves += (w > 0) as u64;
                let bytes = place(cuda, ly, &got.loads, &mut inner.stats, &mut inner.costs, inner.expert_bytes).inspect_err(|_| ly.forget())?;
                // (a copy mode's misses are the waves' copies, each expert copied once over the waves: `LayerCache::waves`)
                let missed: u64 = got.loads.iter().map(|l| ly.bytes_of(l.0 as usize)).sum();
                let st = &mut ly.stats;
                (st.reads, st.hits, st.copies, st.waves, st.copy_bytes, st.missed_bytes, st.whole, st.whole_bytes) = (
                    st.reads + wave.len() as u64,
                    st.hits + got.hits as u64,
                    st.copies + got.loads.len() as u64,
                    st.waves + (w > 0) as u64,
                    st.copy_bytes + bytes,
                    st.missed_bytes + missed,
                    st.whole + (w == 0) as u64,
                    st.whole_bytes + bytes,
                );
                ly.wait_ready(s)?;
                for &e in wave {
                    ly.place[e as usize] = ly.cache.slot_of(e as usize).expect("an expert in a place") as i64;
                }
                // SAFETY: the pinned buffer holds `pairs` int64 and the engine's stream does not read it now.
                fill_way(&host_ids, &ly.place, std::slice::from_raw_parts_mut(cuda.way.ptr as *mut i64, pairs));
                for &e in wave {
                    ly.place[e as usize] = -1;
                }
                copy_h2d_async(call.way, cuda.way.ptr, pairs * 8, s)?;
            }
            if w == 0 {
                // (after this layer's own copies on the copy stream, and before its products are queued)
                self.predicted(layer, &needed, [Some(layer), None])?;
            }
            let all = Hybrid { gpu: true, ids: 0, extra: 0 };
            body(Stage::Products, w, waves.len(), &all)?;
            body(Stage::Sum, w, waves.len(), &all)?;
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            inner.layers[layer].as_ref().expect("a layer").done.record(s)?;
            inner.cuda.as_mut().expect("the copy stream").count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
        }
        Ok(())
    }

    /// `run_hybrid` for a layer of this format whose copies are made by the copy thread: the host's pairs are computed on a thread of their own from the moment the plan is made, beside the copies this call and its
    /// promotions queue, the call's products on the GPU and the promotions' bookkeeping, instead of after them; the sums go up when the host's pairs are done, and the layer has the one wait it had.
    ///
    /// # Safety
    /// As `run`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_hybrid_overlap(
        &self,
        layer: usize,
        at: At,
        call: &Call,
        needed: &[u16],
        host_ids: &[i64],
        host_wts: &[f32],
        parts: bool,
        body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>,
    ) -> Result<()> {
        use crate::cpu_cb::{GlmExpert, GlmPair};
        let s = at.stream;
        let (pairs, k) = (call.pairs, call.pairs / call.tokens);
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let (cuda, ly) = (inner.cuda.as_mut().expect("the copy stream"), inner.layers[layer].as_mut().expect("a layer"));
        // the plan
        ly.calls += 1;
        let (experts, now) = (ly.cache.experts(), ly.calls);
        let mut pairs_of = vec![0usize; experts];
        host_ids.iter().filter(|&&e| e >= 0 && (e as usize) < experts).for_each(|&e| pairs_of[e as usize] += 1);
        let last = &mut ly.cpu.as_mut().expect("a host half").last;
        cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
        let costs = Costs { backlog: cuda.backlog(inner.expert_bytes, &inner.costs), ..inner.costs };
        let served = serve(&mut ly.cache, needed, &pairs_of, last, now, &inner.offload, &costs)?;
        inner.stats.reads += needed.len() as u64;
        inner.stats.hits += served.hits as u64;
        inner.stats.misses += served.misses as u64;
        inner.stats.copied += served.loads.len() as u64;
        inner.stats.cpu += served.kept.len() as u64;
        let missed = ly.missed_bytes(&served);
        ly.stats.served(&served, 0, missed);
        for &e in &served.wanted {
            ly.place[e as usize] = ly.cache.slot_of(e as usize).expect("an expert in a place") as i64;
        }
        // SAFETY: the pinned buffer holds `pairs` int64 and the engine's stream does not read it until the copy below.
        let way = std::slice::from_raw_parts_mut(cuda.way.ptr as *mut i64, pairs);
        fill_way(host_ids, &ly.place, way);
        for &e in &served.wanted {
            ly.place[e as usize] = -1;
        }
        let gpu = way.iter().filter(|&&w| w >= 0).count();
        let mut mine: Vec<usize> = (0..pairs).filter(|&j| way[j] < 0 && host_ids[j] >= 0 && (host_ids[j] as usize) < experts).collect();
        let kept = mine.len();
        if inner.offload.skip > 0.0 {
            mine.retain(|&j| host_wts[j] >= inner.offload.skip);
        }
        // (a verification in parts: its guesses' rows against their own tail)
        let later = if parts && call.tokens > 1 && inner.offload.skip_rank_later > 0 { inner.offload.skip_rank_later } else { inner.offload.skip_rank };
        tail_skip_rows(&mut mine, host_ids, host_wts, k, inner.offload.skip_rank, later);
        let dropped = kept - mine.len();
        inner.stats.skipped += dropped as u64;
        let (host_experts, host_bytes) = ly.host_reads(host_ids, &mine);
        let hidden = ly.hidden();
        let Some(CpuSide { half: HostHalf::Cb2(limit, ..), .. }) = &ly.cpu else { unreachable!("a layer of this format") };
        let limit = *limit;
        // the host's pairs: the experts' views (the mapped file's bytes, which `fetch` does not touch) and the pairs, on a thread of their own
        // (the pages of the host's experts asked for ahead, but not in a decode step in parts: two system calls a pair on its critical path, for pages the filled page tables already map)
        if !parts {
            for &j in &mine {
                let p = &ly.packs[0];
                for kk in 0..2 {
                    p.will_need(kk, host_ids[j] as usize);
                }
            }
        }
        let views: Vec<GlmExpert> = mine.iter().map(|&j| ly.glm_expert(host_ids[j] as usize, limit)).collect::<Result<_>>()?;
        // SAFETY: the views point into the layer's mapped file, which the layer keeps mapped for as long as it lives and nothing writes; `fetch` below reads it and writes the places on the GPU.
        let views: Vec<GlmExpert<'static>> = std::mem::transmute::<Vec<GlmExpert>, Vec<GlmExpert<'static>>>(views);
        // SAFETY: the pinned buffer of z is written by the copy the wait above saw complete and is not touched until the next call.
        let zs: &'static [u16] = std::slice::from_raw_parts(cuda.z.ptr as *const u16, call.tokens * hidden);
        let mut out = std::mem::take(&mut cuda.out);
        out.resize(mine.len() * hidden, 0.0);
        let (threads, pin) = (inner.offload.threads, inner.offload.pin);
        let kernel = crate::cpu_hb::checked_kernel();
        let weights: Vec<(usize, f32)> = mine.iter().map(|&j| (j / k, host_wts[j])).collect();
        let t0 = Instant::now();
        let pending = if mine.is_empty() {
            None
        } else {
            let (tx, rx) = channel();
            let job: Box<dyn FnOnce() + Send> = Box::new(move || {
                let gp: Vec<GlmPair> = views.iter().zip(&weights).map(|(e, &(token, weight))| GlmPair { expert: e, token, weight }).collect();
                let r = crate::cpu_cb::experts_pinned(kernel, Z::Bf16(zs), &gp, &mut out, threads, pin).map(|_| ());
                let _ = tx.send((r, out, t0.elapsed().as_secs_f64() * 1e6));
            });
            cuda.hands()?.send(job).map_err(|_| Error("the host's thread is gone".into()))?;
            Some(rx)
        };
        // beside it: the copies this call needs, the places of the pairs, the products on the GPU, the promotions
        let products = (|| -> Result<()> {
            let bytes = fetch(cuda, ly, &served.loads, &mut inner.stats).inspect_err(|_| ly.forget())?;
            ly.stats.copy_bytes += bytes;
            ly.wait_ready(s)?;
            copy_h2d_async(call.way, cuda.way.ptr, pairs * 8, s)?;
            crate::tmark!(s, "store: planned");
            body(Stage::Products, 0, 1, &Hybrid { gpu: gpu > 0, ids: 0, extra: 0 })?;
            crate::tmark!(s, "store: products queued");
            ly.done.record(s)?;
            let backlog = cuda.backlog(inner.expert_bytes, &inner.costs);
            if !served.promote.is_empty() && backlog <= BACKLOG_US {
                let got = ly.cache.hint(&served.promote)?;
                inner.stats.promoted += got.loads.len() as u64;
                let bytes = fetch(cuda, ly, &got.loads, &mut inner.stats).inspect_err(|_| ly.forget())?;
                ly.stats.ahead(got.loads.len(), bytes);
            }
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)
        })();
        // (the host's pairs are waited for even if the rest failed: they write into `out`, which the call keeps)
        let host_result = pending.map(|rx| recv_spin(&rx));
        products?;
        let hy = Hybrid { gpu: gpu > 0, ids: 0, extra: 0 };
        let Some(joined) = host_result else {
            // (no pair left for the host: the pairs left out have no place and no host sum, and the sum leaves them out by their places)
            ly.stats.hosted(0, 0, dropped, 0, 0.0);
            cuda.out = Vec::new();
            if parts {
                // (the sum of the parts' graph adds the host's sums always: none this time)
                memset_async(call.part, 0, call.tokens * hidden * 4, s)?;
                return Ok(());
            }
            body(Stage::Sum, 0, 1, &Hybrid { ids: if dropped > 0 { call.way } else { 0 }, ..hy })?;
            crate::tmark!(s, "store: sum queued");
            return Ok(());
        };
        let (r, out, us) = joined?;
        cuda.out = out;
        r?;
        inner.stats.cpu_pairs += mine.len() as u64;
        inner.stats.cpu_ms += us / 1e3;
        ly.stats.hosted(host_experts, mine.len(), dropped, host_bytes, us / 1e3);
        if !inner.frozen {
            inner.costs.computed(host_experts, mine.len(), us);
        }
        send_sums(cuda, mine.iter().map(|&j| j / k), hidden, call.tokens, call.part, s)?;
        crate::tmark!(s, "store: host sums sent");
        if parts {
            return Ok(());
        }
        body(Stage::Sum, 0, 1, &Hybrid { gpu: hy.gpu, ids: call.way, extra: call.part })?;
        crate::tmark!(s, "store: sum queued");
        Ok(())
    }

    /// Whether a decode step's calls of `layer` can be served in parts (`request_parts`, `run_parts`): the copies are made by the copy thread, the layer has a host half of codebook rows and no disk tier, no predictor
    /// is set and the mode uses the host.
    pub fn supports_parts(&self, layer: usize) -> bool {
        let inner = self.inner.borrow();
        let Some(ly) = inner.layers.get(layer).and_then(Option::as_ref) else { return false };
        inner.offload.async_copies
            && inner.offload.mode != Mode::Copy
            && inner.predict.is_none()
            && ly.tier.is_none()
            && matches!(ly.cpu.as_ref().map(|c| &c.half), Some(HostHalf::Cb2(..)))
    }

    /// The host's buffers right for calls of `tokens` tokens of `pairs` pairs: a capture of the parts may not reallocate them (that waits for the stream).
    pub fn prepare_parts(&self, layer: usize, dev: &Dev, tokens: usize, pairs: usize) -> Result<()> {
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
        let hidden = inner.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?.hidden();
        cuda.reserve(pairs, tokens * hidden * 2, tokens * hidden * 4, At::of(dev))
    }

    /// The next `run_parts` waits for the mapped host word at `flag` to read `want` instead of synchronizing the stream (the front's last kernel moves it once the host's buffers are written).
    pub fn arm_ready(&self, flag: *const u32, want: u32) {
        self.inner.borrow_mut().ready = Some((flag as usize, want));
    }

    /// The host's buffers of the router's choices (int64 [pairs]), their weights (fp32 [pairs]) and the experts' input (bf16 [tokens, hidden]), mapped for the device: a front's kernels write them in place of
    /// `request_parts`' copies (after `prepare_parts`, which makes them; their addresses then stay, as a graph's kernels name them).
    pub fn parts_host(&self) -> Result<[u64; 3]> {
        let mut inner = self.inner.borrow_mut();
        let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
        cuda.fixed = true;
        Ok([cuda.ids.ptr as u64, cuda.wts.ptr as u64, cuda.z.ptr as u64])
    }

    /// The copies of the router's choices, their weights and the experts' input to the host's buffers, queued on the engine's stream (captured in a graph, they are the graph's nodes).
    ///
    /// # Safety
    /// As `run`; `prepare_parts` has been called for these sizes.
    pub unsafe fn request_parts(&self, layer: usize, dev: &Dev, call: &Call) -> Result<()> {
        let s = dev.stream;
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
        let hidden = inner.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?.hidden();
        copy_d2h_async(cuda.ids.ptr, call.ids, call.pairs * 8, s)?;
        copy_d2h_async(cuda.wts.ptr, call.wts, call.pairs * 4, s)?;
        copy_d2h_async(cuda.z.ptr, call.z, call.tokens * hidden * 2, s)
    }

    /// The host's part of a decode step's call of `layer`: the wait for the choices that `request_parts` queued (the engine's stream is waited for whole), `early` (the GPU work that needs no choices, queued to run while
    /// the host plans), the plan and the copies, `products` for the pairs in places, the host's pairs beside them, and their sums sent up to `call.part` (zeros when the host has none), so that the graph of the part after
    /// can add `call.part` and leave out the pairs without a place (`call.way`) whatever the call was.
    ///
    /// # Safety
    /// As `run`.
    pub unsafe fn run_parts(
        &self,
        layer: usize,
        dev: &Dev,
        call: &Call,
        early: &mut dyn FnMut() -> Result<()>,
        products: &mut dyn FnMut(&Hybrid) -> Result<()>,
    ) -> Result<()> {
        let pairs = call.pairs;
        let (needed, host_ids, host_wts) = {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
            cuda.use_worker = true;
            cuda.ordinal = Some(dev.ordinal);
            let t0 = Instant::now();
            match inner.ready.take() {
                // (the front's last kernel moves the word after writing the host's buffers: looked at until it does, the stream's synchronize after 20 ms of it)
                Some((flag, want)) => {
                    let word = flag as *const u32;
                    while std::ptr::read_volatile(word) != want {
                        if t0.elapsed().as_millis() > 20 {
                            dev.sync()?;
                            if std::ptr::read_volatile(word) != want {
                                return err(format!("the front's word is {}, where {want} was awaited", std::ptr::read_volatile(word)));
                            }
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                None => dev.sync()?,
            }
            let waited = t0.elapsed().as_secs_f64() * 1e3;
            inner.stats.wait_ms += waited;
            crate::tmark!(dev.stream, "store: choices read");
            // SAFETY: the pinned buffers hold `pairs` int64 and fp32, written by the copies the engine's stream has run.
            let host_ids: Vec<i64> = std::slice::from_raw_parts(cuda.ids.ptr as *const i64, pairs).to_vec();
            let host_wts: Vec<f32> = std::slice::from_raw_parts(cuda.wts.ptr as *const f32, pairs).to_vec();
            let ly = inner.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
            ly.stats.called(call.tokens, call.pairs, waited);
            (distinct(&host_ids, ly.cache.experts()), host_ids, host_wts)
        };
        early()?;
        self.run_hybrid_overlap(layer, At::of(dev), call, &needed, &host_ids, &host_wts, true, &mut |stage, _, _, hy| match stage {
            Stage::Products => products(hy),
            Stage::Sum => Ok(()),
        })
    }

    /// The places of `layer`'s experts as a decode step's products read them on the GPU (`run_parts_mapped`): map[e] the place of expert e (the layer's numbers), -1 for none, made the cache's once the copies queued
    /// into the places so far have run (`wait`: waited for; else the map is left as it is where they have not). Whether it was made the cache's.
    pub fn map_sync(&self, layer: usize, map: &mut [i32], wait: bool) -> Result<bool> {
        let mut guard = self.inner.borrow_mut();
        let ly = guard.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
        if map.len() != ly.cache.experts() {
            return err(format!("a map of {} experts for a layer of {}", map.len(), ly.cache.experts()));
        }
        // (the copy thread's last batch queued first: its `ready` is then the one recorded)
        if !wait && (ly.job.as_ref().is_some_and(|j| !j.submitted()) || !ly.ready.done()?) {
            return Ok(false);
        }
        ly.join_job()?;
        ly.ready.wait_host()?;
        for (e, m) in map.iter_mut().enumerate() {
            *m = ly.cache.slot_of(e).map_or(-1, |p| p as i32);
        }
        Ok(true)
    }

    /// A decode step's call of `layer` whose products the front's graph ran for the pairs whose experts have a place in `map` (`map_sync`'s; the front's last kernel read it into `call.way`): the wait for the choices
    /// (`arm_ready`'s word), the pairs of the experts with no place in `map` computed on the host, the experts in places used (their order of use), and the misses kept a short while before copied into places behind
    /// the products (`serve`'s promotions, as `run_parts` in the mode `Cpu`): the places they take out of `map` now, the experts copied in left out of it until their copies have run (`map_sync`). `next` (the GPU's
    /// next work, which waits for the host's sums on the GPU) is queued while the host computes; then the host's sums are written to `sums` (fp32 [tokens, hidden]; nothing where there are none). Whether any expert was
    /// copied, and whether `sums` holds sums.
    ///
    /// # Safety
    /// As `run`; `prepare_parts` and `parts_host` have been called, the engine's stream has the call's products queued, no kernel reads `map` until this returns, and none reads `sums` before it returns.
    pub unsafe fn run_parts_mapped(
        &self,
        layer: usize,
        dev: &Dev,
        call: &Call,
        map: &mut [i32],
        next: &mut dyn FnMut() -> Result<()>,
        sums: &mut [f32],
    ) -> Result<(bool, bool)> {
        use crate::cpu_cb::{GlmExpert, GlmPair};
        let s = dev.stream;
        let (pairs, k) = (call.pairs, call.pairs / call.tokens);
        // the plan and the host's pairs started on their thread, the store borrowed (`next` captures graphs that ask it for its buffers)
        let (pending, mine, hidden, promote, recorded, host_experts) = {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
            cuda.use_worker = true;
            cuda.ordinal = Some(dev.ordinal);
            let t0 = Instant::now();
            match inner.ready.take() {
                // (the front's last kernel moves the word after writing the host's buffers: looked at until it does, the stream's synchronize after 20 ms of it)
                Some((flag, want)) => {
                    let word = flag as *const u32;
                    while std::ptr::read_volatile(word) != want {
                        if t0.elapsed().as_millis() > 20 {
                            dev.sync()?;
                            if std::ptr::read_volatile(word) != want {
                                return err(format!("the front's word is {}, where {want} was awaited", std::ptr::read_volatile(word)));
                            }
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                None => dev.sync()?,
            }
            let waited = t0.elapsed().as_secs_f64() * 1e3;
            inner.stats.wait_ms += waited;
            crate::tmark!(s, "store: choices read");
            // SAFETY: the pinned buffers hold `pairs` int64 and fp32, written by the front's kernel before it moved the word.
            let host_ids: Vec<i64> = std::slice::from_raw_parts(cuda.ids.ptr as *const i64, pairs).to_vec();
            let host_wts: Vec<f32> = std::slice::from_raw_parts(cuda.wts.ptr as *const f32, pairs).to_vec();
            let ly = inner.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
            ly.stats.called(call.tokens, call.pairs, waited);
            ly.calls += 1;
            let (experts, now) = (ly.cache.experts(), ly.calls);
            if map.len() != experts {
                return err(format!("a map of {} experts for a layer of {experts}", map.len()));
            }
            let needed = distinct(&host_ids, experts);
            let mut pairs_of = vec![0usize; experts];
            host_ids.iter().filter(|&&e| e >= 0 && (e as usize) < experts).for_each(|&e| pairs_of[e as usize] += 1);
            let Some(CpuSide { half: HostHalf::Cb2(limit, ..), last, .. }) = ly.cpu.as_mut() else {
                return err("a layer without a host half of format 0 to 4 rows".to_string());
            };
            let limit = *limit;
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
            // (every miss the host's, the ones it kept a short while before promoted; the hits the map's experts and the experts copied in that it does not have yet, whose pairs are the host's too)
            let off = Offload { mode: Mode::Cpu, ..inner.offload };
            let served = serve(&mut ly.cache, &needed, &pairs_of, last, now, &off, &inner.costs)?;
            inner.stats.reads += needed.len() as u64;
            inner.stats.hits += served.hits as u64;
            inner.stats.misses += served.misses as u64;
            inner.stats.cpu += served.kept.len() as u64;
            let missed = ly.missed_bytes(&served);
            ly.stats.served(&served, 0, missed);
            // the host's pairs: those of the experts without a place in the map (what the GPU left out), less the ones the skips leave out
            let mut mine: Vec<usize> = (0..pairs).filter(|&j| host_ids[j] >= 0 && (host_ids[j] as usize) < experts && map[host_ids[j] as usize] < 0).collect();
            let kept = mine.len();
            if inner.offload.skip > 0.0 {
                mine.retain(|&j| host_wts[j] >= inner.offload.skip);
            }
            let later = if call.tokens > 1 && inner.offload.skip_rank_later > 0 { inner.offload.skip_rank_later } else { inner.offload.skip_rank };
            tail_skip_rows(&mut mine, &host_ids, &host_wts, k, inner.offload.skip_rank, later);
            inner.stats.skipped += (kept - mine.len()) as u64;
            let (host_experts, host_bytes) = ly.host_reads(&host_ids, &mine);
            ly.stats.hosted(host_experts, mine.len(), kept - mine.len(), host_bytes, 0.0);
            let hidden = ly.hidden();
            if sums.len() < call.tokens * hidden {
                return err(format!("host sums of {} values for {} tokens of {hidden}", sums.len(), call.tokens));
            }
            // the host's pairs on its thread (`run_hybrid_overlap`'s)
            let views: Vec<GlmExpert> = mine.iter().map(|&j| ly.glm_expert(host_ids[j] as usize, limit)).collect::<Result<_>>()?;
            // SAFETY: the views point into the layer's mapped file, which the layer keeps mapped for as long as it lives and nothing writes; `fetch` below reads it and writes the places on the GPU.
            let views: Vec<GlmExpert<'static>> = std::mem::transmute::<Vec<GlmExpert>, Vec<GlmExpert<'static>>>(views);
            // SAFETY: the pinned buffer of z is written by the front's kernel before it moved the word, and the next front (the next layer's, which writes it) runs after the GPU has the host's sums.
            let zs: &'static [u16] = std::slice::from_raw_parts(cuda.z.ptr as *const u16, call.tokens * hidden);
            let mut out = std::mem::take(&mut cuda.out);
            out.resize(mine.len() * hidden, 0.0);
            let (threads, pin) = (inner.offload.threads, inner.offload.pin);
            let kernel = crate::cpu_hb::checked_kernel();
            let weights: Vec<(usize, f32)> = mine.iter().map(|&j| (j / k, host_wts[j])).collect();
            let t1 = Instant::now();
            let pending = if mine.is_empty() {
                None
            } else {
                let (tx, rx) = channel();
                let job: Box<dyn FnOnce() + Send> = Box::new(move || {
                    let gp: Vec<GlmPair> = views.iter().zip(&weights).map(|(e, &(token, weight))| GlmPair { expert: e, token, weight }).collect();
                    let r = crate::cpu_cb::experts_pinned(kernel, Z::Bf16(zs), &gp, &mut out, threads, pin).map(|_| ());
                    let _ = tx.send((r, out, t1.elapsed().as_secs_f64() * 1e6));
                });
                cuda.hands()?.send(job).map_err(|_| Error("the host's thread is gone".into()))?;
                Some(rx)
            };
            // (the products queued before are the places' last use, which the promotions' copies wait for)
            let recorded = ly.done.record(s);
            (pending, mine, hidden, served.promote, recorded, host_experts)
        };
        // the GPU's next work queued while the host computes (unless the call failed so far: its error then)
        let launched = match &recorded {
            Ok(()) => next(),
            Err(_) => Ok(()),
        };
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
        // beside the host's pairs: the promotions, behind the products
        let moved = (|| -> Result<bool> {
            recorded?;
            let ly = inner.layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| Error(format!("no layer {layer} in the expert cache")))?;
            let backlog = cuda.backlog(inner.expert_bytes, &inner.costs);
            let mut moved = false;
            if !promote.is_empty() && backlog <= BACKLOG_US {
                let got = ly.cache.hint(&promote)?;
                inner.stats.promoted += got.loads.len() as u64;
                moved = !got.loads.is_empty();
                // (the places taken out of the map whatever the copies do: an expert whose place another takes is in none)
                let fetched = fetch(cuda, ly, &got.loads, &mut inner.stats).inspect_err(|_| ly.forget());
                ly.stats.ahead(got.loads.len(), *fetched.as_ref().unwrap_or(&0));
                for (e, m) in map.iter_mut().enumerate() {
                    if *m >= 0 && ly.cache.slot_of(e) != Some(*m as usize) {
                        *m = -1;
                    }
                }
                fetched?;
            }
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
            Ok(moved)
        })();
        // (the host's pairs are waited for even if the rest failed: they write into `out`, which the call keeps)
        let host_result = pending.map(|rx| recv_spin(&rx));
        let Some(joined) = host_result else {
            let moved = moved?;
            launched?;
            cuda.out = Vec::new();
            return Ok((moved, false));
        };
        let (r, out, us) = joined?;
        let moved = moved?;
        launched?;
        r?;
        let sums = &mut sums[..call.tokens * hidden];
        sums.fill(0.0);
        for (row, &j) in out.chunks_exact(hidden).zip(&mine) {
            sums[j / k * hidden..][..hidden].iter_mut().zip(row).for_each(|(a, b)| *a += *b);
        }
        cuda.out = out;
        inner.stats.cpu_pairs += mine.len() as u64;
        inner.stats.cpu_ms += us / 1e3;
        if let Some(ly) = inner.layers.get_mut(layer).and_then(Option::as_mut) {
            ly.stats.host_ms += us / 1e3;
        }
        if !inner.frozen {
            inner.costs.computed(host_experts, mine.len(), us);
        }
        Ok((moved, true))
    }

    /// `run` for a layer with a host half (see there): one wave, the misses served as the mode says.
    ///
    /// # Safety
    /// As `run`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_hybrid(
        &self,
        layer: usize,
        at: At,
        call: &Call,
        needed: &[u16],
        host_ids: &[i64],
        host_wts: &[f32],
        body: &mut dyn FnMut(Stage, usize, usize, &Hybrid) -> Result<()>,
    ) -> Result<()> {
        let s = at.stream;
        let (pairs, k) = (call.pairs, call.pairs / call.tokens);
        let dropped;
        // the plan: the experts that get a place (the hits, and the misses `plan` copies), the pairs of the others, the misses worth a place behind the call
        let (gpu, mine, promote) = {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().expect("the copy stream");
            let ly = inner.layers[layer].as_mut().expect("a layer");
            ly.calls += 1;
            let (experts, now) = (ly.cache.experts(), ly.calls);
            let mut pairs_of = vec![0usize; experts];
            host_ids.iter().filter(|&&e| e >= 0 && (e as usize) < experts).for_each(|&e| pairs_of[e as usize] += 1);
            let last = &mut ly.cpu.as_mut().expect("a host half").last;
            // (the batches that have finished since the last call are counted first: the plan weighs the copying that is really still queued)
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
            let costs = Costs { backlog: cuda.backlog(inner.expert_bytes, &inner.costs), ..inner.costs };
            let served = serve(&mut ly.cache, needed, &pairs_of, last, now, &inner.offload, &costs)?;
            inner.stats.reads += needed.len() as u64;
            inner.stats.hits += served.hits as u64;
            inner.stats.misses += served.misses as u64;
            inner.stats.copied += served.loads.len() as u64;
            inner.stats.cpu += served.kept.len() as u64;
            let bytes = fetch(cuda, ly, &served.loads, &mut inner.stats).inspect_err(|_| ly.forget())?;
            let missed = ly.missed_bytes(&served);
            ly.stats.served(&served, bytes, missed);
            ly.wait_ready(s)?;
            for &e in &served.wanted {
                ly.place[e as usize] = ly.cache.slot_of(e as usize).expect("an expert in a place") as i64;
            }
            // SAFETY: the pinned buffer holds `pairs` int64 and the engine's stream does not read it now.
            let way = std::slice::from_raw_parts_mut(cuda.way.ptr as *mut i64, pairs);
            fill_way(host_ids, &ly.place, way);
            for &e in &served.wanted {
                ly.place[e as usize] = -1;
            }
            copy_h2d_async(call.way, cuda.way.ptr, pairs * 8, s)?;
            let gpu = way.iter().filter(|&&w| w >= 0).count();
            let mut mine: Vec<usize> = (0..pairs).filter(|&j| way[j] < 0 && host_ids[j] >= 0 && (host_ids[j] as usize) < experts).collect();
            let kept = mine.len();
            if inner.offload.skip > 0.0 {
                mine.retain(|&j| host_wts[j] >= inner.offload.skip);
            }
            tail_skip(&mut mine, host_ids, host_wts, k, inner.offload.skip_rank);
            dropped = kept - mine.len();
            inner.stats.skipped += dropped as u64;
            let (host_experts, host_bytes) = ly.host_reads(host_ids, &mine);
            ly.stats.hosted(host_experts, mine.len(), dropped, host_bytes, 0.0);
            (gpu, mine, served.promote)
        };
        self.predicted(layer, needed, [Some(layer), None])?;
        let hy = Hybrid { gpu: gpu > 0, ids: 0, extra: 0 };
        crate::tmark!(s, "store: planned");
        body(Stage::Products, 0, 1, &hy)?;
        crate::tmark!(s, "store: products queued");
        {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().expect("the copy stream");
            let ly = inner.layers[layer].as_mut().expect("a layer");
            // (the places are free of this call's products once they have run: the copies behind it wait for them)
            ly.done.record(s)?;
            let backlog = cuda.backlog(inner.expert_bytes, &inner.costs);
            if !promote.is_empty() && backlog <= BACKLOG_US {
                let got = ly.cache.hint(&promote)?;
                inner.stats.promoted += got.loads.len() as u64;
                let bytes = fetch(cuda, ly, &got.loads, &mut inner.stats).inspect_err(|_| ly.forget())?;
                ly.stats.ahead(got.loads.len(), bytes);
            }
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
        }
        if mine.is_empty() {
            // (the pairs left out have no place and no host sum: the sum leaves them out by their places)
            body(Stage::Sum, 0, 1, &Hybrid { ids: if dropped > 0 { call.way } else { 0 }, ..hy })?;
            crate::tmark!(s, "store: sum queued");
            return Ok(());
        }
        {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().expect("the copy stream");
            let ly = inner.layers[layer].as_mut().expect("a layer");
            let hidden = ly.hidden();
            cuda.out.resize(mine.len() * hidden, 0.0);
            // SAFETY: the pinned buffers hold the tokens' rows of z and of sums, and the engine's stream does not read the sums until the copy below.
            let z = std::slice::from_raw_parts(cuda.z.ptr as *const u16, call.tokens * hidden);
            let t0 = Instant::now();
            if let Some(CpuSide { half: HostHalf::Cb2(limit, ..), .. }) = &ly.cpu {
                for &j in &mine {
                    let p = &ly.packs[0];
                    for k in 0..2 {
                        p.will_need(k, host_ids[j] as usize);
                    }
                }
                let experts: Vec<crate::cpu_cb::GlmExpert> = mine.iter().map(|&j| ly.glm_expert(host_ids[j] as usize, *limit)).collect::<Result<_>>()?;
                let gp: Vec<crate::cpu_cb::GlmPair> =
                    mine.iter().zip(&experts).map(|(&j, e)| crate::cpu_cb::GlmPair { expert: e, token: j / k, weight: host_wts[j] }).collect();
                crate::cpu_cb::experts_pinned(crate::cpu_hb::checked_kernel(), Z::Bf16(z), &gp, &mut cuda.out, inner.offload.threads, inner.offload.pin)?;
            } else if let Some(CpuSide { half: HostHalf::Fp4 { act, .. }, .. }) = &ly.cpu {
                // (each expert's bytes read once for its pairs: `by_expert`)
                let fp: Vec<crate::cpu_fp4::Fp4Pair> =
                    mine.iter().map(|&j| crate::cpu_fp4::Fp4Pair { expert: ly.fp4_expert(host_ids[j] as usize), token: j / k, weight: host_wts[j] }).collect();
                crate::cpu_fp4::experts(
                    *inner.kernel.get_or_insert_with(checked_kernel),
                    z,
                    &fp,
                    &mut cuda.out,
                    *act,
                    inner.offload.threads,
                    inner.offload.pin,
                )?;
            } else {
                let views: Vec<PairView> =
                    mine.iter().map(|&j| PairView { expert: ly.cpu_expert(host_ids[j] as usize), token: j / k, weight: host_wts[j] }).collect();
                experts_views(*inner.kernel.get_or_insert_with(checked_kernel), Z::Bf16(z), &views, &mut cuda.out, inner.offload.threads, inner.offload.pin)?;
            }
            let us = t0.elapsed().as_secs_f64() * 1e6;
            inner.stats.cpu_pairs += mine.len() as u64;
            inner.stats.cpu_ms += us / 1e3;
            ly.stats.host_ms += us / 1e3;
            if !inner.frozen {
                inner.costs.computed(ly.host_reads(host_ids, &mine).0, mine.len(), us);
            }
            send_sums(cuda, mine.iter().map(|&j| j / k), hidden, call.tokens, call.part, s)?;
            crate::tmark!(s, "store: host sums sent");
        }
        body(Stage::Sum, 0, 1, &Hybrid { gpu: hy.gpu, ids: call.way, extra: call.part })?;
        crate::tmark!(s, "store: sum queued");
        Ok(())
    }

    /// Whether a call of `tokens` tokens and `pairs` pairs of a layer in groups is one for `run_groups`: the mode is not `Copy`, the call is `CPU_TOKENS` tokens at most, and every group of the layer that is in the store has
    /// a host half of Hummingbird places. Any other call is `run`'s, a group at a time.
    pub fn serves_groups(&self, groups: &Groups, tokens: usize, pairs: usize) -> bool {
        let inner = self.inner.borrow();
        inner.offload.mode != Mode::Copy
            && (1..=CPU_TOKENS).contains(&tokens)
            && pairs.is_multiple_of(tokens)
            && groups.layers.iter().flatten().all(|&l| {
                inner
                    .layers
                    .get(l)
                    .and_then(Option::as_ref)
                    .is_some_and(|ly| matches!(ly.cpu.as_ref().map(|c| &c.half), Some(HostHalf::Hb(_) | HostHalf::Hbg { .. })))
            })
    }

    /// One call of a model layer whose experts are in groups (`Groups`), served as `run_hybrid` serves a layer with a host half and with the one wait that a layer has: the router's choices, their weights and the experts'
    /// input are read to the host together; the misses of the groups in the store are planned together (`place_groups`), each group's copies are queued and its pairs' places sent to `call.ways[group]`, and `body` is given
    /// `Stage::Products` for each (the group's number, and whether any of its pairs has a place); the pairs of the experts the host keeps are computed from their places in the host tier as they are (`cpu_hb`), in one call,
    /// their weighted sums by token go up to `call.part`, and `body` is given `Stage::Sum` once, with the router's choices leaving the host's pairs out (`Hybrid::ids`, in `call.sum_ids`) and the sums (`Hybrid::extra`), or
    /// with neither (0) where the host kept nothing. The layer's groups that are not in the store are for the caller to run.
    ///
    /// # Safety
    /// As `run`; the buffers of `call` are live device buffers of the sizes it names; the call is one for which `serves_groups`.
    pub unsafe fn run_groups(&self, groups: &Groups, dev: &Dev, call: &GroupsCall, body: &mut dyn FnMut(Stage, usize, &Hybrid) -> Result<()>) -> Result<()> {
        self.run_groups_at(groups, At::of(dev), call, body)
    }

    /// `run_groups` on the GPU `at`.
    ///
    /// # Safety
    /// As `run_groups`.
    pub unsafe fn run_groups_at(&self, groups: &Groups, at: At, call: &GroupsCall, body: &mut dyn FnMut(Stage, usize, &Hybrid) -> Result<()>) -> Result<()> {
        let s = at.stream;
        let (pairs, layers) = (call.pairs, groups.layers);
        if call.tokens == 0 || call.tokens > CPU_TOKENS || !pairs.is_multiple_of(call.tokens) {
            return err(format!("a call of {} tokens and {pairs} pairs is not one for the host's half (1 to {CPU_TOKENS} tokens)", call.tokens));
        }
        let k = pairs / call.tokens;
        // the router's choices, their weights and the experts' input: the one wait of the layer
        let (host_ids, host_wts, hidden) = {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().ok_or_else(|| Error("the expert cache has no layers".into()))?;
            let hidden = layers
                .iter()
                .flatten()
                .find_map(|&l| inner.layers.get(l).and_then(Option::as_ref).map(Resident::hidden))
                .filter(|&h| h > 0)
                .ok_or_else(|| Error("no group of the layer is in the expert cache".into()))?;
            cuda.reserve(pairs, call.tokens * hidden * 4, call.tokens * hidden * 4, at)?;
            copy_d2h_async(cuda.ids.ptr, call.ids, pairs * 8, s)?;
            copy_d2h_async(cuda.wts.ptr, call.wts, pairs * 4, s)?;
            copy_d2h_async(cuda.z.ptr, call.z, call.tokens * hidden * 4, s)?;
            let t0 = Instant::now();
            at.sync()?;
            let waited = t0.elapsed().as_secs_f64() * 1e3;
            inner.stats.wait_ms += waited;
            // (each group's layer a call of the tokens, of its own pairs; the wait the first's)
            for (g, l) in layers.iter().enumerate() {
                if let Some(ly) = l.and_then(|l| inner.layers.get_mut(l)).and_then(Option::as_mut) {
                    let ids = std::slice::from_raw_parts(cuda.ids.ptr as *const i64, pairs);
                    let own = ids.iter().filter(|&&e| e >= 0 && groups.group_of.get(e as usize).is_some_and(|&x| x as usize == g)).count();
                    ly.stats.called(call.tokens, own, if layers[..g].iter().any(Option::is_some) { 0.0 } else { waited });
                }
            }
            // SAFETY: the pinned buffers hold `pairs` int64 and fp32, written by the copies that have completed.
            (
                std::slice::from_raw_parts(cuda.ids.ptr as *const i64, pairs).to_vec(),
                std::slice::from_raw_parts(cuda.wts.ptr as *const f32, pairs).to_vec(),
                hidden,
            )
        };
        // the plan, the copies, and each group's pairs' places
        let (placed, gpu) = {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().expect("the copy stream");
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
            let (offload, costs, per) =
                (inner.offload, Costs { backlog: cuda.backlog(inner.expert_bytes, &inner.costs), ..inner.costs }, inner.expert_bytes.max(1) as f64);
            let placed = {
                let mut slices: [Option<Slice>; 2] = [None, None];
                for (i, ly) in inner.layers.iter_mut().enumerate() {
                    let (Some(ly), Some(g)) = (ly.as_mut(), layers.iter().position(|&l| l == Some(i))) else { continue };
                    ly.calls += 1;
                    let (now, size) = (ly.calls, ly.expert_bytes() as f64 / per);
                    let Resident { cache, cpu, .. } = ly;
                    slices[g] = Some(Slice { cache, last: &mut cpu.as_mut().expect("a host half").last, now, size });
                }
                place_groups(&mut slices, groups.group_of, groups.local_of, &host_ids, &offload, &costs)?
            };
            let mut gpu = [false; 2];
            for g in 0..2 {
                let (Some(l), Some(sv)) = (layers[g], placed.served[g].as_ref()) else { continue };
                let ly = inner.layers[l].as_mut().expect("a layer");
                inner.stats.reads += placed.needed[g].len() as u64;
                inner.stats.hits += sv.hits as u64;
                inner.stats.misses += sv.misses as u64;
                inner.stats.copied += sv.loads.len() as u64;
                inner.stats.cpu += sv.kept.len() as u64;
                let bytes = place(cuda, ly, &sv.loads, &mut inner.stats, &mut inner.costs, inner.expert_bytes).inspect_err(|_| ly.forget())?;
                let missed = ly.missed_bytes(sv);
                ly.stats.served(sv, bytes, missed);
                // (the host's pairs of the group: `cpu_hbg` reads an expert once for its pairs, `cpu_hb` once a pair)
                let mine: Vec<usize> = placed.mine.iter().filter(|&&(h, _)| h == g).map(|&(_, j)| j).collect();
                let (host_experts, host_bytes) = ly.host_reads(&placed.local[g], &mine);
                ly.stats.hosted(host_experts, mine.len(), 0, host_bytes, 0.0);
                ly.wait_ready(s)?;
                // SAFETY: the pinned buffer holds a section of `pairs` int64 for each group and the engine's stream does not read it now.
                let way = std::slice::from_raw_parts_mut((cuda.way.ptr as *mut i64).add(g * pairs), pairs);
                way.copy_from_slice(&placed.ways[g]);
                copy_h2d_async(call.ways[g], way.as_ptr() as *const _, pairs * 8, s)?;
                gpu[g] = way.iter().any(|&w| w >= 0);
            }
            if !placed.mine.is_empty() {
                let sum = std::slice::from_raw_parts_mut((cuda.way.ptr as *mut i64).add(2 * pairs), pairs);
                sum.copy_from_slice(&host_ids);
                placed.mine.iter().for_each(|&(_, j)| sum[j] = -1);
                copy_h2d_async(call.sum_ids, sum.as_ptr() as *const _, pairs * 8, s)?;
            }
            (placed, gpu)
        };
        for (&layer, needed) in layers.iter().zip(&placed.needed) {
            if let Some(l) = layer {
                self.predicted(l, needed, layers)?;
            }
        }
        let any = gpu.iter().any(|&b| b);
        for g in 0..2 {
            let Some(l) = layers[g] else { continue };
            body(Stage::Products, g, &Hybrid { gpu: gpu[g], ids: 0, extra: 0 })?;
            // (the places are free of this group's products once they have run: the copies behind it wait for them)
            self.inner.borrow().layers[l].as_ref().expect("a layer").done.record(s)?;
        }
        {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().expect("the copy stream");
            let backlog = cuda.backlog(inner.expert_bytes, &inner.costs);
            for (&layer, sv) in layers.iter().zip(&placed.served) {
                let (Some(l), Some(sv)) = (layer, sv.as_ref()) else { continue };
                if !sv.promote.is_empty() && backlog <= BACKLOG_US {
                    let ly = inner.layers[l].as_mut().expect("a layer");
                    let got = ly.cache.hint(&sv.promote)?;
                    inner.stats.promoted += got.loads.len() as u64;
                    let bytes = place(cuda, ly, &got.loads, &mut inner.stats, &mut inner.costs, inner.expert_bytes).inspect_err(|_| ly.forget())?;
                    ly.stats.ahead(got.loads.len(), bytes);
                }
            }
            cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
        }
        if placed.mine.is_empty() {
            return body(Stage::Sum, 0, &Hybrid { gpu: any, ids: 0, extra: 0 });
        }
        {
            let mut guard = self.inner.borrow_mut();
            let inner = &mut *guard;
            let cuda = inner.cuda.as_mut().expect("the copy stream");
            // the pairs the host computes, in rounds of the pairs whose experts the host tier's slots hold at once (one round, as ever, where a layer has every expert in memory or its slots are as many as the call has experts), each
            // round's experts read from the packed directory first
            let slots: [usize; 2] =
                [0, 1].map(|g| layers[g].and_then(|l| inner.layers[l].as_ref()?.tier.as_ref().map(|t| t.ram.slots())).unwrap_or(usize::MAX));
            cuda.out.resize(placed.mine.len() * hidden, 0.0);
            // SAFETY: the pinned buffers hold the tokens' rows of z and of sums, and the engine's stream does not read the sums until the copy in `send_sums`.
            let z = std::slice::from_raw_parts(cuda.z.ptr as *const f32, call.tokens * hidden);
            for (start, end, seen) in host_rounds(&placed.mine, &placed.local, slots) {
                if slots != [usize::MAX; 2] {
                    stage_pair(
                        cuda,
                        two_layers(&mut inner.layers, layers[0], layers[1]),
                        [&seen[0], &seen[1]],
                        &mut inner.stats,
                        &mut inner.costs,
                        inner.expert_bytes,
                    )?;
                }
                // (Hummingbird's experts of any mixture of experts: their pairs' products, the family's activation, weighted or normed, src/cpu_hbg.rs)
                let generic = placed.mine[start..end].first().and_then(|&(g, _)| {
                    let ly = inner.layers[layers[g]?].as_ref()?;
                    match ly.cpu.as_ref().map(|c| &c.half) {
                        Some(HostHalf::Hbg { act, finish, .. }) => Some((*act, finish.clone())),
                        _ => None,
                    }
                });
                if let Some((act, finish)) = generic {
                    let experts: Vec<HbgExpert> = placed.mine[start..end]
                        .iter()
                        .map(|&(g, j)| {
                            inner.layers[layers[g].expect("a group in the store")].as_ref().expect("a layer").hbg_expert(placed.local[g][j] as usize)
                        })
                        .collect::<Result<_>>()?;
                    let tokens: Vec<usize> = placed.mine[start..end].iter().map(|&(_, j)| j / k).collect();
                    let wts: Vec<f32> = placed.mine[start..end].iter().map(|&(_, j)| host_wts[j]).collect();
                    let t0 = Instant::now();
                    crate::cpu_hbg::store_pairs(&experts, z, &tokens, &wts, act, &finish, &mut cuda.out[start * hidden..end * hidden], inner.offload.threads)?;
                    let (round, n) = (t0.elapsed().as_secs_f64() * 1e6, experts.len());
                    drop(experts);
                    inner.stats.cpu_pairs += n as u64;
                    inner.stats.cpu_ms += round / 1e3;
                    host_time(&mut inner.layers, layers, &placed.mine[start..end], round / 1e3);
                    if !inner.frozen {
                        inner.costs.computed(n, n, round);
                    }
                    continue;
                }
                let experts: Vec<HbExpert> = placed.mine[start..end]
                    .iter()
                    .map(|&(g, j)| inner.layers[layers[g].expect("a group in the store")].as_ref().expect("a layer").hb_expert(placed.local[g][j] as usize))
                    .collect::<Result<_>>()?;
                let views: Vec<HbPair> =
                    placed.mine[start..end].iter().zip(&experts).map(|(&(_, j), e)| HbPair { expert: e, token: j / k, weight: host_wts[j] }).collect();
                let t0 = Instant::now();
                cpu_hb::experts_pinned(
                    cpu_hb::checked_kernel(),
                    Z::F32(z),
                    &views,
                    &mut cuda.out[start * hidden..end * hidden],
                    inner.offload.threads,
                    inner.offload.pin,
                )?;
                let (round, n) = (t0.elapsed().as_secs_f64() * 1e6, views.len());
                drop(views);
                drop(experts);
                inner.stats.cpu_pairs += n as u64;
                inner.stats.cpu_ms += round / 1e3;
                host_time(&mut inner.layers, layers, &placed.mine[start..end], round / 1e3);
                if !inner.frozen {
                    inner.costs.computed(n, n, round);
                }
            }
            send_sums(cuda, placed.mine.iter().map(|&(_, j)| j / k), hidden, call.tokens, call.part, s)?;
        }
        body(Stage::Sum, 0, &Hybrid { gpu: any, ids: call.sum_ids, extra: call.part })
    }

    /// A step's end (the engine's, after the step's results are read): the copies that have run are counted, and each layer's stats since the step began become the last step's (`last_step`), are added to the
    /// store's (`totals`) and to the process's (`process_totals`, the server's /stats). Returns the step's, by the layer's number in the store (zeros for a number no layer has). Copies still running (promotions
    /// behind the last layers) count in the step that sees them done. A step of no call (an end asked for twice) leaves the last step as it was.
    pub fn end_step(&self) -> Result<Vec<LayerStats>> {
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        let Some(cuda) = inner.cuda.as_mut() else { return Ok(Vec::new()) };
        cuda.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, false)?;
        let ms = std::mem::take(&mut cuda.layer_ms);
        // (the calls the GPU planned since the last step: its counters, the experts' bytes, the host thread's time; the plan's settings from the costs measured so far)
        for ly in inner.layers.iter_mut().flatten() {
            let Some(dp) = ly.dplan.as_mut() else { continue };
            let [calls, _, reads, hits, fills, host, host_pairs, promos] = dp.counters()?.map(u64::from);
            let b = ly.packs.iter().map(|p| p.stride[0] + p.stride[1]).sum::<u64>();
            let st = &mut ly.stats;
            (st.calls, st.reads, st.hits, st.copies, st.host, st.host_pairs, st.ahead) =
                (st.calls + calls, st.reads + reads, st.hits + hits, st.copies + fills, st.host + host, st.host_pairs + host_pairs, st.ahead + promos);
            (st.copy_bytes, st.host_bytes, st.missed_bytes, st.ahead_bytes) =
                (st.copy_bytes + fills * b, st.host_bytes + host * b, st.missed_bytes + (fills + host) * b, st.ahead_bytes + promos * b);
            let ms = inner.mail.as_ref().map_or(0.0, |m| m.host_ms(ly.index));
            st.host_ms += ms;
            // (the host's time for the GPU's requests prices its next misses as a call of the host's own does)
            if host > 0 && ms > 0.0 && !inner.frozen {
                inner.costs.computed(host as usize, host_pairs as usize, ms * 1e3);
            }
        }
        if let Some(m) = inner.mail.as_mut() {
            m.set_params(crate::devplan::Params::of(&inner.offload, &inner.costs))?;
        }
        let step: Vec<LayerStats> = inner
            .layers
            .iter_mut()
            .enumerate()
            .map(|(l, ly)| {
                let mut st = ly.as_mut().map(|ly| std::mem::take(&mut ly.stats)).unwrap_or_default();
                st.copy_ms += ms.get(l).copied().unwrap_or(0.0);
                st
            })
            .collect();
        let add = |to: &mut Vec<LayerStats>| {
            if to.len() < step.len() {
                to.resize(step.len(), LayerStats::default());
            }
            to.iter_mut().zip(&step).for_each(|(a, b)| a.add(b));
        };
        add(&mut inner.totals);
        add(&mut TOTALS.lock().unwrap_or_else(|e| e.into_inner()));
        if step.iter().any(|l| l.calls > 0) {
            inner.last = step.clone();
        }
        Ok(step)
    }

    /// Each layer's stats of the last step ended (`end_step`).
    pub fn last_step(&self) -> Vec<LayerStats> {
        self.inner.borrow().last.clone()
    }

    /// Each layer's stats of every step ended since the store was made (or made `cold`).
    pub fn totals(&self) -> Vec<LayerStats> {
        self.inner.borrow().totals.clone()
    }

    /// The bytes of an expert of the layers added last (the codes and scales of all its matrices).
    pub fn expert_bytes(&self) -> u64 {
        self.inner.borrow().expert_bytes
    }

    /// What the cache has done so far (the copy stream waited for, so that every copy is counted).
    pub fn stats(&self) -> Result<Stats> {
        let mut guard = self.inner.borrow_mut();
        let inner = &mut *guard;
        if let Some(c) = inner.cuda.as_mut() {
            c.count(&mut inner.stats, &mut inner.costs, inner.expert_bytes, true)?;
        }
        Ok(inner.stats)
    }

    /// The times the planner weighs now (what it was told to start with, and what the run has moved it to).
    pub fn costs(&self) -> Costs {
        self.inner.borrow().costs
    }

    /// The counts started again (a measurement's own window).
    pub fn reset_stats(&self) -> Result<()> {
        self.stats()?;
        self.inner.borrow_mut().stats = Stats::default();
        Ok(())
    }

    /// The counts in a line.
    pub fn report(&self) -> Result<String> {
        let st = self.stats()?;
        let (slots, experts) = self.inner.borrow().layers.iter().flatten().map(|l| (l.cache.slots(), l.cache.experts())).max().unwrap_or((0, 0));
        let mut line = format!(
            "expert cache: {slots} of {experts} experts a layer in the GPU's memory; {} expert reads, {} found there ({:.1}%), {} copied in and {} for a predictor; {:.3} GB in {:.1} ms of copying ({:.1} GB/s); {} extra waves of a call",
            st.reads,
            st.hits,
            100.0 * st.hit_rate(),
            st.copied,
            st.hinted,
            st.bytes as f64 / 1e9,
            st.copy_ms,
            st.gbs(),
            st.waves
        );
        if st.cpu > 0 || self.inner.borrow().offload.mode != Mode::Copy {
            let (mode, threads, costs) = {
                let inner = self.inner.borrow();
                (inner.offload.mode, inner.offload.threads, inner.costs)
            };
            line += &format!(
                "; mode {} ({threads} threads): {} experts computed by the host ({} pairs, {:.1} ms), {} copied in behind them, the host waited {:.1} ms for the GPU; the planner's times {:.0} us an expert copied, {:.0} us a pair computed",
                mode.name(),
                st.cpu,
                st.cpu_pairs,
                st.cpu_ms,
                st.promoted,
                st.wait_ms,
                costs.copy,
                costs.pair
            );
        }
        // the disk tier, where the computer's memory holds only some of the experts: the slots a layer has, and what was read from the packed directory
        let tier = self.inner.borrow().layers.iter().flatten().filter_map(|l| l.tier.as_ref().map(|t| (t.ram.slots(), t.ram.experts()))).max();
        if let Some((held, of)) = tier {
            line += &format!(
                "; experts on the disk: up to {held} of {of} experts a group of a layer in the computer's memory, {} found there when wanted, {} read from the packed directory ({:.3} GB in {:.1} ms of the host's waiting, {:.2} GB/s)",
                st.ram_hits,
                st.disk_reads,
                st.disk_bytes as f64 / 1e9,
                st.disk_ms,
                st.disk_gbs()
            );
        }
        Ok(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache made fewer places keeps the experts of the first ones in their order of use; made more again, the places added are free.
    /// A GPU's table read back (src/dynamic/tp_cache.rs's device path): each expert's place from it, a repeat and an id past the experts left free, the
    /// free places taken first by the next request, and the experts placed hit.
    #[test]
    fn places_set_from_a_table_are_the_cache_s() {
        let mut c = LayerCache::new(8, 4);
        c.request(&[0, 1, 2, 3]).unwrap();
        c.set_places(&[5, -1, 5, 99]);
        assert_eq!((c.slot_of(5), c.slot_of(0), c.resident()), (Some(0), None, 1));
        let got = c.request(&[5, 6, 7]).unwrap();
        assert_eq!(got.hits, 1);
        assert!(got.loads.iter().all(|&(_, p)| p != 0) && got.loads.len() == 2, "{:?}", got.loads);
        assert_eq!(c.resident(), 3);
    }

    #[test]
    fn a_cache_resized_keeps_its_first_places() {
        let mut c = LayerCache::new(16, 4);
        c.request(&[1, 2, 3, 4]).unwrap();
        c.request(&[1]).unwrap();
        c.resize(2);
        assert_eq!((c.slots(), c.slot_of(1), c.slot_of(2), c.slot_of(3), c.slot_of(4)), (2, Some(0), Some(1), None, None));
        // the least recently used of those kept leaves for a miss
        assert_eq!(c.request(&[5]).unwrap().loads, vec![(5, 1)]);
        c.resize(4);
        assert_eq!(c.request(&[6, 7]).unwrap().loads, vec![(6, 2), (7, 3)]);
        assert_eq!((c.slot_of(1), c.slot_of(5), c.resident()), (Some(0), Some(1), 4));
    }

    /// The tail skip keeps an expert some row ranks above the tail, and drops one every row holding it ranks in it.
    #[test]
    fn the_tail_skip() {
        // two rows of 4 pairs: expert 7 is row 0's smallest and row 1's largest, expert 9 the smallest of row 1 alone, expert 5 row 0's third
        let ids = [1i64, 2, 5, 7, 7, 3, 4, 9];
        let wts = [0.4f32, 0.3, 0.2, 0.1, 0.5, 0.25, 0.15, 0.1];
        let mut mine = vec![2, 3, 4, 7];
        tail_skip(&mut mine, &ids, &wts, 4, 2);
        assert_eq!(mine, vec![3, 4], "7 kept for row 1's rank, 5 and 9 dropped");
        let mut all = vec![2, 3, 4, 7];
        tail_skip(&mut all, &ids, &wts, 4, 0);
        assert_eq!(all, vec![2, 3, 4, 7]);
        // row 1 against a deeper tail (3 of 4): its expert 7 (rank 0) stays, 3 (rank 1) and 9 go; row 0's tail of 1: 5 (rank 2) stays
        let mut rows = vec![2, 4, 5, 7];
        tail_skip_rows(&mut rows, &ids, &wts, 4, 1, 3);
        assert_eq!(rows, vec![2, 4]);
    }
    use flate2::read::DeflateDecoder;
    use std::io::Read;

    /// A call's experts in places: free places go first and in order; then a miss takes the place of the least recently used expert, and a hit makes its expert the most recent.
    #[test]
    fn a_miss_takes_the_least_recently_used_place() {
        let mut c = LayerCache::new(8, 3);
        let g = c.request(&[0, 1, 2]).unwrap();
        assert_eq!((g.hits, g.loads.len()), (0, 3));
        assert_eq!(c.resident(), 3);
        // 0 is the least recently used: 3 takes its place
        let place0 = c.slot_of(0).unwrap();
        let g = c.request(&[3]).unwrap();
        assert_eq!((g.hits, g.loads), (0, vec![(3, place0 as u32)]));
        assert_eq!((c.slot_of(0), c.slot_of(3)), (None, Some(place0)));
        // 1 is touched, so 2 is the next to go
        let place2 = c.slot_of(2).unwrap();
        let g = c.request(&[1, 4]).unwrap();
        assert_eq!((g.hits, g.loads), (1, vec![(4, place2 as u32)]));
        assert_eq!((c.slot_of(2), c.resident()), (None, 3));
    }

    /// A call never takes the place of an expert it needs, even the least recently used: a plain LRU would evict 0 for 3 and miss it a moment later.
    #[test]
    fn a_call_keeps_the_experts_it_needs() {
        let mut c = LayerCache::new(8, 3);
        c.request(&[0, 1, 2]).unwrap();
        let place1 = c.slot_of(1).unwrap();
        let g = c.request(&[3, 0]).unwrap();
        assert_eq!((g.hits, g.loads), (1, vec![(3, place1 as u32)]));
        assert!(c.slot_of(0).is_some() && c.slot_of(1).is_none());
        // all the places needed: nothing else can go
        let g = c.request(&[5, 6, 3]).unwrap();
        assert_eq!(g.hits, 1);
        assert_eq!(g.loads.len(), 2);
        assert!(c.slot_of(3).is_some() && c.slot_of(5).is_some() && c.slot_of(6).is_some() && c.slot_of(0).is_none());
    }

    #[test]
    fn requests_that_cannot_be_served_are_refused() {
        let mut c = LayerCache::new(8, 3);
        assert!(c.request(&[0, 1, 2, 3]).unwrap_err().msg.contains("4 experts asked of a cache of 3"));
        assert!(c.request(&[1, 1]).unwrap_err().msg.contains("twice"));
        assert!(c.request(&[8]).unwrap_err().msg.contains("expert 8 of 8"));
        assert_eq!(c.resident(), 0, "a refused request left nothing half done in the places it did not take");
    }

    /// A hint places experts as a request does, the last the most recent, repeats and non-experts left out, and a request after it finds them; of more than the places, the last are kept.
    #[test]
    fn a_hint_places_experts_for_the_next_call() {
        let mut c = LayerCache::new(16, 4);
        let g = c.hint(&[5, 6, 5, 99, 7]).unwrap();
        assert_eq!((g.hits, g.loads.iter().map(|l| l.0).collect::<Vec<_>>()), (0, vec![5, 6, 7]));
        let g = c.request(&[7, 5, 9]).unwrap();
        assert_eq!((g.hits, g.loads.len()), (2, 1));
        // more than the places: the first are left out
        let mut d = LayerCache::new(16, 3);
        d.hint(&[1, 2, 3, 4, 5]).unwrap();
        assert!(d.slot_of(1).is_none() && d.slot_of(2).is_none() && d.slot_of(3).is_some() && d.slot_of(5).is_some());
        // seeding: the profile's last expert is the most recent, so a full cache gives up the first
        let mut e = LayerCache::new(16, 3);
        e.hint(&[10, 11, 12]).unwrap();
        e.request(&[13]).unwrap();
        assert!(e.slot_of(10).is_none() && e.slot_of(11).is_some() && e.slot_of(12).is_some());
    }

    /// A call of more experts than places: waves of as many as fit, the ones already there first, and over the waves each expert the call lacks is copied once and none that is there is.
    #[test]
    fn a_big_call_is_waves_and_each_missing_expert_is_copied_once() {
        let mut c = LayerCache::new(32, 4);
        c.request(&[10, 11, 12]).unwrap();
        assert_eq!(c.waves(&[1, 2, 3]), vec![vec![1, 2, 3]], "a call that fits is one wave, as given");
        let needed: Vec<u16> = vec![1, 12, 2, 3, 11, 4, 5, 6, 7];
        let waves = c.waves(&needed);
        assert_eq!(waves, vec![vec![12, 11, 1, 2], vec![3, 4, 5, 6], vec![7]]);
        let (mut hits, mut loaded) = (0, Vec::new());
        for w in &waves {
            let g = c.request(w).unwrap();
            hits += g.hits;
            loaded.extend(g.loads.into_iter().map(|l| l.0));
            assert!(w.iter().all(|&e| c.slot_of(e as usize).is_some()), "a wave's experts are in places while it runs");
        }
        loaded.sort_unstable();
        assert_eq!((hits, loaded), (2, vec![1, 2, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn choices_and_profiles() {
        assert_eq!(distinct(&[5, 3, 5, -1, 3, 9, 700], 16), vec![5, 3, 9]);
        // the most used first, the lower expert first among equals
        assert_eq!(profile(&[3, 9, 9, 0, 3, 1], 4), vec![1, 2, 0, 4]);
        let mut out = [0i64; 4];
        fill_way(&[2, 7, -1, 40], &[-1, -1, 5, -1, -1, -1, -1, 0], &mut out);
        assert_eq!(out, [5, 0, -1, -1]);
    }

    /// The places and memory a cache is sized to: Nemotron 3 Super's experts at the swift level are 3,784,704 bytes (two matrices of 2,688 steps of 640 bytes and 172,032 of scales apiece) in 40 layers of 512.
    #[test]
    fn the_budget_is_places_a_layer() {
        let one = 3_784_704u64;
        assert_eq!(2 * (2688 * 640 + 2688 / 2 * 128), one);
        let gib = 1u64 << 30;
        assert_eq!(places_for(15 * gib, 40, one, 512, 44), Some(106));
        assert_eq!(places_for(23 * gib, 40, one, 512, 44), Some(163));
        assert_eq!(places_for(500 * gib, 40, one, 512, 44), Some(512), "no more than all of them");
        assert_eq!(places_for(3 * gib, 40, one, 512, 44), None, "21 places are fewer than a step's choices and as many again");
        assert_eq!(places_for(3 * gib, 40, one, 512, 22), None, "and fewer than a step's 22 choices");
        assert_eq!(places_for(4 * gib, 40, one, 512, 22), Some(28));
        let p = CachePlan { layers: 40, experts: 512, slots: 106, expert_bytes: one, in_place: false, ram: 40 * 512 * one };
        assert_eq!((p.gpu_bytes(), p.host_bytes()), (40 * 106 * one, 40 * 512 * one));
        // the host's half is the first matrices again: half the experts' bytes, so the computer needs 1.5 times the experts' and a margin (wren: 59.1 GiB of experts, 29.5 of the half)
        assert_eq!(p.host_half_bytes(), p.host_bytes() / 2);
        let wren = CachePlan { layers: 40, experts: 512, slots: 52, expert_bytes: 3_096_576, in_place: false, ram: 40 * 512 * 3_096_576 };
        let margin = 6 * gib;
        assert!((wren.host_half_bytes() as f64 / gib as f64 - 29.53).abs() < 0.01);
        assert!(!wren.host_half_fits(Some(88 * gib), margin) && wren.host_half_fits(Some(95 * gib), margin) && wren.host_half_fits(None, margin));
        // the Hummingbird level's host half takes nothing besides the experts: it fits whatever the memory free is when asked again
        let hb = CachePlan { in_place: true, ram: 26 * gib, ..wren };
        assert!(hb.host_half_bytes() == 0 && hb.host_half_fits(Some(20 * gib), margin) && hb.host_half_fits(None, margin));
        assert!(places_for(1 << 40, 40, 0, 512, 1).is_none(), "an expert of no bytes is no plan");
    }

    /// A layer is refused what the cache cannot hold before anything is allocated: no places, more places than experts, and a matrix whose bytes are not every expert's end to end.
    #[test]
    fn a_layer_is_refused_what_it_cannot_hold() {
        let store = ExpertStore::new();
        let bytes = vec![0u8; 8 * 10];
        let pack = |n: usize| PackIn { bytes: [&bytes[..n], &bytes[..8]], stride: [10, 1], dev: [0, 0] };
        for (slots, packs, why) in [
            (0, vec![], "a cache of 0 places for 8 experts"),
            (9, vec![], "a cache of 9 places for 8 experts"),
            (4, vec![pack(79)], "a matrix of 79 + 8 bytes for 8 experts of 10 + 1"),
        ] {
            let e = store.add_layer(0, 8, slots, &packs, None).unwrap_err();
            assert!(e.msg.contains(why), "{e}");
        }
        assert!(store.hint(0, &[1]).unwrap_err().msg.contains("no layers"));
        assert!(store.report().unwrap().starts_with("expert cache: 0 of 0 experts a layer"));
    }

    #[test]
    fn the_counts_are_rates() {
        let s = Stats { reads: 200, hits: 150, misses: 50, bytes: 5_000_000_000, copy_ms: 200.0, ..Stats::default() };
        assert_eq!((s.hit_rate(), s.gbs()), (0.75, 25.0));
        assert_eq!((Stats::default().hit_rate(), Stats::default().gbs()), (0.0, 0.0));
    }

    // ---- what is done with a miss ----------------------------------------------------------------------------------------------------------------------------------------------------------

    fn miss(expert: u16, pairs: usize, hot: bool) -> Miss {
        Miss { expert, pairs, hot }
    }

    fn copies(flags: &[bool]) -> usize {
        flags.iter().filter(|&&c| c).count()
    }

    /// The modes are named as the flag says, and a name that is none is no mode.
    /// The command line's offload settings: the defaults, the threads asked for, and (a development build) the mode, the pinning, the window and the places a call may start copying; a sweep's variants name a mode, threads and pinning.
    #[test]
    fn the_command_line_sets_the_offload() {
        let args = |v: &[&str]| crate::args::Args::from(v.iter().map(|s| s.to_string()));
        assert_eq!(offload_args(&args(&["generate", "dir"])).unwrap(), Offload::default());
        assert_eq!(offload_args(&args(&["run", "m", "--cpu-threads", "6"])).unwrap().threads, 6);
        assert_eq!(offload_args(&args(&["run", "m", "--cpu-threads", "0"])).unwrap().threads, 1);
        assert!(offload_args(&args(&["run", "m", "--cpu-threads", "many"])).is_err());
        #[cfg(feature = "dev")]
        {
            let o = offload_args(&args(&["generate", "d", "--offload-mode", "split", "--cpu-pin", "--offload-window", "8", "--offload-promote", "1"])).unwrap();
            assert_eq!((o.mode, o.pin, o.window, o.promote), (Mode::Split, true, 8, 1));
            assert!(offload_args(&args(&["generate", "d", "--offload-mode", "gpu"])).is_err());
            let base = Offload { threads: 20, ..Offload::default() };
            assert_eq!(parse_variant("cpu", &base), Some(Offload { mode: Mode::Cpu, ..base }));
            assert_eq!(parse_variant("split:8:pin", &base), Some(Offload { mode: Mode::Split, threads: 8, pin: true, ..base }));
            assert!(
                parse_variant("cpu:0", &base).is_none()
                    && parse_variant("fast", &base).is_none()
                    && parse_variant("cpu:x", &base).is_none()
                    && parse_variant("", &base).is_none()
            );
        }
    }

    #[test]
    fn the_modes_have_their_names() {
        for (name, mode) in [("copy", Mode::Copy), ("cpu", Mode::Cpu), ("split", Mode::Split), ("auto", Mode::Auto)] {
            assert_eq!((Mode::parse(name), mode.name()), (Some(mode), name));
        }
        assert!(Mode::parse("gpu").is_none() && Mode::parse("").is_none() && Mode::parse("Auto").is_none());
        assert_eq!(Offload::default().mode, Mode::Auto);
        assert!(Offload::default().threads >= 1);
    }

    /// `Copy` copies every miss and `Cpu` none; with the times of a PCIe 4 card (an expert copied in 250 microseconds, the host's call 100 and 50 a pair: the records') seven misses of a decode step are the host's but one
    /// (`Auto` and `Split` agree: 400 microseconds against 450 for the host alone), with a PCIe 5 card's (62) four of the seven are copied and the host's other three finish with them (250 against 450); a copy that costs
    /// nothing copies all that the places leave room for, one that costs the earth none; and no room, no copy.
    #[test]
    fn a_miss_goes_where_the_times_expected_say() {
        let seven: Vec<Miss> = (0..7).map(|e| miss(e, 1, false)).collect();
        let (pcie4, pcie5) = (Costs::default(), Costs { copy: 62.0, ..Costs::default() });
        assert_eq!((copies(&plan(Mode::Copy, &seven, 0, &pcie4)), copies(&plan(Mode::Cpu, &seven, 99, &pcie4))), (7, 0));
        for mode in [Mode::Auto, Mode::Split] {
            assert_eq!(copies(&plan(mode, &seven, 99, &pcie4)), 1, "{mode:?} on PCIe 4");
            assert_eq!(copies(&plan(mode, &seven, 99, &pcie5)), 4, "{mode:?} on PCIe 5");
            assert_eq!(copies(&plan(mode, &seven, 99, &Costs { copy: 1.0, ..pcie4 })), 7, "{mode:?}: copies for nothing");
            assert_eq!(copies(&plan(mode, &seven, 3, &Costs { copy: 1.0, ..pcie4 })), 3, "{mode:?}: as many as there is room for");
            assert_eq!(copies(&plan(mode, &seven, 99, &Costs { copy: 1e6, ..pcie4 })), 0, "{mode:?}: copies for the earth");
            assert_eq!(copies(&plan(mode, &seven, 0, &pcie5)), 0, "{mode:?}: no room");
            assert!(plan(mode, &[], 5, &pcie4).is_empty());
        }
        // one miss: the quicker of the host's call and the copy
        assert_eq!(plan(Mode::Auto, &[miss(9, 1, false)], 5, &pcie4), [false]);
        assert_eq!(plan(Mode::Auto, &[miss(9, 1, false)], 5, &Costs { copy: 100.0, ..pcie4 }), [true]);
    }

    /// The copy stream is one queue: copies a call asks for start after the copying already queued (the promotions of the calls before) and the call's products wait for them. On the real 16 GB card (benchmarks/gpu/nh-hb-cpu-4090-2026-10-10:
    /// 39 microseconds a pair computed by 26 threads, 160 an expert of 2 MB copied at 13 GB/s) a decode step's misses of a pair each are the host's while the stream is behind (a backlog of 10 ms, 1 ms or 0.5 ms: the
    /// copies would come after it, and the host is done in 480 microseconds), with an idle stream two of the ten are copied (320 microseconds against the host's other eight in 404), and the heavy misses of a prefill
    /// (20 pairs and more each) are copied while the backlog is shorter than the host's work they save; `Split` ignores the backlog.
    #[test]
    fn a_copy_waits_for_the_backlog_in_front_of_it() {
        let idle = Costs { copy: 160.0, fixed: 100.0, pair: 38.0, row: 38.0, backlog: 0.0 };
        let behind = |us: f64| Costs { backlog: us, ..idle };
        let decode: Vec<Miss> = (0..10).map(|e| miss(e, 1, e % 3 == 0)).collect();
        for us in [10_000.0, 1_000.0, 500.0] {
            assert_eq!(copies(&plan(Mode::Auto, &decode, 99, &behind(us))), 0, "a backlog of {us} microseconds");
        }
        assert_eq!(copies(&plan(Mode::Auto, &decode, 99, &idle)), 2);
        // (the copies are the hot ones: the first two of the misses of a pair each)
        assert_eq!(plan(Mode::Auto, &decode, 99, &idle)[..4], [true, false, false, true]);
        let prefill = [miss(1, 30, false), miss(2, 24, false), miss(3, 22, false), miss(4, 1, false), miss(5, 1, false), miss(6, 1, false)];
        assert_eq!(plan(Mode::Auto, &prefill, 99, &idle), [true, true, true, false, false, false]);
        assert_eq!(plan(Mode::Auto, &prefill, 99, &behind(1_000.0)), [true, true, false, false, false, false]);
        assert_eq!(copies(&plan(Mode::Auto, &prefill, 99, &behind(10_000.0))), 0, "a backlog longer than the host's whole call");
        // the groups are planned alike, and split and the modes that decide nothing do not look at the stream
        assert_eq!(plan_groups(Mode::Auto, &[decode.clone(), vec![]], &[99, 9], &[1.0, 0.7], &behind(10_000.0)), vec![vec![false; 10], vec![]]);
        assert_eq!(plan(Mode::Split, &decode, 99, &behind(10_000.0)), plan(Mode::Split, &decode, 99, &idle));
        assert_eq!(copies(&plan(Mode::Copy, &decode, 99, &behind(10_000.0))), 10);
        // the queue's time is by the bytes: three experts of 2 MB and one of 1 MB at 160 microseconds the 2 MB
        assert_eq!(idle.queued(7 << 20, 2 << 20), 560.0);
        assert_eq!(idle.queued(0, 0), 0.0);
    }

    /// The copies go to the misses with the most pairs (the host's time is by the pair), the ones missed lately first among equals, and the earlier first among those.
    #[test]
    fn the_copies_go_to_the_heavy_and_the_hot() {
        let c = Costs { copy: 120.0, fixed: 100.0, pair: 50.0, row: 50.0, backlog: 0.0 };
        let misses = [miss(1, 1, false), miss(2, 30, false), miss(3, 1, true), miss(4, 1, true), miss(5, 12, false), miss(6, 1, false)];
        // k = 1: 30 pairs leave 16 for the host (900 microseconds); k = 2: 12 more leave 4 (300) against 240 copying; k = 3: 3 leave 3 (250) against 360
        assert_eq!(plan(Mode::Auto, &misses, 9, &c), [false, true, false, false, true, false]);
        assert_eq!(plan(Mode::Split, &misses, 9, &c), [false, true, false, false, true, false]);
        // equal pairs: the hot first, then the earlier
        let same = [miss(1, 1, false), miss(2, 1, false), miss(3, 1, true), miss(4, 1, true), miss(5, 1, false)];
        assert_eq!(plan(Mode::Auto, &same, 2, &Costs { copy: 1.0, ..c }), [false, false, true, true, false]);
        assert_eq!(plan(Mode::Auto, &same, 3, &Costs { copy: 1.0, ..c }), [true, false, true, true, false]);
    }

    /// The misses of several groups are planned together: one group is `plan`; with two, the host's call is for the pairs of both, the copies go to the heaviest misses of whichever group has a place left, and a copy of a
    /// group whose experts are larger costs more of the call's time.
    #[test]
    fn the_misses_of_two_groups_are_planned_together() {
        let c = Costs { copy: 120.0, fixed: 100.0, pair: 50.0, row: 50.0, backlog: 0.0 };
        let one = [miss(1, 1, false), miss(2, 30, false), miss(3, 1, true), miss(4, 12, false)];
        for mode in [Mode::Copy, Mode::Cpu, Mode::Split, Mode::Auto] {
            for room in [0, 2, 9] {
                assert_eq!(plan_groups(mode, &[one.to_vec()], &[room], &[1.0], &c), vec![plan(mode, &one, room, &c)], "{mode:?} room {room}");
            }
        }
        let (x, y) = ([miss(1, 30, false), miss(2, 1, false)], [miss(7, 12, false), miss(8, 12, false), miss(9, 1, false)]);
        let both = [x.to_vec(), y.to_vec()];
        // 56 pairs: the 30 copied leave 26 for the host (1,400 microseconds), the two 12s after them 2 (200) against 288 copying; one more copy of a pair costs more than the pair
        assert_eq!(plan_groups(Mode::Auto, &both, &[9, 9], &[1.0, 0.7], &c), vec![vec![true, false], vec![true, true, false]]);
        // a group with one place left gets one copy
        let got = plan_groups(Mode::Auto, &both, &[9, 1], &[1.0, 0.7], &c);
        assert_eq!(got[1].iter().filter(|&&c| c).count(), 1);
        assert!(got[0][0] && got[1][0]);
        // a group whose copies cost 30 times as much is left to the host
        assert_eq!(plan_groups(Mode::Auto, &both, &[9, 9], &[1.0, 30.0], &c), vec![vec![true, false], vec![false, false, false]]);
        assert_eq!(plan_groups(Mode::Copy, &both, &[0, 0], &[1.0, 1.0], &c), vec![vec![true, true], vec![true, true, true]]);
        assert_eq!(plan_groups(Mode::Cpu, &both, &[9, 9], &[1.0, 1.0], &c), vec![vec![false, false], vec![false, false, false]]);
        assert_eq!(plan_groups(Mode::Split, &[vec![], vec![]], &[3, 3], &[1.0, 1.0], &c), vec![Vec::<bool>::new(), vec![]]);
    }

    /// A call of two groups: each is served from its own cache (hits keep their places, the copies take places of their own group), the pairs of the misses the host keeps are noted by group, and the places behind the
    /// call are `promote` for both together.
    #[test]
    fn a_call_of_two_groups_is_served_by_each_cache() {
        let costs = Costs::default();
        let (mut c0, mut c1) = (LayerCache::new(8, 3), LayerCache::new(6, 2));
        c0.request(&[0, 1]).unwrap();
        c1.request(&[5]).unwrap();
        let (mut l0, mut l1) = (vec![0u64; 8], vec![0u64; 6]);
        let off = Offload { mode: Mode::Cpu, window: 10, promote: 3, ..Offload::default() };
        let run = |c0: &mut LayerCache, c1: &mut LayerCache, l0: &mut Vec<u64>, l1: &mut Vec<u64>, now: u64, off: &Offload| {
            let (n0, n1) = ([0u16, 6, 7, 1], [5u16, 2, 3]);
            let mut parts = [
                Part { cache: c0, needed: &n0, pairs_of: &pairs_one(8), last: l0, now, size: 1.0 },
                Part { cache: c1, needed: &n1, pairs_of: &pairs_one(6), last: l1, now, size: 0.7 },
            ];
            serve_groups(&mut parts, off, &costs).unwrap()
        };
        let sv = run(&mut c0, &mut c1, &mut l0, &mut l1, 1, &off);
        assert_eq!((sv[0].hits, sv[0].kept.clone(), sv[0].wanted.clone()), (2, vec![6, 7], vec![0, 1]));
        assert_eq!((sv[1].hits, sv[1].kept.clone(), sv[1].wanted.clone()), (1, vec![2, 3], vec![5]));
        assert!(sv.iter().all(|s| s.promote.is_empty() && s.loads.is_empty()));
        // the same again: all four kept experts are hot, three places behind the call between them (the first group's two, then the second's one)
        let sv = run(&mut c0, &mut c1, &mut l0, &mut l1, 2, &off);
        assert_eq!((sv[0].promote.clone(), sv[1].promote.clone()), (vec![6, 7], vec![2]));
        let sv = run(&mut c0, &mut c1, &mut l0, &mut l1, 3, &Offload { promote: 1, ..off });
        assert_eq!((sv[0].promote.len(), sv[1].promote.len()), (1, 0));
        // a mode that copies: cheap copies fill each group's own places, never more than it has, and what is not copied is kept
        let (mut f0, mut f1) = (LayerCache::new(8, 3), LayerCache::new(6, 2));
        let (n0, n1, cheap) = ([0u16, 6, 7, 1], [5u16, 2, 3], Costs { copy: 1.0, ..costs });
        let mut parts = [
            Part { cache: &mut f0, needed: &n0, pairs_of: &pairs_one(8), last: &mut [0; 8], now: 1, size: 1.0 },
            Part { cache: &mut f1, needed: &n1, pairs_of: &pairs_one(6), last: &mut [0; 6], now: 1, size: 0.7 },
        ];
        let sv = serve_groups(&mut parts, &Offload { mode: Mode::Split, ..off }, &cheap).unwrap();
        assert_eq!((sv[0].loads.len(), sv[0].kept.len(), sv[1].loads.len(), sv[1].kept.len()), (3, 1, 2, 1));
        assert!(sv[0].loads.iter().all(|&(e, p)| e < 8 && p < 3) && sv[1].loads.iter().all(|&(e, p)| e < 6 && p < 2));
    }

    /// A layer's call in groups on the host alone: every pair of a group in the store has a place in its group's cache or is the host's, never both and never a place in the other group; the host's pairs are those whose
    /// expert has no place, in the pairs' order; the places of a group's experts are distinct; the pairs of a group that is not in the store are neither; in every mode, over random calls and caches that keep their places
    /// from call to call.
    #[test]
    fn a_layer_in_groups_is_placed_pair_by_pair() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as usize % n
        };
        // 40 experts, 15 of the first group (4 places) and 25 of the second (6)
        let experts = 40;
        let group_of: Vec<u8> = (0..experts).map(|e| (e % 8 < 3) as u8).collect();
        let mut local_of = vec![0u16; experts];
        let mut counts = [0usize; 2];
        for e in 0..experts {
            local_of[e] = counts[group_of[e] as usize] as u16;
            counts[group_of[e] as usize] += 1;
        }
        assert_eq!(counts, [25, 15]);
        let costs = Costs::default();
        for mode in [Mode::Copy, Mode::Cpu, Mode::Split, Mode::Auto] {
            for absent in [None, Some(1usize)] {
                let off = Offload { mode, window: 6, promote: 3, ..Offload::default() };
                let mut caches = [LayerCache::new(counts[0], 6), LayerCache::new(counts[1], 4)];
                let mut lasts = [vec![0u64; counts[0]], vec![0u64; counts[1]]];
                let (mut kept, mut copied) = (0usize, 0usize);
                for call in 1..=150u64 {
                    let (tokens, k) = (1 + next(6), 4);
                    let mut ids = Vec::new();
                    for _ in 0..tokens {
                        let mut pick = Vec::new();
                        while pick.len() < k {
                            let e = next(experts) as i64;
                            if !pick.contains(&e) {
                                pick.push(e);
                            }
                        }
                        ids.extend(pick);
                    }
                    let [c0, c1] = &mut caches;
                    let [l0, l1] = &mut lasts;
                    let mut slices = [
                        Some(Slice { cache: c0, last: l0, now: call, size: 1.0 }),
                        (absent != Some(1)).then_some(Slice { cache: c1, last: l1, now: call, size: 0.7 }),
                    ];
                    let placed = place_groups(&mut slices, &group_of, &local_of, &ids, &off, &costs).unwrap();
                    let mut mine = Vec::new();
                    for (j, &e) in ids.iter().enumerate() {
                        let (g, l) = (group_of[e as usize] as usize, local_of[e as usize] as i64);
                        assert_eq!(placed.local[g][j], if absent == Some(g) { -1 } else { l }, "{mode:?} call {call} pair {j}");
                        assert_eq!(placed.local[1 - g][j], -1);
                        if absent == Some(g) {
                            assert!(placed.ways[g].is_empty() && placed.served[g].is_none());
                            continue;
                        }
                        assert_eq!(
                            placed.ways[g][j],
                            caches[g].slot_of(l as usize).map_or(-1, |p| p as i64),
                            "{mode:?} call {call} pair {j}: its place is its expert's"
                        );
                        if placed.ways[g][j] < 0 {
                            mine.push((g, j));
                        }
                        if let Some(other) = placed.ways.get(1 - g).filter(|w| !w.is_empty()) {
                            assert_eq!(other[j], -1, "a place in the other group");
                        }
                    }
                    assert_eq!(placed.mine, mine, "{mode:?} call {call}");
                    for (g, cache) in caches.iter().enumerate() {
                        let Some(sv) = &placed.served[g] else { continue };
                        let places: Vec<usize> = placed.needed[g].iter().filter_map(|&e| cache.slot_of(e as usize)).collect();
                        assert_eq!(places.iter().collect::<std::collections::HashSet<_>>().len(), places.len());
                        assert_eq!(sv.wanted.len() + sv.kept.len(), placed.needed[g].len());
                        assert!(sv.kept.iter().all(|&e| cache.slot_of(e as usize).is_none()));
                        kept += sv.kept.len();
                        copied += sv.loads.len();
                    }
                    if mode == Mode::Cpu {
                        assert!(placed.served.iter().flatten().all(|sv| sv.loads.is_empty()), "cpu mode copies nothing");
                    }
                }
                assert!(copied > 0 || mode == Mode::Cpu, "{mode:?} copied nothing");
                assert!(kept > 0, "{mode:?}: the host kept nothing in a cache of fewer places than experts");
            }
        }
    }

    /// The times follow what is measured: a batch of copies moves the time of an expert toward its share of the batch, a call of the host's the time of a pair toward what is left of the call's time after the fixed part; empty
    /// measurements move nothing, and none goes out of its range.
    #[test]
    fn the_costs_follow_the_measurements() {
        let mut c = Costs::default();
        // 40 experts of 3 MB in 5 ms: 125 microseconds an expert
        for _ in 0..60 {
            c.copied(120_000_000, 5.0, 3_000_000);
        }
        assert!((c.copy - 125.0).abs() < 1.0, "{c:?}");
        for _ in 0..60 {
            c.computed(6, 6, 100.0 + 6.0 * 31.0);
        }
        assert!((c.pair - 31.0).abs() < 0.5, "{c:?}");
        let before = c;
        c.copied(0, 5.0, 3_000_000);
        c.copied(100, 0.0, 3_000_000);
        c.computed(0, 0, 500.0);
        assert_eq!(c, before);
        c.computed(3, 3, 1.0);
        assert!(c.pair >= 2.0 * 0.25 && c.pair < before.pair, "a call shorter than its fixed part: {c:?}");
        // calls of one row an expert and of four (8 experts, 8 or 32 pairs) on a host that reads an expert in 300 microseconds and computes a further row in 40: both times found
        let mut c = Costs::default();
        for i in 0..400 {
            let pairs = if i % 2 == 0 { 8 } else { 32 };
            c.computed(8, pairs, 100.0 + 8.0 * 300.0 + (pairs - 8) as f64 * 40.0);
        }
        assert!((c.pair - 300.0).abs() < 1.0 && (c.row - 40.0).abs() < 1.0, "{c:?}");
        assert_eq!((c.host(1), c.host(3)), (c.pair, c.pair + 2.0 * c.row));
        // more experts than pairs is no call
        let before = c;
        c.computed(4, 2, 900.0);
        assert_eq!(c, before);
    }

    /// A call whose misses do not fit the places its hits leave is copied whole in waves where the host's share of it would take longer (many rows a miss), and planned with the host where it would not; only `Auto`
    /// decides so, and a call whose misses fit is never waves.
    #[test]
    fn many_rows_of_misses_past_the_places_go_in_waves() {
        let c = Costs { copy: 480.0, fixed: 100.0, pair: 300.0, row: 250.0, backlog: 0.0 };
        // (30 misses of 3 rows each, 2 places left: the host's 28 take 22,500 microseconds against 14,400 copying all 30)
        let heavy: Vec<Miss> = (0..30).map(|e| miss(e, 3, false)).collect();
        assert!(waves_quicker(Mode::Auto, &heavy, 2, &c));
        assert!(!waves_quicker(Mode::Cpu, &heavy, 2, &c) && !waves_quicker(Mode::Split, &heavy, 2, &c));
        // (one row a miss: the host's 28 take 8,500 against 14,400)
        let light: Vec<Miss> = (0..30).map(|e| miss(e, 1, false)).collect();
        assert!(!waves_quicker(Mode::Auto, &light, 2, &c));
        // (room for all: never waves)
        assert!(!waves_quicker(Mode::Auto, &heavy, 30, &c));
        // (a long backlog makes the copies dear)
        assert!(!waves_quicker(Mode::Auto, &heavy, 2, &Costs { backlog: 50_000.0, ..c }));
    }

    /// The host's time for a miss is its expert's first row and then its other rows (`Costs::host`): where a row after the first costs the products alone, a miss of many rows is still the host's when a pair-by-pair
    /// host would have it copied, and the copies go to the misses whose rows make the host's time the longest.
    #[test]
    fn a_miss_of_many_rows_costs_the_host_its_rows_products() {
        let grouped = Costs { copy: 400.0, fixed: 100.0, pair: 300.0, row: 20.0, backlog: 0.0 };
        let by_pair = Costs { row: 300.0, ..grouped };
        let misses = [miss(1, 8, false), miss(2, 2, false), miss(3, 1, false)];
        // (by the pair: 2,400, 600 and 300 microseconds; copying the first two leaves 400 against 800 of copies)
        assert_eq!(plan(Mode::Auto, &misses, 9, &by_pair), [true, true, false]);
        // (an expert read once: 440, 320 and 300; one copy leaves the host 720 against 400 of copying, two 400 against 800)
        assert_eq!(plan(Mode::Auto, &misses, 9, &grouped), [true, false, false]);
        assert_eq!(grouped.host(8), 440.0);
    }

    fn pairs_one(experts: usize) -> Vec<usize> {
        vec![1; experts]
    }

    /// A decode step of 8 rows (6 of 64 experts a row, 16 places; uniform choices) in each mode, the step's layer statistics made as the store makes them (`LayerStats::served`, `hosted` with `host_reads`): the
    /// bytes read from the computer's memory for the misses are the distinct missed experts' bytes, each expert once whatever its rows, copied or computed by a host that takes an expert's pairs together; a host that
    /// computes pair by pair reads more where rows share an expert.
    #[test]
    fn a_step_of_many_rows_reads_each_missed_expert_once() {
        let (experts, k, rows, places, bytes) = (64usize, 6usize, 8usize, 16usize, 12_582_912u64);
        let mut x = 0x0DDB_1A5E_5BAD_5EEDu64;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as usize % n
        };
        for mode in [Mode::Copy, Mode::Cpu, Mode::Split, Mode::Auto] {
            let off = Offload { mode, ..Offload::default() };
            let (mut cache, mut last) = (LayerCache::new(experts, places), vec![0u64; experts]);
            let mut per_pair_more = 0u64;
            for step in 1..=40u64 {
                let mut ids = Vec::new();
                for _ in 0..rows {
                    let mut pick: Vec<i64> = Vec::new();
                    while pick.len() < k {
                        let e = next(experts) as i64;
                        if !pick.contains(&e) {
                            pick.push(e);
                        }
                    }
                    ids.extend(pick);
                }
                let needed = distinct(&ids, experts);
                let mut pairs_of = vec![0usize; experts];
                ids.iter().for_each(|&e| pairs_of[e as usize] += 1);
                let sv = serve(&mut cache, &needed, &pairs_of, &mut last, step, &off, &Costs::default()).unwrap();
                let mut st = LayerStats::default();
                st.called(rows, ids.len(), 0.0);
                st.served(&sv, sv.loads.len() as u64 * bytes, (sv.loads.len() + sv.kept.len()) as u64 * bytes);
                // (the host's pairs: those of the experts left with no place)
                let mine: Vec<usize> = (0..ids.len()).filter(|&j| cache.slot_of(ids[j] as usize).is_none()).collect();
                let (n, b) = host_reads(&ids, &mine, experts, true, |_| bytes);
                st.hosted(n, mine.len(), 0, b, 0.0);
                assert_eq!(st.copies + st.host, st.misses(), "{mode:?} step {step}: every miss copied or computed, never both");
                assert_eq!(st.miss_bytes(), st.misses() * bytes, "{mode:?} step {step}: each missed expert's bytes read once");
                assert_eq!(st.miss_bytes(), st.missed_bytes);
                assert_eq!((st.reads, st.rows, st.pairs), (needed.len() as u64, rows as u64, (rows * k) as u64));
                per_pair_more += host_reads(&ids, &mine, experts, false, |_| bytes).1 - b;
            }
            assert!(mode == Mode::Copy || per_pair_more > 0, "{mode:?}: no host pairs shared an expert in 40 steps");
        }
        // (/stats's field: none before a layer has run a call, then a JSON object a layer)
        TOTALS.lock().unwrap().clear();
        assert_eq!(stats_field(), "");
        TOTALS.lock().unwrap().extend([LayerStats::default(), LayerStats { calls: 2, hits: 1, ..LayerStats::default() }]);
        let f = stats_field();
        assert!(f.starts_with(",\"experts\":[{\"layer\":0,") && f.contains("{\"layer\":1,\"calls\":2,") && f.ends_with(']'), "{f}");
        TOTALS.lock().unwrap().clear();
        let one = LayerStats { calls: 1, reads: 3, hits: 1, copy_bytes: 5, host_bytes: 7, copy_ms: 0.5, ..LayerStats::default() };
        let two = LayerStats::sum([&one, &one]);
        assert_eq!((two.calls, two.misses(), two.miss_bytes(), two.copy_ms), (2, 4, 24, 1.0));
        assert!(one.json(3).starts_with("{\"layer\":3,\"calls\":1,") && one.json(3).contains("\"copy_ms\":0.500"));
    }

    /// A call's bookkeeping: hits keep their places, the copies get places that are not the call's other experts', the misses left to the host have none, every expert of the call is one or the other, and a
    /// second miss within the window is worth a place (at most `promote` a call), the first is not.
    #[test]
    fn a_call_serves_its_hits_copies_and_leaves_the_rest_to_the_host() {
        let off = Offload { mode: Mode::Cpu, window: 4, promote: 2, ..Offload::default() };
        let costs = Costs::default();
        let mut c = LayerCache::new(32, 6);
        let mut last = vec![0u64; 32];
        c.request(&[0, 1, 2]).unwrap();
        // 0 and 1 hit; 7, 8, 9 are the host's: no copies, none promoted (the first miss of each)
        let sv = serve(&mut c, &[0, 7, 1, 8, 9], &pairs_one(32), &mut last, 1, &off, &costs).unwrap();
        assert_eq!((sv.wanted.clone(), sv.loads.len(), sv.hits, sv.misses, sv.kept.clone(), sv.promote.clone()), (vec![0, 1], 0, 2, 3, vec![7, 8, 9], vec![]));
        assert!(c.slot_of(7).is_none() && c.slot_of(0).is_some() && c.slot_of(2).is_some());
        // 7, 8 and 9 again in the 3rd call (within 4 of the 1st): hot, two of them are promoted (the order the call asked in), all are noted
        let sv = serve(&mut c, &[9, 7, 8, 1], &pairs_one(32), &mut last, 3, &off, &costs).unwrap();
        assert_eq!((sv.kept.clone(), sv.promote.clone(), last[7], last[9], last[0]), (vec![9, 7, 8], vec![9, 7], 3, 3, 0));
        let got = c.hint(&sv.promote).unwrap();
        assert_eq!(got.loads.len(), 2);
        assert!(c.slot_of(9).is_some() && c.slot_of(7).is_some() && c.slot_of(8).is_none());
        // 8 was kept in the 3rd call and again in the 9th: past the window, not promoted; 9 and 7 are hits now
        let sv = serve(&mut c, &[8, 9, 7], &pairs_one(32), &mut last, 9, &off, &costs).unwrap();
        assert_eq!((sv.hits, sv.kept.clone(), sv.promote.clone()), (2, vec![8], vec![]));
        // a mode that copies puts the misses in places (never one the call needs); when there is no room for all they are the host's
        let copy = Offload { mode: Mode::Split, ..off };
        let cheap = Costs { copy: 1.0, ..costs };
        let mut c = LayerCache::new(32, 4);
        let sv = serve(&mut c, &[1, 2, 3], &pairs_one(32), &mut vec![0; 32], 1, &copy, &cheap).unwrap();
        assert_eq!((sv.wanted.clone(), sv.loads.len(), sv.kept.len()), (vec![1, 2, 3], 3, 0));
        let sv = serve(&mut c, &[1, 4, 5, 6, 2], &pairs_one(32), &mut vec![0; 32], 2, &copy, &cheap).unwrap();
        assert_eq!((sv.hits, sv.loads.len(), sv.kept.len(), sv.wanted.len()), (2, 2, 1, 4));
        assert!(sv.wanted.iter().all(|&e| c.slot_of(e as usize).is_some()) && sv.kept.iter().all(|&e| c.slot_of(e as usize).is_none()));
        // the places of the copies are not the places of the hits
        let places: Vec<usize> = sv.wanted.iter().map(|&e| c.slot_of(e as usize).unwrap()).collect();
        assert!(places.iter().collect::<std::collections::HashSet<_>>().len() == places.len());
    }

    /// Random calls through a small cache in every mode: what a call needs is in a place or left to the host, never both; the places of what it keeps are distinct; the copies are the loads; and in the mode that copies
    /// all (`Copy`, with room) a call is served as `LayerCache::request` serves it.
    #[test]
    fn calls_in_every_mode_keep_the_cache_whole() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for mode in [Mode::Copy, Mode::Cpu, Mode::Split, Mode::Auto] {
            for costs in [Costs::default(), Costs { copy: 40.0, ..Costs::default() }] {
                let off = Offload { mode, window: 8, promote: 3, ..Offload::default() };
                let (mut c, mut last) = (LayerCache::new(40, 9), vec![0u64; 40]);
                for call in 1..=300u64 {
                    let k = 1 + next(8);
                    let mut needed = Vec::new();
                    while needed.len() < k {
                        let e = next(40) as u16;
                        if !needed.contains(&e) {
                            needed.push(e);
                        }
                    }
                    let pairs: Vec<usize> = (0..40).map(|_| 1 + next(3)).collect();
                    let resident: Vec<bool> = needed.iter().map(|&e| c.slot_of(e as usize).is_some()).collect();
                    let sv = serve(&mut c, &needed, &pairs, &mut last, call, &off, &costs).unwrap();
                    assert_eq!(sv.hits, resident.iter().filter(|&&r| r).count());
                    assert_eq!(sv.hits + sv.misses, needed.len());
                    assert_eq!(sv.loads.len() + sv.kept.len(), sv.misses, "{mode:?} call {call}");
                    for &e in &needed {
                        let (placed, kept) = (c.slot_of(e as usize).is_some(), sv.kept.contains(&e));
                        assert!(placed != kept, "{mode:?} call {call}: expert {e} placed {placed}, kept {kept}");
                        assert_eq!(sv.wanted.contains(&e), placed);
                    }
                    let places: std::collections::HashSet<usize> = sv.wanted.iter().map(|&e| c.slot_of(e as usize).unwrap()).collect();
                    assert_eq!(places.len(), sv.wanted.len());
                    assert!(sv.promote.len() <= off.promote && sv.promote.iter().all(|e| sv.kept.contains(e)));
                    if mode == Mode::Copy && needed.len() <= 9 {
                        assert!(sv.kept.is_empty() && sv.wanted == needed);
                    }
                    if mode == Mode::Cpu {
                        assert!(sv.loads.is_empty());
                    }
                    c.hint(&sv.promote).unwrap();
                }
            }
        }
    }

    #[test]
    fn host_memory_is_what_meminfo_says_available() {
        let text = "MemTotal:       131604356 kB\nMemFree:         1000000 kB\nMemAvailable:   104857600 kB\nBuffers:            1000 kB\n";
        assert_eq!(parse_meminfo(text), Some(100 << 30));
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None);
        // a container of 100 GiB that holds 99 of them, 60 of which are file cache: it has 61 free (version 2's names, then version 1's)
        let gib = 1u64 << 30;
        let v2 = format!("anon {}\nfile {}\ninactive_file {}\nactive_file {}\nslab 5\n", 30 * gib, 60 * gib, 40 * gib, 20 * gib);
        assert_eq!(cgroup_free(100 * gib, 99 * gib, &v2), 61 * gib);
        let v1 = format!(
            "cache {}\ninactive_file {}\nactive_file {}\ntotal_inactive_file {}\ntotal_active_file {}\n",
            60 * gib,
            40 * gib,
            20 * gib,
            40 * gib,
            20 * gib
        );
        assert_eq!(cgroup_free(100 * gib, 99 * gib, &v1), 61 * gib, "version 1 lists the group's and the total: the cache is counted once");
        assert_eq!(cgroup_free(100 * gib, 99 * gib, ""), gib, "no statistics: the group's own count");
        assert_eq!(cgroup_free(10 * gib, 99 * gib, &v2), 0, "no more than the limit");
        if cfg!(target_os = "linux") {
            assert!(host_memory().is_some_and(|b| b > 0));
        }
    }

    // ---- the cache against the record: the routing traces of 20 decode generations (benchmarks/moe-skew-2026-10-09/nemotron) ----------------------------------------------------------------------

    const RECORD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../benchmarks/moe-skew-2026-10-09/nemotron");

    /// The arrays of an .npz that numpy's savez_compressed wrote (entries deflated, their sizes in a zip64 field of the local header): name -> (shape, the elements' bytes).
    fn npz(path: &str) -> std::collections::HashMap<String, (Vec<usize>, Vec<u8>)> {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (u16_at, u32_at, u64_at) = (
            |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize,
            |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as u64,
            |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap()),
        );
        let (mut out, mut at) = (std::collections::HashMap::new(), 0usize);
        while b.len() >= at + 30 && b[at..at + 4] == *b"PK\x03\x04" {
            let (method, mut csize, name_len, extra_len) = (u16_at(at + 8), u32_at(at + 18), u16_at(at + 26), u16_at(at + 28));
            let name = String::from_utf8(b[at + 30..at + 30 + name_len].to_vec()).unwrap();
            let mut x = at + 30 + name_len;
            while x < at + 30 + name_len + extra_len {
                let (id, len) = (u16_at(x), u16_at(x + 2));
                if id == 1 && csize == 0xffff_ffff {
                    csize = u64_at(x + 4 + 8);
                }
                x += 4 + len;
            }
            let start = at + 30 + name_len + extra_len;
            let raw = &b[start..start + csize as usize];
            let mut data = Vec::new();
            if method == 8 {
                DeflateDecoder::new(raw).read_to_end(&mut data).unwrap();
            } else {
                data.extend_from_slice(raw);
            }
            at = start + csize as usize;
            // the .npy inside: version 1.0, a dictionary of descr, fortran_order and shape
            assert!(data.starts_with(b"\x93NUMPY\x01\x00"), "{name}: not a version 1.0 .npy");
            let hl = u16::from_le_bytes([data[8], data[9]]) as usize;
            let head = std::str::from_utf8(&data[10..10 + hl]).unwrap();
            let shape: Vec<usize> =
                head.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter_map(|v| v.trim().parse().ok()).collect();
            out.insert(name.trim_end_matches(".npy").to_string(), (shape, data[10 + hl..].to_vec()));
        }
        out
    }

    /// (steps, layers, choices, ids [steps, layers, choices]) of the record's decode traces.
    fn trace() -> &'static (usize, usize, usize, Vec<u16>) {
        static T: std::sync::OnceLock<(usize, usize, usize, Vec<u16>)> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let z = npz(&format!("{RECORD}/decode.npz"));
            let (shape, bytes) = &z["ids"];
            assert_eq!(shape, &[2540, 40, 22]);
            (shape[0], shape[1], shape[2], bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        })
    }

    /// The share of routed (token, expert) pairs found in a place, replaying a layer at a time as the record's skew job does: `per_access` asks for the experts of a step one at a time, in the router's order (the
    /// record's replay: a miss may take the place of an expert the same step asks for next); else a step's experts are one request, as a call of the engine's (those it needs are kept). `seed`: the profile each
    /// layer's cache starts from, in the order the record puts it there (ascending expert).
    fn replay(cap: usize, per_access: bool, seed: Option<&[Vec<u16>]>) -> f64 {
        let &(steps, layers, k, ref ids) = trace();
        let mut hits = 0usize;
        for l in 0..layers {
            let mut c = LayerCache::new(512, cap);
            if let Some(s) = seed {
                c.hint(&s[l]).unwrap();
            }
            for t in 0..steps {
                let row = &ids[(t * layers + l) * k..][..k];
                if per_access {
                    hits += row.iter().map(|&e| c.request(&[e]).unwrap().hits).sum::<usize>();
                } else {
                    hits += c.request(row).unwrap().hits;
                }
            }
        }
        hits as f64 / (steps * layers * k) as f64
    }

    /// The record's capacities: 17%, 32% and 46% of 512 experts, rounded as its job does (87, 164, 236), with its LRU hit rates cold and seeded with the profile, in percent.
    const CAPS: [(usize, f64, f64); 3] = [(87, 60.7, 60.7), (164, 74.2, 74.2), (236, 82.4, 82.6)];

    /// The cache over the measured traces (20 generations, 2,540 steps, 40 layers) finds the share of reads the record's LRU does at 17%, 32% and 46% (60.7, 74.2, 82.4 percent): replayed an access at a time as the
    /// record's job does, to the record's one decimal; and a call's experts asked for together, as the engine asks them (a step's experts are not taken by its own misses), a little above it: 1.0, 0.5 and 0.3 points.
    #[test]
    fn the_cache_reproduces_the_records_lru_hit_rates() {
        for (cap, lru, _) in CAPS {
            let record = 100.0 * replay(cap, true, None);
            assert!((record - lru).abs() <= 0.05, "{cap} places: {record:.3}, the record's {lru}");
            let engine = 100.0 * replay(cap, false, None);
            eprintln!("{cap} places: the record's replay {record:.3}, a step asked for together {engine:.3}");
            assert!(engine >= record && engine - record < 1.1, "{cap} places: {engine:.3} against the record's replay {record:.3}");
        }
    }

    /// The cache served as the store serves a call (`serve`, then the promotions a call starts), over the record's traces: (the share of reads found in a place, copies before the products, misses left to the host, and
    /// promotions) a step over the 40 layers. A step's experts are all one pair each, as a decode step's are.
    fn replay_served(cap: usize, off: &Offload, costs: &Costs) -> (f64, f64, f64, f64) {
        let &(steps, layers, k, ref ids) = trace();
        let (mut reads, mut hits, mut copied, mut kept, mut promoted) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for l in 0..layers {
            let (mut c, mut last) = (LayerCache::new(512, cap), vec![0u64; 512]);
            for t in 0..steps {
                let needed: Vec<u16> = ids[(t * layers + l) * k..][..k].to_vec();
                let sv = serve(&mut c, &needed, &vec![1; 512], &mut last, t as u64 + 1, off, costs).unwrap();
                promoted += c.hint(&sv.promote).unwrap().loads.len();
                (reads, hits, copied, kept) = (reads + k, hits + sv.hits, copied + sv.loads.len(), kept + sv.kept.len());
            }
        }
        (hits as f64 / reads as f64, copied as f64 / steps as f64, kept as f64 / steps as f64, promoted as f64 / steps as f64)
    }

    /// Serving a call as the store does, over the record's decode traces: with every miss copied it is the record's LRU (the same share found in places, to the digit); with every miss left to the host and a second
    /// miss within a few calls given a place behind the call, the share found is what the places held by those second misses find: printed for the windows and the places a call may start copying, at the places of a 16 GB,
    /// a 24 GB and a 32 GB card at wren (52, 122 and 191 of 512).
    #[test]
    fn serving_the_traces_with_the_host_and_promotions() {
        for cap in [52usize, 122, 191] {
            let all = Offload { mode: Mode::Copy, ..Offload::default() };
            let (hit, copied, kept, _) = replay_served(cap, &all, &Costs::default());
            assert!((hit - replay(cap, false, None)).abs() < 1e-12 && kept == 0.0, "{cap} places: every miss copied is the record's cache");
            eprintln!("{cap} places, every miss copied: {:.1}% found, {copied:.0} copies a step", 100.0 * hit);
            for window in [4u64, 16, 32, 128, u64::MAX / 2] {
                for promote in [1usize, 2, 4, 22] {
                    let off = Offload { mode: Mode::Cpu, window, promote, ..Offload::default() };
                    let (hit, copied, kept, promoted) = replay_served(cap, &off, &Costs::default());
                    eprintln!("{cap} places, the host and promotions (window {window}, {promote} a call): {:.1}% found, {kept:.0} misses a step for the host, {promoted:.0} promoted a step, {copied:.0} copies", 100.0 * hit);
                }
            }
            // the host alone and no promotion never finds anything
            let off = Offload { mode: Mode::Cpu, promote: 0, ..Offload::default() };
            assert_eq!(replay_served(cap, &off, &Costs::default()).0, 0.0);
        }
        // the defaults (window 16, 4 a call): the share found in places, the misses a step the host computes (22 reads a layer, 880 a step) and the experts promoted a step, at 52, 122 and 191 places
        for (cap, found, kept, promoted) in [(52usize, 0.46, 480.0, 140.0), (122, 0.63, 330.0, 105.0), (191, 0.73, 240.0, 80.0)] {
            let off = Offload { mode: Mode::Cpu, ..Offload::default() };
            let (hit, copied, left, made) = replay_served(cap, &off, &Costs::default());
            assert!(
                hit >= found && left <= kept && made <= promoted && copied == 0.0,
                "{cap} places: {hit:.3} found, {left:.0} for the host, {made:.0} promoted"
            );
        }
    }

    /// Seeded with the static profile of the six calibration domains (a layer's most used experts, the lowest expert first) the cache starts warm and ends where the record's does: 60.7, 74.2 and 82.6 percent.
    #[test]
    fn a_cache_seeded_with_a_profile_reproduces_the_records_seeded_hit_rates() {
        let mut counts = vec![vec![0u64; 512]; 40];
        for d in ["chat", "code", "math", "multi", "reason", "wiki"] {
            let (shape, bytes) = npz(&format!("{RECORD}/counts/calib_{d}.npz")).remove("counts").unwrap();
            assert_eq!(shape, [40, 512]);
            for (i, c) in bytes.chunks_exact(8).enumerate() {
                counts[i / 512][i % 512] += i64::from_le_bytes(c.try_into().unwrap()) as u64;
            }
        }
        for (cap, _, seeded) in CAPS {
            let seed: Vec<Vec<u16>> = counts
                .iter()
                .map(|c| {
                    let mut top = profile(c, cap);
                    top.sort_unstable();
                    top
                })
                .collect();
            let got = 100.0 * replay(cap, true, Some(&seed));
            eprintln!("{cap} places seeded: the record's replay {got:.3}");
            assert!((got - seeded).abs() <= 0.05, "{cap} places seeded: {got:.3}, the record's {seeded}");
        }
    }

    /// The record's 20 generations (127 decode steps each) decoded `rows` at a time (generations g to g + rows side by side, each group once), each layer's cache of `cap` places asked for a step's experts as a call
    /// of the store asks in copy mode (one request, or waves of as many as fit where there are more): the step's distinct experts, the distinct missed ones (each copied once) and the share of pairs found in a place,
    /// a layer and step on average. `first`: the experts in the order they first appear, as calls asked before `by_rows`.
    fn replay_rows(cap: usize, rows: usize, first: bool) -> (f64, f64, f64) {
        let &(_, layers, k, ref ids) = trace();
        let (gens, steps) = (20usize, 127usize);
        let (mut distinct_n, mut missed, mut found, mut pairs, mut calls) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for g0 in (0..gens).step_by(rows).filter(|g| g + rows <= gens) {
            for l in 0..layers {
                let mut c = LayerCache::new(512, cap);
                for t in 0..steps {
                    let step: Vec<i64> = (g0..g0 + rows).flat_map(|g| ids[((g * steps + t) * layers + l) * k..][..k].iter().map(|&e| e as i64)).collect();
                    let mut needed = distinct(&step, 512);
                    if !first {
                        needed = by_rows(needed, &step, 512);
                    }
                    found += step.iter().filter(|&&e| c.slot_of(e as usize).is_some()).count();
                    for wave in c.waves(&needed) {
                        missed += c.request(&wave).unwrap().loads.len();
                    }
                    (distinct_n, pairs, calls) = (distinct_n + needed.len(), pairs + step.len(), calls + 1);
                }
            }
        }
        (distinct_n as f64 / calls as f64, missed as f64 / calls as f64, found as f64 / pairs as f64)
    }

    /// The slot policy and the planner's model of it over the record's decode traces at 1 to 20 rows a step (Nemotron 3 Super, 512 experts, 22 chosen; places for 17, 32 and 46% of them):
    /// - asking for a step's experts by their rows, fewest first (`by_rows`), never finds fewer pairs in places than in the order they first appear, and more at 16 and 20 rows;
    /// - the planner's distinct missed experts a step (`plan::step_misses` with `HIT_EXPONENT`: the reads from the computer's memory a step) are within 20% of the cache's at every point; at one row its
    ///   f^0.23 is 61, 73 and 80% where the cache finds 61, 73 and 80... (printed), the misses' error the largest there, and at many rows the places' reuse between steps goes as the rows' experts fill them.
    #[test]
    fn the_slot_policy_and_the_planners_model_over_the_traces_at_many_rows() {
        let mut worst = 0f64;
        for cap in [87usize, 164, 236] {
            for rows in [1usize, 4, 8, 16, 20] {
                let (d, missed, found) = replay_rows(cap, rows, false);
                let (_, missed0, found0) = replay_rows(cap, rows, true);
                let model = 22.0 * crate::plan::step_misses(512, 22, cap, rows, crate::plan::HIT_EXPONENT);
                let e = model / missed - 1.0;
                worst = worst.max(e.abs());
                eprintln!(
                    "{cap} places, {rows:>2} rows: {d:.1} experts a step, {missed:.1} missed ({missed0:.1} in the order they first appear), {:.1}% of pairs found ({:.1}%); the planner's {model:.1} missed ({:+.1}%)",
                    100.0 * found,
                    100.0 * found0,
                    100.0 * e
                );
                assert!(found >= found0 - 1e-9, "{cap} places, {rows} rows: by rows {found} against {found0}");
                if rows >= 16 {
                    assert!(found > found0, "{cap} places, {rows} rows: by rows finds more ({found} against {found0})");
                }
                assert!(e.abs() < 0.20, "{cap} places, {rows} rows: the planner's {model:.1} missed experts a step against the cache's {missed:.1}");
            }
        }
        eprintln!("the planner's distinct missed experts a step: the largest error {:.1}%", 100.0 * worst);
        // (a uniform router's share is the places' share at any rows; at one row the model is the old one, f^s)
        for rows in [1usize, 8, 32] {
            assert!((crate::plan::step_hit(64, 6, 32, rows, 1.0) - 0.5).abs() < 1e-12);
        }
        assert_eq!(crate::plan::step_hit(512, 22, 164, 1, crate::plan::HIT_EXPONENT), (164.0f64 / 512.0).powf(crate::plan::HIT_EXPONENT));
    }

    // ---- the places hold the bytes of the experts the cache says they do ----------------------------------------------------------------------------------------------------------------------------

    /// The copies the store queues, done on byte vectors: the place of each expert in a wave holds that expert's bytes, over requests, hints and waves of every size, and every pair's place (`fill_way`) is the place
    /// of its expert. This is what makes the plan of the places the plan of the experts.
    #[test]
    fn the_places_hold_the_bytes_of_their_experts() {
        let (experts, slots) = (40usize, 7usize);
        let host: Vec<Vec<u8>> = (0..experts).map(|e| (0..16).map(|i| (e * 16 + i) as u8 ^ 0x5a).collect()).collect();
        let mut places = vec![vec![0u8; 16]; slots];
        let mut c = LayerCache::new(experts, slots);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        let copy = |places: &mut Vec<Vec<u8>>, loads: &[(u16, u32)]| loads.iter().for_each(|&(e, p)| places[p as usize] = host[e as usize].clone());
        for round in 0..400 {
            if round % 5 == 4 {
                let h: Vec<u16> = (0..next(10)).map(|_| next(experts) as u16).collect();
                let g = c.hint(&h).unwrap();
                copy(&mut places, &g.loads);
                continue;
            }
            // a call of 1 to 6 tokens of 4 choices; a big one when the rounds say so
            let t = if round % 7 == 0 { 12 } else { 1 + next(3) };
            let call: Vec<i64> = (0..t * 4).map(|_| next(experts) as i64).collect();
            let waves = c.waves(&distinct(&call, experts));
            let mut seen = 0;
            let mut place = vec![-1i64; experts];
            for wave in &waves {
                let g = c.request(wave).unwrap();
                copy(&mut places, &g.loads);
                seen += wave.len();
                for &e in wave {
                    place[e as usize] = c.slot_of(e as usize).unwrap() as i64;
                }
                let mut way = vec![0i64; call.len()];
                fill_way(&call, &place, &mut way);
                for (&id, &w) in call.iter().zip(&way) {
                    if place[id as usize] >= 0 {
                        assert_eq!(places[w as usize], host[id as usize], "round {round}: pair of expert {id} at place {w}");
                    } else {
                        assert_eq!(w, -1);
                    }
                }
                for &e in wave {
                    place[e as usize] = -1;
                }
            }
            assert_eq!(seen, distinct(&call, experts).len(), "every expert of the call is in a wave once");
            assert!(c.resident() <= slots);
        }
    }

    /// A layer computed through the places is the layer all there, to the bit: the host's reference of the experts' products (moe_ref) on weights put where the cache puts them, the pairs' ids replaced by their places
    /// (`fill_way`), against the same on the experts in their own order. The plan of the places is the plan of the experts' pairs, sorted by place. (The library's products need a GPU: `glyd-engine nhoffload`.)
    #[cfg(feature = "dev")]
    #[test]
    fn a_layer_through_the_places_is_the_layer_all_there() {
        use crate::moe::{Act, MoeCfg, Router};
        use crate::moe_ref::{plan_of, RefLayer};
        use crate::qwen35_ref::bf16;
        let (e, slots, d, i, k) = (24usize, 9usize, 64usize, 64usize, 4usize);
        let cfg = MoeCfg { experts: e, router_hidden: d, hidden: d, inter: i, router: Router::sigmoid(k, 5.0), act: Act::Relu2 };
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut value = |scale: f32| bf16(((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale);
        let up: Vec<f32> = (0..e * i * d).map(|_| value(0.3)).collect();
        let down: Vec<f32> = (0..e * d * i).map(|_| value(0.3)).collect();
        let router = vec![0.0f32; e * d];
        let all = RefLayer { cfg: &cfg, router: &router, bias: None, escale: None, up: &up, down: &down };
        let (mut pup, mut pdown) = (vec![0f32; slots * i * d], vec![0f32; slots * d * i]);
        let mut c = LayerCache::new(e, slots);
        let mut served = 0;
        for round in 0..60 {
            let t = 1 + round % 2;
            let xin: Vec<f32> = (0..t * d).map(|_| value(4.0)).collect();
            // a token's k distinct experts, and their weights
            let mut ids = Vec::new();
            for _ in 0..t {
                let mut pick = Vec::new();
                while pick.len() < k {
                    let v = (value(1.0) * 2.0 * e as f32 + e as f32 / 2.0).clamp(0.0, e as f32 - 1.0) as i64;
                    if !pick.contains(&v) {
                        pick.push(v);
                    }
                }
                ids.extend(pick);
            }
            let wts: Vec<f32> = (0..t * k).map(|_| value(1.0) + 0.6).collect();
            let want = all.layer(&xin, t, &ids, &wts);
            let need = distinct(&ids, e);
            let g = c.request(&need).unwrap();
            for &(ex, p) in &g.loads {
                pup[p as usize * i * d..][..i * d].copy_from_slice(&up[ex as usize * i * d..][..i * d]);
                pdown[p as usize * d * i..][..d * i].copy_from_slice(&down[ex as usize * d * i..][..d * i]);
            }
            let mut place = vec![-1i64; e];
            need.iter().for_each(|&x| place[x as usize] = c.slot_of(x as usize).unwrap() as i64);
            let mut way = vec![0i64; ids.len()];
            fill_way(&ids, &place, &mut way);
            let plan = plan_of(&way, e);
            let mut hit_places: Vec<i32> = need.iter().map(|&x| place[x as usize] as i32).collect();
            hit_places.sort_unstable();
            assert_eq!((&plan.hits, plan.order.len()), (&hit_places, ids.len()), "round {round}: the plan of the places");
            let got = RefLayer { cfg: &cfg, router: &router, bias: None, escale: None, up: &pup, down: &pdown }.layer(&xin, t, &way, &wts);
            assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()), "round {round}: the layer through the places differs");
            served += g.hits;
        }
        assert!(served > 0, "the cache was never hit: the test did not reuse a place");
    }

    /// A layer's experts on the host for the tests that follow: the reference's f32 weights as the GPU decodes them (`unpack_int4` and its kin, the second matrices from the row layout), and the experts as the host's
    /// kernels read them (the first matrices placed in rows, the second ones in the row layout). Random bf16 weights, or (`grid`) integers of at most the layout's largest code times 2^-8, the largest in every run of 32
    /// along a row of the matrix as it is packed, which the scale fit lands on exactly: the packs then hold the weights to the bit, the host's included.
    #[cfg(feature = "dev")]
    struct HostLayer {
        cfg: crate::moe::MoeCfg,
        router: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
        experts: Vec<crate::cpu_experts::Expert>,
    }

    #[cfg(feature = "dev")]
    struct Xs(u64);

    #[cfg(feature = "dev")]
    impl Xs {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn uniform(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }

        /// A call of `t` tokens: x [t, d] (bf16 values), each token's `k` distinct experts of `e`, their weights.
        fn call(&mut self, t: usize, e: usize, k: usize, d: usize) -> (Vec<f32>, Vec<i64>, Vec<f32>) {
            let x: Vec<f32> = (0..t * d).map(|_| crate::qwen35_ref::bf16(self.uniform() * 8.0)).collect();
            let mut ids = Vec::new();
            for _ in 0..t {
                let mut pick = Vec::new();
                while pick.len() < k {
                    let v = (self.next() % e as u64) as i64;
                    if !pick.contains(&v) {
                        pick.push(v);
                    }
                }
                ids.extend(pick);
            }
            let wts: Vec<f32> = (0..t * k).map(|_| 0.6 + self.uniform().abs() * 2.0).collect();
            (x, ids, wts)
        }
    }

    #[cfg(feature = "dev")]
    fn host_layer(bits: u32, grid: bool, (e, d, i, k): (usize, usize, usize, usize), rng: &mut Xs) -> HostLayer {
        use crate::cpu_experts::{Expert, Rows};
        use crate::moe::{Act, MoeCfg, Router};
        use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6, pack_rows, transpose, unpack_int4, unpack_int5, unpack_int6, unpack_rows};
        type Pack = fn(&[u16], usize, usize) -> Option<glyd_gpu::pack::Tiered>;
        type Unpack = fn(&glyd_gpu::pack::Tiered, &mut [u16]) -> bool;
        let (pack, unpack, kmax) = match bits {
            6 => (pack_int6 as Pack, unpack_int6 as Unpack, 31i32),
            5 => (pack_int5 as Pack, unpack_int5 as Unpack, 15),
            _ => (pack_int4 as Pack, unpack_int4 as Unpack, 7),
        };
        let bits_of = |v: f32| (v.to_bits() >> 16) as u16;
        let mut weights = |rows: usize, cols: usize| -> Vec<u16> {
            (0..rows * cols)
                .map(|n| {
                    if grid {
                        let code = if n % 32 == 0 {
                            if rng.next() & 1 == 0 {
                                kmax
                            } else {
                                -kmax
                            }
                        } else {
                            (rng.next() % (2 * kmax as u64 + 1)) as i32 - kmax
                        };
                        bits_of(code as f32 * 2f32.powi(-8))
                    } else {
                        bits_of(rng.uniform() * 0.3)
                    }
                })
                .collect()
        };
        let (mut up, mut down, mut experts) = (vec![0f32; e * i * d], vec![0f32; e * d * i], Vec::new());
        for x in 0..e {
            // (the second matrix is held transposed, a neuron's weights a row, its runs of 32 along the hidden size: generated so)
            let (uw, dt) = (weights(i, d), weights(i, d));
            let (up_p, down_p) = (pack(&uw, i, d).unwrap(), pack_rows(bits, &dt, i, d).unwrap());
            let (mut ub, mut db) = (vec![0u16; i * d], vec![0u16; i * d]);
            assert!(unpack(&up_p, &mut ub) && unpack_rows(&down_p, &mut db));
            up[x * i * d..][..i * d].iter_mut().zip(&ub).for_each(|(o, &b)| *o = f32::from_bits((b as u32) << 16));
            // (the reference's down is [hidden, inter]: the row layout holds its transpose)
            down[x * d * i..][..d * i].iter_mut().zip(transpose(&db, i, d)).for_each(|(o, b)| *o = f32::from_bits((b as u32) << 16));
            experts.push(Expert::new(Rows::from_block(&up_p).unwrap(), Rows::new(bits, i, d, down_p.data.clone(), down_p.blocks.clone()).unwrap()).unwrap());
        }
        HostLayer {
            cfg: MoeCfg { experts: e, router_hidden: d, hidden: d, inter: i, router: Router::sigmoid(k, 5.0), act: Act::Relu2 },
            router: vec![0.0; e * d],
            up,
            down,
            experts,
        }
    }

    /// The layer with some of its experts on the host is the layer all there, to a bound: the reference's products over the weights as the GPU decodes them for the experts that have places, the host's kernels
    /// (`cpu_experts`, from the row layouts the store holds) for the others, their sums added in fp32 and rounded to bf16 once, as `moe_rows_sum` does with the host's sums, against the reference of every expert.
    /// The experts have six, five and four bits; random weights, then weights on the grid the packs hold exactly. What differs on random weights is the host's weights not rounded to bf16 (`cpu_experts` measures 1.5e-3 to
    /// 3.7e-3 of a pair's norm) and the order of an fp32 sum, each seen through the final rounding to bf16 (a place of bf16 is 2^-8 to 2^-7 of a value): 3.4e-3 of the layer's norm on average and 5.5e-3 to 6.9e-3 at most
    /// here, against a bound of 1.2e-2 (a place of bf16 at every element, 7.8e-3 at the most, and the host's weights, 3.7e-3); on the grid the host's weights are the GPU's, and what differs is the order of an fp32 sum
    /// alone: the same to the bit but for a rounding now and then (5.9e-5 at most), against a bound of 2e-4.
    #[cfg(feature = "dev")]
    #[test]
    fn a_layer_with_misses_on_the_host_is_the_layer_all_there_to_a_bound() {
        use crate::cpu_experts::{experts_with, Kernel, Pair, Z};
        use crate::moe_ref::RefLayer;
        use crate::qwen35_ref::bf16;
        let (e, d, i, k) = (24usize, 128usize, 256usize, 6usize);
        let mut rng = Xs(0x2545_F491_4F6C_DD1D);
        for (grid, limit) in [(false, 1.2e-2f64), (true, 2e-4)] {
            for bits in [6u32, 5, 4] {
                let l = host_layer(bits, grid, (e, d, i, k), &mut rng);
                let all = RefLayer { cfg: &l.cfg, router: &l.router, bias: None, escale: None, up: &l.up, down: &l.down };
                let (mut worst, mut sum, mut rounds, mut on_host, mut total) = (0f64, 0f64, 0usize, 0usize, 0usize);
                for round in 0..40 {
                    let t = 1 + round % 3;
                    let (x, ids, wts) = rng.call(t, e, k, d);
                    let want = all.layer(&x, t, &ids, &wts);
                    // the experts with no place: about 4 in 10 of the call's
                    let host: Vec<bool> = (0..e).map(|_| rng.next() % 10 < 4).collect();
                    let zb: Vec<u16> = x.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                    let hp: Vec<usize> = (0..t * k).filter(|&j| host[ids[j] as usize]).collect();
                    (on_host, total) = (on_host + hp.len(), total + t * k);
                    let pairs: Vec<Pair> = hp.iter().map(|&j| Pair { expert: &l.experts[ids[j] as usize], token: j / k, weight: wts[j] }).collect();
                    let mut out = vec![0f32; pairs.len() * d];
                    experts_with(Kernel::best(), Z::Bf16(&zb), &pairs, &mut out, 2).unwrap();
                    let (mut diff, mut norm) = (0f64, 0f64);
                    for tok in 0..t {
                        let mut acc = vec![0f64; d];
                        for c in 0..k {
                            let j = tok * k + c;
                            if !host[ids[j] as usize] {
                                let ex = ids[j] as usize;
                                let y = all.down_row(ex, &all.hidden_row(ex, &x[tok * d..(tok + 1) * d]));
                                acc.iter_mut().zip(y).for_each(|(a, v)| *a += v * wts[j] as f64);
                            }
                        }
                        let mut extra = vec![0f32; d];
                        for (row, &j) in out.chunks_exact(d).zip(&hp) {
                            if j / k == tok {
                                extra.iter_mut().zip(row).for_each(|(a, b)| *a += *b);
                            }
                        }
                        for o in 0..d {
                            let got = bf16(acc[o] as f32 + extra[o]) as f64;
                            diff += (got - want[tok * d + o] as f64).powi(2);
                            norm += (want[tok * d + o] as f64).powi(2);
                        }
                    }
                    let rel = (diff / norm).sqrt();
                    (worst, sum, rounds) = (worst.max(rel), sum + rel, rounds + 1);
                }
                eprintln!("{bits} bits, {}: {on_host} of {total} pairs on the host; the layer differs from all there by {:.2e} of its norm on average, {worst:.2e} at most", if grid { "on the grid" } else { "random weights" }, sum / rounds as f64);
                assert!(worst < limit, "{bits} bits, grid {grid}: {worst:e} past {limit:e}");
            }
        }
    }

    /// A layer served as the store serves it, call after call, in each mode: `serve` says the places and the host's experts, the copies it queues put the experts' weights in the places (and the promotions behind the call,
    /// after its products), the products read the places by `fill_way`'s plan and the host computes the pairs of the rest. Every pair the products read finds its own expert's weights in its place, the plan of the places
    /// holds the pairs that have one and the host the others, and the sum (the products' f64 here, the host's fp32 sums, one rounding) is the layer all there to the bound above.
    #[cfg(feature = "dev")]
    #[test]
    fn a_layer_served_through_places_and_the_host_is_the_layer_all_there() {
        use crate::cpu_experts::{experts_with, Kernel, Pair, Z};
        use crate::moe_ref::{plan_of, RefLayer};
        use crate::qwen35_ref::bf16;
        let (e, slots, d, i, k) = (24usize, 7usize, 128usize, 256usize, 6usize);
        let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
        for (mode, bits, grid, limit) in
            [(Mode::Cpu, 4u32, true, 2e-4f64), (Mode::Split, 5, true, 2e-4), (Mode::Auto, 6, false, 1.2e-2), (Mode::Cpu, 5, false, 1.2e-2)]
        {
            let l = host_layer(bits, grid, (e, d, i, k), &mut rng);
            let all = RefLayer { cfg: &l.cfg, router: &l.router, bias: None, escale: None, up: &l.up, down: &l.down };
            let (mut pup, mut pdown) = (vec![0f32; slots * i * d], vec![0f32; slots * d * i]);
            // (the copy of expert x into place p)
            fn put(l: &HostLayer, (pup, pdown): (&mut [f32], &mut [f32]), p: u32, x: u16) {
                let (d, i) = (l.cfg.hidden, l.cfg.inter);
                pup[p as usize * i * d..][..i * d].copy_from_slice(&l.up[x as usize * i * d..][..i * d]);
                pdown[p as usize * d * i..][..d * i].copy_from_slice(&l.down[x as usize * d * i..][..d * i]);
            }
            let (mut cache, mut last) = (LayerCache::new(e, slots), vec![0u64; e]);
            let off = Offload { mode, window: 6, promote: 2, ..Offload::default() };
            let costs = Costs::default();
            let (mut worst, mut kept, mut promoted, mut gpu_pairs, mut total) = (0f64, 0usize, 0usize, 0usize, 0usize);
            for round in 1..=80u64 {
                let t = 1 + round as usize % 3;
                let (x, ids, wts) = rng.call(t, e, k, d);
                let want = all.layer(&x, t, &ids, &wts);
                let needed = distinct(&ids, e);
                let mut pairs_of = vec![0usize; e];
                ids.iter().for_each(|&v| pairs_of[v as usize] += 1);
                let sv = serve(&mut cache, &needed, &pairs_of, &mut last, round, &off, &costs).unwrap();
                sv.loads.iter().for_each(|&(x, p)| put(&l, (&mut pup, &mut pdown), p, x));
                // the plan of the places: each pair's place if its expert has one, else -1, which the plan leaves out
                let mut place = vec![-1i64; e];
                sv.wanted.iter().for_each(|&x| place[x as usize] = cache.slot_of(x as usize).unwrap() as i64);
                let mut way = vec![0i64; ids.len()];
                fill_way(&ids, &place, &mut way);
                let plan = plan_of(&way, e);
                assert_eq!(plan.order.len(), way.iter().filter(|&&w| w >= 0).count());
                let mut ps: Vec<i32> = sv.wanted.iter().map(|&x| place[x as usize] as i32).collect();
                ps.sort_unstable();
                assert_eq!(plan.hits, ps, "round {round}: the plan of the places is the places of the experts that have one");
                // the products read the places: every pair's place holds its own expert's weights
                for (j, &w) in way.iter().enumerate().filter(|(_, &w)| w >= 0) {
                    let (ex, p) = (ids[j] as usize, w as usize);
                    assert!(
                        pup[p * i * d..][..i * d] == l.up[ex * i * d..][..i * d] && pdown[p * d * i..][..d * i] == l.down[ex * d * i..][..d * i],
                        "round {round}: pair {j} of expert {ex} at place {p}"
                    );
                }
                // (the GPU's sums of the pairs that have places, by the places' weights)
                let by_place = RefLayer { cfg: &l.cfg, router: &l.router, bias: None, escale: None, up: &pup, down: &pdown };
                let mut acc = vec![vec![0f64; d]; t];
                for (j, &w) in way.iter().enumerate().filter(|(_, &w)| w >= 0) {
                    let y = by_place.down_row(w as usize, &by_place.hidden_row(w as usize, &x[j / k * d..(j / k + 1) * d]));
                    acc[j / k].iter_mut().zip(y).for_each(|(a, v)| *a += v * wts[j] as f64);
                }
                // behind the call: the promotions' copies (after its products)
                for &(x, p) in &cache.hint(&sv.promote).unwrap().loads {
                    put(&l, (&mut pup, &mut pdown), p, x);
                    promoted += 1;
                }
                // the host's: the pairs with no place
                let hp: Vec<usize> = (0..t * k).filter(|&j| way[j] < 0).collect();
                let zb: Vec<u16> = x.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                let host_pairs: Vec<Pair> = hp.iter().map(|&j| Pair { expert: &l.experts[ids[j] as usize], token: j / k, weight: wts[j] }).collect();
                let mut out = vec![0f32; host_pairs.len() * d];
                experts_with(Kernel::best(), Z::Bf16(&zb), &host_pairs, &mut out, 2).unwrap();
                let (mut diff, mut norm) = (0f64, 0f64);
                for tok in 0..t {
                    let mut extra = vec![0f32; d];
                    for (row, &j) in out.chunks_exact(d).zip(&hp) {
                        if j / k == tok {
                            extra.iter_mut().zip(row).for_each(|(a, b)| *a += *b);
                        }
                    }
                    for o in 0..d {
                        let got = bf16(acc[tok][o] as f32 + extra[o]) as f64;
                        diff += (got - want[tok * d + o] as f64).powi(2);
                        norm += (want[tok * d + o] as f64).powi(2);
                    }
                }
                worst = worst.max((diff / norm).sqrt());
                (kept, gpu_pairs, total) = (kept + sv.kept.len(), gpu_pairs + (t * k - hp.len()), total + t * k);
            }
            eprintln!("{mode:?}, {bits} bits, {}: {gpu_pairs} of {total} pairs on the GPU, {kept} experts left to the host, {promoted} promoted; the layer differs from all there by {worst:.2e} of its norm at most", if grid { "on the grid" } else { "random weights" });
            assert!(worst < limit, "{mode:?} {bits} bits: {worst:e} past {limit:e}");
            assert!(kept > 0 && gpu_pairs > 0 && promoted > 0, "{mode:?}: the host, the places and the promotions were all used");
        }
    }

    /// The trace is what the record says: 2,540 steps of 40 layers of 22 distinct experts of 512.
    #[test]
    fn the_replay_reads_the_trace_as_the_record_describes_it() {
        let &(steps, layers, k, ref ids) = trace();
        assert_eq!((steps, layers, k, ids.len()), (2540, 40, 22, 2540 * 40 * 22));
        assert!(ids.iter().all(|&e| e < 512));
        // a step's choices are 22 distinct experts of the layer
        for row in ids.chunks_exact(k).take(200) {
            assert_eq!(distinct(&row.iter().map(|&e| e as i64).collect::<Vec<_>>(), 512).len(), 22);
        }
    }

    // ---- the disk tier ------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// A source of numbers for the tests of the tier (xorshift; the other tests' `Xs` is a development build's).
    struct Pick(u64);

    impl Pick {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// The bytes of expert `e` for the tests of the tier that need no directory: a pattern of its number, as a read from the disk would leave them.
    fn pattern(e: usize, data: &mut [u8], meta: &mut [u8]) {
        data.iter_mut().enumerate().for_each(|(i, b)| *b = (e * 31 + i * 7) as u8 ^ 0xa5);
        meta.iter_mut().enumerate().for_each(|(i, b)| *b = (e * 17 + i * 3) as u8 ^ 0x3c);
    }

    /// The slot `s` of two byte vectors, the host tier's memory in a test.
    fn slot(mem: &Slots, s: usize) -> (&[u8], &[u8]) {
        // SAFETY: the vectors the test made hold `slots` slots of these strides, and nothing writes them while the slices live.
        unsafe {
            (
                std::slice::from_raw_parts(mem.base[0].add(s * mem.stride[0]), mem.stride[0]),
                std::slice::from_raw_parts(mem.base[1].add(s * mem.stride[1]), mem.stride[1]),
            )
        }
    }

    /// A call's `n` distinct experts out of `experts`: most from a hot set, as a router's are, the rest from all of them.
    fn call_of(x: &mut Pick, n: usize, experts: usize) -> Vec<u16> {
        let mut out: Vec<u16> = Vec::new();
        while out.len() < n {
            let e = if x.next() % 10 < 7 { x.next() % (experts as u64 / 4) } else { x.next() % experts as u64 } as u16;
            if !out.contains(&e) {
                out.push(e);
            }
        }
        out
    }

    /// A tier of 7 slots over 40 experts, asked for calls of 1 to 7 experts for 2,000 rounds with a hot set: after each call every expert asked for is in a slot, the slot holds that expert's bytes exactly, the slots are
    /// all different, the reads are the experts that were not in a slot, and the ones found are the rest; the slots start with the first experts; a read that fails leaves the tier empty (what its slots hold is not known) and the next
    /// call reads again.
    #[test]
    fn the_slots_of_a_disk_tier_hold_the_bytes_of_the_experts_asked_for() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (experts, slots, ds, ms) = (40usize, 7usize, 24usize, 16usize);
        let (reads, fail) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(usize::MAX)));
        let read: Reader = {
            let (reads, fail) = (reads.clone(), fail.clone());
            Arc::new(move |e, d, m| {
                if fail.load(Ordering::SeqCst) == e {
                    return err(format!("expert {e}: the disk said no"));
                }
                reads.fetch_add(1, Ordering::SeqCst);
                pattern(e, d, m);
                Ok(())
            })
        };
        let (mut data, mut meta) = (vec![0xEEu8; slots * ds], vec![0xEEu8; slots * ms]);
        let mem = Slots { base: [data.as_mut_ptr(), meta.as_mut_ptr()], stride: [ds, ms] };
        let mut t = Tier::new(experts, slots, read);
        t.seed(mem).unwrap();
        assert_eq!(reads.load(Ordering::SeqCst), slots, "the slots start with the first experts");
        for e in 0..slots {
            assert_eq!(t.ram.slot_of(e), Some(e));
            let (mut d, mut m) = (vec![0u8; ds], vec![0u8; ms]);
            pattern(e, &mut d, &mut m);
            assert_eq!(slot(&mem, e), (&d[..], &m[..]), "seeded expert {e}");
        }
        let mut x = Pick(0x9E37_79B9_7F4A_7C15);
        let (mut found, mut read_in) = (0usize, 0usize);
        for round in 0..2000 {
            let n = 1 + (x.next() % slots as u64) as usize;
            let wanted = call_of(&mut x, n, experts);
            let had = wanted.iter().filter(|&&e| t.ram.slot_of(e as usize).is_some()).count();
            let before = reads.load(Ordering::SeqCst);
            let got = t.stage(&wanted, mem, 0, &mut || Ok(0)).unwrap();
            assert_eq!((got.hits, got.reads, reads.load(Ordering::SeqCst) - before), (had, wanted.len() - had, wanted.len() - had), "round {round}");
            let mut used = std::collections::HashSet::new();
            for &e in &wanted {
                let s = t.ram.slot_of(e as usize).unwrap_or_else(|| panic!("round {round}: expert {e} is in no slot"));
                let (mut d, mut m) = (vec![0u8; ds], vec![0u8; ms]);
                pattern(e as usize, &mut d, &mut m);
                assert_eq!(slot(&mem, s), (&d[..], &m[..]), "round {round}: expert {e} in slot {s}");
                assert!(used.insert(s), "round {round}: two experts in slot {s}");
            }
            assert!(t.ram.resident() <= slots);
            (found, read_in) = (found + got.hits, read_in + got.reads);
        }
        assert!(found > 2000 && read_in > 2000, "the hot set was found {found} times and the churn read {read_in} experts");
        // a read that fails: the call errs and the tier is empty; asked again, it reads everything
        fail.store(13, Ordering::SeqCst);
        let e = t.stage(&[3, 13, 20], mem, 0, &mut || Ok(0)).unwrap_err().msg;
        assert!(e.contains("expert 13: the disk said no"), "{e}");
        assert_eq!(t.ram.resident(), 0);
        fail.store(usize::MAX, Ordering::SeqCst);
        let got = t.stage(&[3, 13, 20], mem, 0, &mut || Ok(0)).unwrap();
        assert_eq!((got.hits, got.reads), (0, 3));
        // more experts than slots are refused, as the GPU's places refuse them
        assert!(t.stage(&(0..8).collect::<Vec<u16>>(), mem, 0, &mut || Ok(0)).unwrap_err().msg.contains("8 experts asked of a cache of 7"));
    }

    /// The copy stream on the host: a batch queued reads its slots when the test runs it (not when it is queued, as the stream's copies do), and the bytes it reads are checked against the experts it was queued for.
    struct Stream {
        issued: u64,
        retired: u64,
        queue: VecDeque<(u64, Vec<(u32, u16)>)>,
        wrong: usize,
        waits: usize,
    }

    impl Stream {
        /// The copies of `loads` ((expert, slot)) queued: the slots they read are busy until the batch has run.
        fn queue(&mut self, t: &mut Tier, loads: &[(u16, u32)]) {
            self.issued += 1;
            loads.iter().for_each(|&(_, s)| t.busy[s as usize] = self.issued);
            self.queue.push_back((self.issued, loads.iter().map(|&(e, s)| (s, e)).collect()));
        }

        /// The oldest batch run: each copy reads its slot now.
        fn run_one(&mut self, mem: &Slots) {
            let Some((n, copies)) = self.queue.pop_front() else { return };
            for (s, e) in copies {
                let (mut d, mut m) = (vec![0u8; mem.stride[0]], vec![0u8; mem.stride[1]]);
                pattern(e as usize, &mut d, &mut m);
                self.wrong += (slot(mem, s as usize) != (&d[..], &m[..])) as usize;
            }
            self.retired = n;
        }
    }

    /// A slot that a queued copy has yet to read is not written: a tier of 5 slots is asked for calls of 1 to 5 experts, a batch of copies of a call's experts is queued after each call and the stream runs some of them (from none
    /// to all) between calls; the copies read, when they run, the bytes of the expert they were queued for in every case, because a victim whose batch has not run is waited for (`wait` runs the stream's queue) before it is
    /// written; and a call whose victims are all past their batches does not wait. The same run with a wait that runs nothing finds copies that read another expert's bytes: the test can see the fault it is for.
    #[test]
    fn a_slot_a_queued_copy_has_yet_to_read_is_not_written() {
        let (experts, slots, ds, ms) = (30usize, 5usize, 24usize, 16usize);
        let run = |honest: bool| -> (usize, usize, usize) {
            let read: Reader = Arc::new(|e, d, m| {
                pattern(e, d, m);
                Ok(())
            });
            let (mut data, mut meta) = (vec![0u8; slots * ds], vec![0u8; slots * ms]);
            let mem = Slots { base: [data.as_mut_ptr(), meta.as_mut_ptr()], stride: [ds, ms] };
            let mut t = Tier::new(experts, slots, read);
            t.seed(mem).unwrap();
            let mut st = Stream { issued: 0, retired: 0, queue: VecDeque::new(), wrong: 0, waits: 0 };
            let (mut x, mut free_calls) = (Pick(0xD1CE), 0usize);
            for _ in 0..1500 {
                let n = 1 + (x.next() % slots as u64) as usize;
                let wanted = call_of(&mut x, n, experts);
                let waits = st.waits;
                let retired = st.retired;
                // (a call stages, then queues the copies of the experts it wanted that were read: the GPU's misses)
                t.stage(&wanted, mem, retired, &mut || {
                    st.waits += 1;
                    if honest {
                        while !st.queue.is_empty() {
                            st.run_one(&mem);
                        }
                    }
                    Ok(st.retired)
                })
                .unwrap();
                free_calls += (st.waits == waits) as usize;
                let loads: Vec<(u16, u32)> = wanted.iter().filter(|_| x.next() % 3 != 0).map(|&e| (e, t.ram.slot_of(e as usize).unwrap() as u32)).collect();
                st.queue(&mut t, &loads);
                // (the stream makes progress by itself between calls: none, one batch, or all)
                for _ in 0..[0, 1, 9][(x.next() % 3) as usize] {
                    st.run_one(&mem);
                }
            }
            while !st.queue.is_empty() {
                st.run_one(&mem);
            }
            (st.wrong, st.waits, free_calls)
        };
        let (wrong, waits, free_calls) = run(true);
        assert_eq!(wrong, 0, "a copy read the bytes of another expert");
        assert!(waits > 20 && free_calls > 100, "the stream was waited for {waits} times and not at all in {free_calls} calls");
        assert!(run(false).0 > 0, "a stream that is not waited for must show the fault");
    }

    /// A cold start puts the slots back as they were loaded and reads only the slots that changed.
    #[test]
    fn a_cold_start_puts_the_slots_back() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (experts, slots, ds, ms) = (40usize, 6usize, 24usize, 16usize);
        let reads = Arc::new(AtomicUsize::new(0));
        let read: Reader = {
            let reads = reads.clone();
            Arc::new(move |e, d, m| {
                reads.fetch_add(1, Ordering::SeqCst);
                pattern(e, d, m);
                Ok(())
            })
        };
        let (mut data, mut meta) = (vec![0u8; slots * ds], vec![0u8; slots * ms]);
        let mem = Slots { base: [data.as_mut_ptr(), meta.as_mut_ptr()], stride: [ds, ms] };
        let mut t = Tier::new(experts, slots, read);
        t.seed(mem).unwrap();
        // one call that takes the slots of two seeded experts
        t.stage(&[1, 2, 20, 21], mem, 0, &mut || Ok(0)).unwrap();
        let changed = (0..slots).filter(|&e| t.ram.slot_of(e).is_none()).count();
        assert!(changed >= 2, "{changed} experts left");
        reads.store(0, Ordering::SeqCst);
        t.reseed(mem).unwrap();
        assert_eq!(reads.load(Ordering::SeqCst), changed, "only the slots that changed are read");
        for e in 0..slots {
            let s = t.ram.slot_of(e).expect("a seeded expert");
            let (mut d, mut m) = (vec![0u8; ds], vec![0u8; ms]);
            pattern(e, &mut d, &mut m);
            assert_eq!((s, slot(&mem, s)), (e, (&d[..], &m[..])), "expert {e}");
        }
        assert_eq!(t.ram.resident(), slots);
    }

    /// The memory the experts may take is shared by the groups by their bytes: a group of 205 experts of 2.5 MB and one of 307 of 1.8 MB out of 20 of the 39.2 GiB get 104 and 156 of their experts, all together no more than the
    /// budget; a budget that holds every expert is none (`host_memory` is None: the store is as it has always been); at least one slot to a group; and an error where nothing is left.
    #[test]
    fn the_budget_is_shared_by_the_groups_by_their_bytes() {
        let (gib, store) = (1u64 << 30, ExpertStore::new());
        let (threes, twos, b3, b2) = (205usize, 307usize, 2_494_464u64, 1_763_328u64);
        let total = 40 * (threes as u64 * b3 + twos as u64 * b2);
        assert_eq!(store.host_slots(205, b3).unwrap(), 205, "no budget: every expert");
        store.set_host_memory(Some(HostMemory { bytes: total, total }));
        assert!(store.host_memory().is_none(), "a budget that holds every expert is no budget");
        store.set_host_memory(Some(HostMemory { bytes: 20 * gib, total }));
        let (mut slots, mut pinned) = (Vec::new(), 0u64);
        for _ in 0..40 {
            let (a, b) = (store.host_slots(threes, b3).unwrap(), store.host_slots(twos, b2).unwrap());
            slots.push((a, b));
            pinned += a as u64 * b3 + b as u64 * b2;
        }
        assert_eq!(slots[0], (104, 156));
        assert!(pinned <= 20 * gib && pinned > 20 * gib - 40 * (b3 + b2), "{pinned} of {}", 20 * gib);
        assert!(slots.iter().all(|&s| s == slots[0]), "every layer the same");
        // a budget of a fraction of an expert: one slot each; and nothing left once the budget is spent
        store.set_host_memory(Some(HostMemory { bytes: b3 + b2, total }));
        assert_eq!((store.host_slots(205, b3).unwrap(), store.host_slots(307, b2).unwrap()), (1, 1));
        assert!(store.host_slots(205, b3).unwrap_err().msg.contains("spent"));
    }

    /// A tier of 3 slots over a group of the tiny model's directory (read by `HbDisk`, in shards of 24 KB, where an expert's tensors are in two files now and then), asked for calls of 1 to 3 experts over a long random run,
    /// gives the host's half, through `hb_view` (what `Resident::hb_expert` reads the experts through), exactly the experts all there: the codes, the scales and the codebooks of the up and the down matrices of the expert in
    /// the slot are the layer read whole's, in both widths of all three layers, and the bytes of the slot are the place's to the zero padding.
    #[cfg(feature = "dev")]
    #[test]
    fn a_tier_over_a_directory_gives_the_host_the_experts_all_there() {
        use crate::moe_hb::{index_layer, read_layer, HbDisk};
        let dir = std::env::temp_dir().join(format!("glyd-engine-tier-{}", std::process::id()));
        let (out, c) = crate::hbpack::tests::packed_toy(&dir, 24 << 10, 40);
        let ck = glyd_gpu::safetensors::Checkpoint::open(&out).unwrap();
        let (_, manifest) = crate::weights::v7_manifest(&out).unwrap().unwrap();
        let mut x = Pick(0xFACE_FEED);
        let (mut found, mut read_in) = (0usize, 0usize);
        for l in [1usize, 3, 5] {
            let host = read_layer(&ck, &manifest, &c, l).unwrap();
            let ix = index_layer(&manifest, &c, l).unwrap();
            let disk = Arc::new(crate::moe_hb::HbDisk::open(&ck, &c, l, &ix).unwrap());
            let _: &HbDisk = &disk;
            for g in 0..2 {
                let h = host.groups[g].as_ref().unwrap();
                let (place, slots) = (h.place, 3usize);
                let (mut data, mut meta) = (vec![0xEEu8; slots * place.data], vec![0xEEu8; slots * place.meta]);
                let mem = Slots { base: [data.as_mut_ptr(), meta.as_mut_ptr()], stride: [place.data, place.meta] };
                let mut t = Tier::new(h.experts, slots, disk.reader(ix.members[g].clone()));
                t.seed(mem).unwrap();
                for round in 0..300 {
                    let n = 1 + (x.next() % slots as u64) as usize;
                    let wanted = call_of(&mut x, n, h.experts);
                    let got = t.stage(&wanted, mem, 0, &mut || Ok(0)).unwrap();
                    (found, read_in) = (found + got.hits, read_in + got.reads);
                    for &k in &wanted {
                        let k = k as usize;
                        let at = t.ram.slot_of(k).unwrap();
                        // SAFETY: the vectors hold 3 slots of the place's sizes, which nothing writes while the views live.
                        let view = unsafe { hb_view(&place, mem.base[0] as *const u8, mem.base[1] as *const u8, at) }.unwrap();
                        let all = place.expert(&h.data[k * place.data..(k + 1) * place.data], &h.meta[k * place.meta..(k + 1) * place.meta]).unwrap();
                        assert!(
                            view.up.codes == all.up.codes && view.up.scales == all.up.scales && view.up.codebook == all.up.codebook,
                            "layer {l} group {g} round {round}: the up matrix of expert {k} in slot {at}"
                        );
                        assert!(
                            view.down.codes == all.down.codes && view.down.scales == all.down.scales && view.down.codebook == all.down.codebook,
                            "layer {l} group {g} round {round}: the down matrix of expert {k} in slot {at}"
                        );
                        let (d, m) = slot(&mem, at);
                        assert!(
                            d == &h.data[k * place.data..(k + 1) * place.data] && m == &h.meta[k * place.meta..(k + 1) * place.meta],
                            "layer {l} group {g} round {round}: the bytes of expert {k}"
                        );
                    }
                }
            }
        }
        assert!(found > 300 && read_in > 300, "{found} found, {read_in} read: the run churned the slots");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The host's pairs are cut into rounds whose experts the slots hold at once: with slots for all of a call's experts one round (the call as ever), else rounds that cover every pair exactly once, in order, each group's
    /// distinct experts in a round no more than its slots, and an expert met again in a round not counted twice.
    #[test]
    fn the_hosts_pairs_are_computed_in_rounds_of_the_experts_the_slots_hold() {
        let mut x = Pick(0xBEEF);
        for trial in 0..200 {
            let pairs = 1 + (x.next() % 60) as usize;
            let local: [Vec<i64>; 2] = [0, 1].map(|g| (0..pairs).map(|_| (x.next() % [7u64, 13][g]) as i64).collect());
            let mine: Vec<(usize, usize)> = (0..pairs).map(|j| ((x.next() % 2) as usize, j)).collect();
            let all = host_rounds(&mine, &local, [usize::MAX; 2]);
            assert_eq!((all.len(), all[0].0, all[0].1), (1, 0, pairs), "trial {trial}: slots for every expert is one round");
            let slots = [1 + (x.next() % 4) as usize, 1 + (x.next() % 5) as usize];
            let rounds = host_rounds(&mine, &local, slots);
            let mut at = 0;
            for (start, end, seen) in &rounds {
                assert_eq!(*start, at, "trial {trial}: rounds follow each other");
                assert!(end > start && *end <= pairs, "trial {trial}: a round has pairs");
                for g in 0..2 {
                    let mut want: Vec<u16> = Vec::new();
                    for &(gg, j) in &mine[*start..*end] {
                        if gg == g && !want.contains(&(local[g][j] as u16)) {
                            want.push(local[g][j] as u16);
                        }
                    }
                    assert_eq!(&want, &seen[g], "trial {trial}: a round's experts of group {g}");
                    assert!(seen[g].len() <= slots[g], "trial {trial}: {} experts of group {g} in slots of {}", seen[g].len(), slots[g]);
                }
                at = *end;
            }
            assert_eq!(at, pairs, "trial {trial}: every pair is in a round");
        }
        assert!(host_rounds(&[], &[Vec::new(), Vec::new()], [1, 1]).is_empty());
    }

    /// The plan on the GPU (src/devplan.rs, kernels/moe.cu xp_plan: a clock over the places) over the record's decode traces, every miss copied: the share of pairs it finds in places against the
    /// store's least-recently-used cache's, a row a step and 8 rows a step (generations side by side, as `replay_rows`). The clock finds 0.5 to 0.6 points fewer at one row (60.7, 72.8 and 79.4%
    /// against 61.3, 73.3 and 79.9), within 2.5 points as checked; at 8 rows 5.1 points more to 0.6 fewer (printed).
    #[test]
    fn the_plan_on_the_gpu_over_the_traces() {
        let &(_, layers, k, ref ids) = trace();
        let copy_all = |_: usize| crate::devplan::Params { copy: 1.0, fixed: 0.0, pair: 1.0, row: 1.0, mode: Mode::Copy, window: 0, max_promote: 0 };
        for cap in [87usize, 164, 236] {
            for rows in [1usize, 8] {
                let (mut found, mut pairs) = (0usize, 0usize);
                for g0 in (0..20).step_by(rows).filter(|g| g + rows <= 20) {
                    for l in 0..layers {
                        let mut pl = crate::devplan::Policy::new(512, cap);
                        for t in 0..127 {
                            let step: Vec<i64> = (g0..g0 + rows).flat_map(|g| ids[((g * 127 + t) * layers + l) * k..][..k].iter().map(|&e| e as i64)).collect();
                            found += step.iter().filter(|&&e| pl.place_of[e as usize] >= 0).count();
                            pairs += step.len();
                            // (a call's distinct experts past the places: the rest the host's, as the plan on the GPU sends them)
                            pl.call(&step, &copy_all(cap));
                        }
                    }
                }
                let clock = found as f64 / pairs as f64;
                let lru = replay_rows(cap, rows, false).2;
                eprintln!(
                    "{cap} places, {rows} rows: the plan on the GPU finds {:.1}% of pairs in places, the store's cache {:.1}%",
                    100.0 * clock,
                    100.0 * lru
                );
                if rows == 1 {
                    assert!((clock - lru).abs() < 0.025, "{cap} places: {clock:.3} against {lru:.3}");
                }
            }
        }
    }
}

// The expert cache on a driver of the host (tools/cuda_fake.cpp), for the development build's check of the copies, the events, the host's pairs and the disk tier without a GPU.
#[cfg(all(test, feature = "dev"))]
#[path = "offload_fake.rs"]
mod fake_gpu;
