//! The description executor as a scheduler's step (`sched::ModelStep`, docs/design/runtime.md section 2): several sequences, each with its own room of the context (`add_seqs`), run by the one
//! scheduler (`src/sched.rs`). A step of one sequence is that sequence's pass as a reply alone runs it (its prompt chunk, or its graph's one-token step). A step of several runs their rows
//! in one pass: the operations that act row by row (the products, norms, activations) on all the rows at once, so each weight is read once for the step, and the operations on a sequence's cache or
//! states (the attention's cache write and read, the recurrent layers) on each sequence's rows by themselves with the kernels and row counts its own pass has; the pass's rows that decode, a token
//! of a sequence each, take an attention layer in one launch of each kernel (one call of the library's decode), each row through its own block table (`RowTabs`), which is each row's own pass's
//! arithmetic; so do the recurrent layers' operations (the gated delta rule's and Mamba-2's convolution and recurrence: `rows_state`), each row through its own state (`RowTabs::states`). A
//! step of nothing but such rows replays a CUDA graph of its rows padded to a ladder of sizes (`rows_bucket`), where the description is one of attention layers, such recurrent layers and
//! operations that act row by row: its padding rows read and write no key and no state. What a row computes never turns
//! on what the other sequences hold. It can turn on how many rows the step has, because a product's kernel is picked by them: in a step of 16 rows or fewer every row's logits are, bit for bit,
//! those of its own pass; in a larger one (17 to 64 rows another kernel, past 64 `decoded_way`'s) they differ by the rounding of the engine's bf16 values, as the engine differs from an f32
//! reference. Measured by `glyd-engine roundcheck` (src/logits.rs) on every family and on gemma-4-12B at wren with a text's tokens: a KL of 0.002 to 0.005 from the row alone on average, the
//! largest logit another token's in 1 or 2 steps of 32 (where the two largest are tied or nearly); benchmarks/gpu/rtx4080-one-step-round-2026-11-22 has the figures. GLYD_ENGINE_BATCH_PREFILL=0 runs prompts one
//! a pass, the round as two passes and the rows that decode in passes of at most 16 (a mixture of experts' one by one): every reply is then the reply alone, bit for bit.
//!
//! A finished sequence stays in its slot (`ModelStep::keeps`), and a request that begins with its tokens goes on from them (`resume`): a conversation's next turn, which sends the turns
//! before it again. The whole of what a slot holds can always be gone on with. Before its end: anywhere, where the description has no recurrent state and no latent cache (Gemma 4, Muse
//! Spark: the keys and values past the place are written over); from the prompt's end, where the states were copied aside (`mark`), where it has recurrent states (the Qwen3.5 layout,
//! and Kimi-K3 and GLM-5.3-Flash, whose latent caches are rings like any keys). `glyd-engine keepcheck` compares each against a sequence that never ended, bit for bit.
//!
//! A request that begins as another sequence does, kept or running, shares that start (`share`): it names the other's full blocks (each then held by both, and written by
//! neither: `Pages::share`) and runs only the rest. A model with recurrent states shares at the other's mark, with the states copied.
//!
//! What a step of several sequences supports is decided per description (`Dyn::batchable`): every operation must be one that acts row by row, or one of the cache operations below that knows where a
//! sequence's rows begin. A description with another runs its sequences one pass each, concurrently but not batched.

use super::pages::{Pages, BLOCK};
use super::snaps;
use super::{Dyn, Op};
use crate::dev::{copy_d2d_async, Buf};
use crate::ffi::{err, Result};
use crate::sched::{ModelStep, PrefillJob, Row, Sample, FUSED_ROWS};

/// Paged keys and values: the blocks every sequence holds, each ring's layer group, and each sequence's block tables on the GPU (a buffer a group: every slot's row of `tlen`
/// entries, int32 a BLOCK of tokens, slot after slot), and the tables of a pass's decode rows (`RowTabs`; None where an attention layer's calls cannot take them).
pub(super) struct Paged {
    pub(super) pages: Pages,
    pub(super) group: Vec<usize>,
    tabs: Vec<Buf>,
    tlen: usize,
    pub(super) rows: Option<RowTabs>,
}

/// The block tables of a pass's rows that decode, one token of a sequence each (as SGLang's decode backend takes a batch's every sequence in one
/// call through its own block table): row r of group g's `tab` holds `dup[g]` copies of row r's sequence's table, made on the GPU at the pass's start (`Dyn::row_tables`, from the
/// slots the host writes to `slot`, -1 a padding row) so that one launch of the preparation and one call of the decode take a layer's every such row; a call of the library's
/// decode whose rows' query heads are `sp` sequences finds row r's j-th at stride `dup / sp` tables (every `sp` of the group's divides `dup`). `win`: each group's window (0: none).
/// `meta`: the step's metadata, which the host writes for a pass in one copy and which a graph's replay reads where it was captured (no capture again for other rows): the
/// pass's positions (`pcap` entries, a chunk's), the slots of its rows that decode (`rows` entries, -1 a padding row), and the addresses of each such row's recurrent states
/// (row r's state i at entry r `nst` + i, as vLLM's state indices pick a request's slot of the state pools; 0 for a padding row), read by the
/// recurrent layers' kernels of such rows (`Dyn::state_lead`).
/// `graphs`: whether a step of such rows alone may replay a CUDA graph of its padded rows (`Dyn::rows_bucket`): every operation of the description acts row by row, is an
/// attention layer or is a recurrent layer's operation that takes such rows (`rows_state`).
pub(super) struct RowTabs {
    pub(super) tab: Vec<Buf>,
    pub(super) dup: Vec<usize>,
    pub(super) win: Vec<usize>,
    pub(super) meta: Buf,
    pub(super) pcap: usize,
    pub(super) nst: usize,
    pub(super) rows: usize,
    pub(super) graphs: bool,
}

impl RowTabs {
    /// Where the pass's positions are (`meta`).
    pub(super) fn pos_at(&self) -> u64 {
        self.meta.ptr
    }

    /// Where the slots of the pass's rows that decode are.
    pub(super) fn slot_at(&self) -> u64 {
        self.meta.ptr + (4 * self.pcap) as u64
    }

    /// Where the rows' state table is.
    pub(super) fn states_at(&self) -> u64 {
        self.meta.ptr + meta_states(self.pcap, self.rows) as u64
    }

    /// The pass's positions, its decode rows' slots and their state table (`state_table`), in one copy (the GPU waited for).
    pub(super) fn write(&self, pos: &[i32], slots: &[i32], states: &[u64]) -> Result<()> {
        self.meta.write(0, &meta_bytes(self.pcap, self.rows, pos, slots, states))
    }
}

/// The byte where `RowTabs::meta`'s state table begins: after `pcap` positions and `rows` slots, at 8 bytes.
fn meta_states(pcap: usize, rows: usize) -> usize {
    (4 * (pcap + rows)).next_multiple_of(8)
}

/// `RowTabs::meta`'s bytes for a pass: positions from 0, slots from 4 pcap, the state table from `meta_states`; only as far as what is written reaches.
pub(super) fn meta_bytes(pcap: usize, rows: usize, pos: &[i32], slots: &[i32], states: &[u64]) -> Vec<u8> {
    assert!(pos.len() <= pcap && slots.len() <= rows, "a pass within the metadata's room");
    let end = if !states.is_empty() {
        meta_states(pcap, rows) + 8 * states.len()
    } else if !slots.is_empty() {
        4 * (pcap + slots.len())
    } else {
        4 * pos.len()
    };
    let mut b = vec![0u8; end];
    for (i, p) in pos.iter().enumerate() {
        b[4 * i..4 * i + 4].copy_from_slice(&p.to_le_bytes());
    }
    for (i, s) in slots.iter().enumerate() {
        b[4 * (pcap + i)..4 * (pcap + i) + 4].copy_from_slice(&s.to_le_bytes());
    }
    let at = meta_states(pcap, rows);
    for (i, a) in states.iter().enumerate() {
        b[at + 8 * i..at + 8 * i + 8].copy_from_slice(&a.to_le_bytes());
    }
    b
}

/// Whether an operation on a sequence's recurrent state has a kernel that takes a step's rows that decode, a token of a sequence each, each through its own state
/// (`Dyn::state_lead`): the gated delta rule's and Mamba-2's convolution and recurrence.
pub(super) fn rows_state(op: &Op) -> bool {
    matches!(op, Op::GdnConv { .. } | Op::GdnDelta { .. } | Op::MambaConv { .. } | Op::MambaScan { .. })
}

/// Whether a step of rows that decode alone, padded, can be one CUDA graph (`RowTabs::graphs`): every operation acts row by row, is an attention layer, or is a recurrent
/// layer's that takes such rows; the latent attention runs its sequences by their own pointers, so not it.
pub(super) fn rows_graphable<'a>(mut ops: impl Iterator<Item = &'a Op>) -> bool {
    ops.all(|op| match op {
        Op::Attn(_) => true,
        Op::Mla(_) => false,
        op => row_wise(op) || rows_state(op),
    })
}

/// The entries of the rows' state table (`RowTabs::states`) for a pass whose first rows decode, each with its sequence's states `lead[r]` (`Part::states`), the rest of `rows`
/// padding (0): row r's state i at r nst + i.
pub(super) fn state_table(lead: &[&[u64]], rows: usize, nst: usize) -> Vec<u64> {
    let mut t = vec![0u64; rows * nst];
    for (r, st) in lead.iter().enumerate() {
        t[r * nst..(r + 1) * nst].copy_from_slice(st);
    }
    t
}

/// The least common multiple of a and b (both at least 1).
fn lcm(a: usize, b: usize) -> usize {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// Each group's copies of a row's table (`RowTabs::dup`): the least common multiple of the sequences a row's query heads are in the decode's calls on the group's rings (`sp`:
/// each attention layer's ring and its calls' sequences a row), 0 for a group no attention layer reads.
pub(super) fn row_dups(groups: usize, sp: impl Iterator<Item = (usize, usize)>) -> Vec<usize> {
    let mut dup = vec![0usize; groups];
    for (g, sp) in sp {
        dup[g] = lcm(dup[g].max(1), sp.max(1));
    }
    dup
}

/// The rows a CUDA graph of a step of `n` rows that decode is captured for (`Dyn::rows_bucket`): the least power of two at or above n below `most`, else `most` (the
/// head's rows, which a step never passes); None for one row (the sequence's own graph) or more than `most`.
pub(super) fn bucket(n: usize, most: usize) -> Option<usize> {
    if n < 2 || n > most {
        return None;
    }
    Some(n.next_power_of_two().min(most))
}

/// The sampler's candidates of a step's rows (`ModelStep::top`): the `kern::TOPK` largest logits of each head row of the last step, made by one launch for all
/// of them at the first ask after the step and copied to the host, where the scheduler would read the whole row (Qwen3.5's 248,320 logits, Gemma 4's 262,144) to
/// copy, widen and draw from on the host. `rows`: the last step's head rows; `made`: whether its
/// candidates are; `bufs`: the GPU's and the host's room (tokens, logits; for `cap` rows), made at the first ask.
#[derive(Default)]
pub(super) struct Cands {
    bufs: Option<(Buf, Buf, crate::dev::Pinned, usize)>,
    rows: usize,
    made: bool,
    /// Each head row's sum of exponentials made with the step (a launched one's, `ModelStep::draws`): the temperature and the sum.
    sums: Vec<Option<(f32, f64)>>,
    /// Where a grammar's mask made the candidates, the rows' own largest logits before it (the log-probabilities' top: the row's, not the allowed tokens'), in the host's
    /// pinned room (tokens, then logits, `cap` rows of each), and the rows they were made for.
    raw: Option<crate::dev::Pinned>,
    raw_rows: usize,
}

impl Cands {
    /// A step that wrote `rows` head rows ran: its candidates are not made yet.
    fn fresh(&mut self, rows: usize) {
        (self.rows, self.made, self.raw_rows) = (rows, false, 0);
        self.sums.clear();
    }
}

/// The steps launched ahead after which a model with an expert cache has one land before the next is launched, for the store's end of a step (`ModelStep::quiet`).
const QUIET_EVERY: usize = 16;

/// How many steps' rooms `Lanes` keeps: one running, the next queued behind it, and the one before them, whose GPU times `Gaps` still reads.
const LANES: usize = 3;

/// What a step that `overlaps` is launched with (`ModelStep::launch`, docs/design/runtime.md section 5, "Launch and finish"): the GPU's token ids for the embedding's lookup
/// (`Lib::rows`, which reads them on the GPU: a row's token may be the one the step before chose, `Row::after`) and their sources, and a lane for each of LANES steps in turn:
/// the host's pinned room for the step's inputs (positions, the ids' sources, the block tables it changed) and for its samples, and an event after its last work, which
/// `finish` waits for (not the whole stream). A lane is written again three launches later, when its step and the one after it have been finished: its copies have run.
pub(super) struct Lanes {
    ids: Buf,
    src: Buf,
    lane: Vec<Lane>,
    next: usize,
    /// Steps launched and not finished.
    out: usize,
    /// Whether a step's head rows are the logits' rows the scheduler may read (`top`, `logits`, `gather`): not once a later step is launched over them.
    pub(super) stale: bool,
    /// Each lane's step's slot in `Gaps` (None: not timed).
    gap: [Option<usize>; LANES],
}

struct Lane {
    /// Positions (int32 a row), then the ids' sources (int32 a row: a token, or -1 - the head row of the step before), a chunk's rows of each.
    ins: crate::dev::Pinned,
    /// The head rows' tokens (int32) and largest logits (f32).
    outs: crate::dev::Pinned,
    /// The block tables as this step's inputs have them ([slot][group][entry], int32), from which the entries a step changed are copied.
    tabs: crate::dev::Pinned,
    done: crate::dev::Ev,
    /// The step: its decode rows, whether each prompt chunk ends its prompt, its prompt tokens, whether it was the step graph's replay, the host's time to queue it (ms),
    /// and GLYD_PROF's events of its operations (an eager pass's).
    rows: usize,
    lasts: Vec<bool>,
    prompt: usize,
    graph: bool,
    queue_ms: f64,
    evs: Vec<(&'static str, crate::dev::Ev)>,
    /// Whether the step's candidates were queued with it (`ModelStep::draws`).
    drawn: bool,
    /// The 12-bit pages' status words of each group after the step (its void pages' ring, `kv12_lost`), copied with its samples.
    status: crate::dev::Pinned,
    /// The head rows' sums of exponentials the step was asked for (f64 a row), and the asks.
    sums: crate::dev::Pinned,
    asks: Vec<Option<f32>>,
}

impl Lanes {
    /// Room for steps of up to `chunk` rows and `head` sampled rows, with tables of `tab_words` entries in all and `status_words` of the 12-bit pages' status.
    fn new(chunk: usize, head: usize, tab_words: usize, status_words: usize) -> Result<Lanes> {
        let lane = (0..LANES)
            .map(|_| {
                Ok(Lane {
                    ins: crate::dev::Pinned::new(8 * chunk)?,
                    outs: crate::dev::Pinned::new(8 * head)?,
                    tabs: crate::dev::Pinned::new(4 * tab_words)?,
                    done: crate::dev::Ev::untimed()?,
                    rows: 0,
                    lasts: Vec::new(),
                    prompt: 0,
                    graph: false,
                    queue_ms: 0.0,
                    evs: Vec::new(),
                    drawn: false,
                    status: crate::dev::Pinned::new(4 * status_words)?,
                    sums: crate::dev::Pinned::new(8 * head)?,
                    asks: Vec::new(),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Lanes { ids: Buf::zeroed(8 * chunk)?, src: Buf::zeroed(4 * chunk)?, lane, next: 0, out: 0, stale: false, gap: [None; LANES] })
    }
}

/// The GPU's own time of each step and its idle time before it (GLYD_ENGINE_STEP_PROF=1 and GLYD_PROF=1): an event at a step's first work and one after its last, in a ring
/// of LANES; a step's idle time is from the end of the step before to its start, both read once the step is done. With a step that waits for its tokens, the GPU is idle for
/// the host's work between steps (the sampler's, the scheduler's, the server's) and for the host's time to queue the step's first work; with the next step launched before
/// a step is read, for less.
pub(super) struct Gaps {
    start: Vec<crate::dev::Ev>,
    end: Vec<crate::dev::Ev>,
    /// The ring's slot of the step begun last, and whether the slot before it has an end recorded.
    at: std::cell::Cell<usize>,
    ended: std::cell::Cell<[bool; LANES]>,
}

impl Gaps {
    pub(super) fn new() -> Result<Gaps> {
        let evs = || (0..LANES).map(|_| crate::dev::Ev::new()).collect::<Result<Vec<_>>>();
        Ok(Gaps { start: evs()?, end: evs()?, at: std::cell::Cell::new(LANES - 1), ended: std::cell::Cell::new([false; LANES]) })
    }

    /// A step begins: its slot, the start recorded on `s` (before its first work).
    pub(super) fn begin(&self, s: crate::ffi::Stream) -> Result<usize> {
        let k = (self.at.get() + 1) % LANES;
        self.at.set(k);
        let mut e = self.ended.get();
        e[k] = false;
        self.ended.set(e);
        self.start[k].record(s)?;
        Ok(k)
    }

    /// The step of slot `k` has queued its last work: its end recorded on `s` (again where it is recorded more than once: the last counts).
    pub(super) fn end(&self, k: usize, s: crate::ffi::Stream) -> Result<()> {
        self.end[k].record(s)?;
        let mut e = self.ended.get();
        e[k] = true;
        self.ended.set(e);
        Ok(())
    }

    /// The slot of the step begun last.
    pub(super) fn last(&self) -> usize {
        self.at.get()
    }

    /// The step of slot `k`, done: its GPU time and the GPU's idle time before it (ms; None for the first step, or one whose step before recorded no end).
    pub(super) fn read(&self, k: usize) -> Result<(f64, Option<f64>)> {
        let gpu = self.end[k].since(&self.start[k])? as f64;
        let before = (k + LANES - 1) % LANES;
        let idle = match self.ended.get()[before] {
            true => Some((self.start[k].since(&self.end[before])? as f64).max(0.0)),
            false => None,
        };
        Ok((gpu, idle))
    }
}

/// One sequence's rows in a pass over several: the first row, how many, where its tokens begin in the sequence, and its caches and recurrent states.
pub(super) struct Part {
    /// The sequence (its index in `Dyn::seqs`, its latent attention caches' too).
    pub slot: usize,
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
            | Op::GatedResidual { .. }
            | Op::KdaOut { .. }
            | Op::LinearAt { .. }
            | Op::Parts { .. }
            | Op::DepthMix(_)
            | Op::Add { .. }
            | Op::MoeEarly { .. }
            | Op::GatedAct { .. }
            | Op::Moe(_)
            | Op::Streams
            | Op::StreamsPre { .. }
            | Op::StreamsPost { .. }
            | Op::StreamsHead { .. }
            // (the latent attention runs its products over every row and its caches' parts by sequence itself: `Dyn::seq_parts`)
            | Op::Mla(_)
            | Op::MambaNorm { .. }
            // (DeepSeek-V4's: by row, or over the rows as the pass lays them out; its attention's output is laid out by group over the pass's rows, which its low rank reads)
            | Op::Fp8 { .. }
            | Op::CompQ { .. }
            | Op::IdxQ { .. }
            | Op::OGroupLora { .. }
            | Op::OGroupLoraStack { .. }
            | Op::Nop
    )
}

/// Whether a description's keys and values can be paged: it has a ring, and every operation that touches one reads it through a block table (the attention layer
/// kind's (a sparse one's too: its keys and values, its indexer's raw keys and pools), the latent attention kind's, the compressed attention kind's: its window, its
/// compressed entries and its indexer's).
pub(super) fn pages_all(d: &super::ModelDesc) -> bool {
    std::iter::once(&d.first).chain(d.layers.iter().flatten()).all(|op| {
        op.rings().is_empty()
            || matches!(
                op,
                Op::Attn(_)
                    | Op::Mla(_)
                    | Op::CompKv { .. }
                    | Op::Compress { .. }
                    | Op::IdxKeys { .. }
                    | Op::IdxPick(_)
                    | Op::CompAttn { .. }
                    | Op::IdxMean { .. }
            )
    }) && d.layers.iter().flatten().any(|op| !op.rings().is_empty())
}

/// The blocks of a layer group one sequence holds at most (`Dyn::page` caps a group's pool at this many for each user): its window and a chunk (`keep`), or the
/// whole context (None), up to the context, in blocks of `covers` tokens, and one more (the block a window straddles). The planner sizes a sliding layer's ring
/// by it (`plan::sizing`, Gemma 4's `kv_bytes`).
pub(crate) fn seq_blocks(keep: Option<usize>, ctx: usize, covers: usize) -> usize {
    keep.map_or(ctx, |k| k.min(ctx)).div_ceil(covers) + 1
}

/// The blocks of each layer group's pool, from `budget` bytes: one sequence's blocks in every group first (`least[g]`, where the budget holds them all: a sequence alone
/// always fits, as long as the plan's context, which is sized for one where the users share the pools), then every group as many tokens' blocks again (a block of group g
/// is `covers[g]` tokens of each of its rings, `per[g]` bytes a token), none more than `most[g]`; what the groups capped there leave goes to the full groups (`keep[g]`
/// None), as many tokens' again. (As many tokens' blocks alone gave a sliding group, whose bytes a token are most of a model's, as many tokens as the full group
/// where the budget was short of every user's: one sequence then held less than the context the plan gave it.)
pub(crate) fn pool_blocks(budget: usize, per: &[usize], covers: &[usize], keep: &[Option<usize>], most: &[usize], least: &[usize]) -> Vec<usize> {
    let bytes = |b: &[usize]| (0..per.len()).map(|g| b[g] * covers[g] * per[g]).sum::<usize>();
    let one: Vec<usize> = (0..per.len()).map(|g| least[g].min(most[g])).collect();
    let base = if bytes(&one) <= budget { one } else { vec![0; per.len()] };
    let even = (budget - bytes(&base)) / per.iter().sum::<usize>().max(1);
    let mut blocks: Vec<usize> = (0..per.len()).map(|g| (base[g] + even / covers[g]).min(most[g])).collect();
    let left = budget - bytes(&blocks);
    let full: usize = (0..per.len()).filter(|&g| keep[g].is_none()).map(|g| per[g]).sum();
    if full > 0 {
        let more = left / full;
        for g in (0..per.len()).filter(|&g| keep[g].is_none()) {
            blocks[g] = (blocks[g] + more / covers[g]).min(most[g]);
        }
    }
    blocks
}

/// The pools' layer groups of `rings`, of which the engine holds those `held` says (`Dyn::page`): a group is the rings of one window and one ratio (a ring it does not
/// hold, a group of its own). Each ring's group, and each group's window (0: the whole context; usize::MAX: rings not held) and ratio.
pub(crate) fn ring_groups(rings: &[super::Ring], held: &[bool]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut windows: Vec<(usize, usize)> = Vec::new();
    let mut group = Vec::with_capacity(rings.len());
    for (r, ring) in rings.iter().enumerate() {
        let w = (if held[r] { ring.window } else { usize::MAX }, ring.ratio.max(1));
        if !windows.contains(&w) {
            windows.push(w);
        }
        group.push(windows.iter().position(|&x| x == w).expect("just added"));
    }
    (group, windows)
}

/// How a description's sequences hold their keys and values on the executor (`Dyn::serve_rows`): one sequence alone (an operation keeps a sequence's data outside
/// its rings and states: `one_sequence_op`), in pools of blocks they share (`pages_all`), or each in rings of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hold {
    One,
    Pooled,
    Own,
}

/// What a description's sequences take on the executor beside its weights, the whole model on one GPU with no draft model (`Dyn::serve_rows`; keys and values in
/// bf16), at a context and a prompt chunk: the one rule a plan's keys, values and states are sized by (src/plan.rs `sizing`) and the loader takes a plan's bytes by
/// (`Dyn::take_plan`): a plan never promises more than the pools hold, and the pools never take more than the plan gives them.
#[derive(Clone, Debug)]
pub(crate) struct SeqRoom {
    pub hold: Hold,
    /// A sequence's recurrent states (every state of the description, each at least the 4 bytes a buffer takes), and whether the engine keeps a copy of each
    /// sequence's at its mark and a pool of them at block boundaries (`ModelStep::keeps`, `snaps::count`).
    pub state: u64,
    pub marks: bool,
    /// One sequence's rings as the engine loads them (each ring's rows at the context: `Ring::rows`).
    pub rings: u64,
    /// The pools' groups (`ring_groups`): each one's bytes a token (bf16), the tokens a block covers, the blocks one sequence holds of it at most (`seq_blocks` of
    /// its window and a chunk, else the whole context), and a block's bytes of its rings (block 0 of each group, which no sequence holds, is one).
    pub per: Vec<usize>,
    pub covers: Vec<usize>,
    pub least: Vec<usize>,
    pub block: Vec<u64>,
}

/// Whether the engine keeps a model's states at each sequence's mark and at block boundaries, where it has states and no draft model (`ModelStep::keeps`): unless
/// GLYD_ENGINE_KEEP=0 or GLYD_ENGINE_BATCH_PREFILL=0 says not.
pub(crate) fn marks_kept() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_KEEP").is_none_or(|v| v != "0")) && Dyn::batch_prefill()
}

/// Whether the engine pages a description's keys and values where it can (`pages_all`): unless a development build's GLYD_ENGINE_PAGED=0 says not.
fn paging_asked() -> bool {
    crate::tune!("GLYD_ENGINE_PAGED").is_none_or(|v| v != "0")
}

impl SeqRoom {
    /// The rule for the description `d` at `ctx` tokens of context and prompt chunks of `chunk`, rounded as the engine rounds them (`Dyn::build`).
    pub(crate) fn of(d: &super::ModelDesc, ctx: usize, chunk: usize) -> SeqRoom {
        let (ctx, chunk) = (ctx.div_ceil(16) * 16, chunk.max(1));
        let hold = match (one_sequence_op(d).is_some(), pages_all(d) && paging_asked()) {
            (true, _) => Hold::One,
            (false, true) => Hold::Pooled,
            (false, false) => Hold::Own,
        };
        let state = d.states.iter().map(|&b| b.max(4) as u64).sum::<u64>();
        let room = |r: &super::Ring, rows: usize| (rows * r.nk * r.d * 2 * (1 + r.values as usize)) as u64;
        let rings = d.rings.iter().map(|r| room(r, r.rows(ctx, chunk))).sum();
        let (group, windows) = ring_groups(&d.rings, &vec![true; d.rings.len()]);
        let (mut per, mut block) = (vec![0usize; windows.len()], vec![0u64; windows.len()]);
        for (r, ring) in d.rings.iter().enumerate() {
            per[group[r]] += ring.token_bytes();
            block[group[r]] += room(ring, BLOCK);
        }
        let covers: Vec<usize> = windows.iter().map(|w| BLOCK * w.1).collect();
        let keep: Vec<Option<usize>> = windows.iter().map(|&(w, _)| (w > 0 && w != usize::MAX).then_some(w + chunk)).collect();
        let least = keep.iter().zip(&covers).map(|(&k, &c)| seq_blocks(k, ctx, c)).collect();
        SeqRoom { hold, state, marks: !d.states.is_empty() && marks_kept(), rings, per, covers, least, block }
    }

    /// The pools' budget (`pool_blocks`'s) of `k` sequences' blocks in every group, each one's at most (`least`): what one sequence's context takes, and every user's.
    fn budget_of(&self, k: usize) -> u64 {
        (0..self.per.len()).map(|g| (k * self.least[g] * self.covers[g] * self.per[g]) as u64).sum()
    }

    /// What `users` sequences take whatever their keys and values: pooled, their states, where the engine keeps them, a copy of each at its mark and the pool of
    /// copies at block boundaries, and each group's block 0; each in its own rings, every one's rings and states and its mark's copy; one alone, its rings and states.
    fn fixed(&self, users: usize) -> u64 {
        let (u, m) = (users as u64, self.marks as u64);
        match self.hold {
            Hold::One => self.rings + self.state,
            Hold::Own => u * (self.rings + self.state * (1 + m)),
            Hold::Pooled => u * self.state * (1 + m) + m * snaps::count(self.state as usize) as u64 * self.state + self.block.iter().sum::<u64>(),
        }
    }

    /// The bytes of `users` sequences where they share one sequence's context: the least a plan gives them (each holding the whole context, the most: `plan` with
    /// no bound).
    pub(crate) fn least_bytes(&self, users: usize) -> u64 {
        match self.hold {
            Hold::Pooled => self.fixed(users) + self.budget_of(1),
            Hold::Own => self.fixed(1),
            Hold::One => self.fixed(1),
        }
    }

    /// The sequences a plan for `users` holds, and their bytes, where `left` bytes of the GPU are theirs: in pools, every user's states and the pools what is left,
    /// at most every user's context and at least one sequence's (as many users as leave room for that beside their states: they share the pools); each in its own
    /// rings, as many as fit; one alone. None where not one fits.
    pub(crate) fn plan(&self, users: usize, left: u64) -> Option<(usize, u64)> {
        let users = users.max(1);
        match self.hold {
            Hold::One => (self.fixed(1) <= left).then_some((1, self.fixed(1))),
            Hold::Own => {
                let k = ((left / self.fixed(1).max(1)) as usize).min(users);
                (k >= 1).then(|| (k, self.fixed(k)))
            }
            Hold::Pooled => {
                let k = (1..=users).rev().find(|&k| self.least_bytes(k) <= left)?;
                let pool = (left - self.fixed(k)).clamp(self.budget_of(1), self.budget_of(k));
                Some((k, self.fixed(k) + pool))
            }
        }
    }

    /// The pools' budget (`pool_blocks`') a plan gives `users` sequences in `kv` bytes, of which `once` the 12-bit pages spend once (`kv12::once_bytes`): what their
    /// states, marks and each group's block 0 leave.
    pub(crate) fn pool_budget(&self, users: usize, kv: u64, once: u64) -> usize {
        kv.saturating_sub(self.fixed(users) + once) as usize
    }

    /// The sequences a plan of `kv` bytes holds where each has rings of its own: as many as it has the bytes of, at most `users`, at least one.
    pub(crate) fn own_seqs(&self, users: usize, kv: u64) -> usize {
        ((kv / self.fixed(1).max(1)) as usize).clamp(1, users.max(1))
    }
}

/// The rings and the states the layers of stage `st` of `d` read or write (what a stage's engine holds of a sequence: `Dyn::build`).
pub(crate) fn stage_holds(d: &super::ModelDesc, st: &super::Stage) -> (std::collections::BTreeSet<usize>, std::collections::BTreeSet<usize>) {
    let ops = || d.layers[st.layers.0..st.layers.1.min(d.layers.len())].iter().flatten();
    (ops().flat_map(Op::rings).collect(), ops().flat_map(Op::states).collect())
}

/// One sequence's rings and states on the executor at `ctx` tokens and prompt chunks of `chunk` (`SeqRoom::rings`, `state`), each counted with the first layer of
/// `d` whose operations read or write it (one no layer names, with the first): what the layers of a stage hold of a sequence (`Dyn::build`).
pub(crate) fn layer_rooms(d: &super::ModelDesc, ctx: usize, chunk: usize) -> Vec<u64> {
    let (ctx, chunk) = (ctx.div_ceil(16) * 16, chunk.max(1));
    let ring = |r: usize| {
        let x = &d.rings[r];
        (x.rows(ctx, chunk) * x.nk * x.d * 2 * (1 + x.values as usize)) as u64
    };
    let state = |s: usize| d.states[s].max(4) as u64;
    let mut out = vec![0u64; d.layers.len().max(1)];
    let (mut ring_seen, mut state_seen) = (vec![false; d.rings.len()], vec![false; d.states.len()]);
    for (l, ops) in d.layers.iter().enumerate() {
        for op in ops {
            for r in op.rings() {
                if r < d.rings.len() && !std::mem::replace(&mut ring_seen[r], true) {
                    out[l] += ring(r);
                }
            }
            for s in op.states() {
                if s < d.states.len() && !std::mem::replace(&mut state_seen[s], true) {
                    out[l] += state(s);
                }
            }
        }
    }
    out[0] += (0..d.rings.len()).filter(|&r| !ring_seen[r]).map(ring).sum::<u64>() + (0..d.states.len()).filter(|&s| !state_seen[s]).map(state).sum::<u64>();
    out
}

/// Whether an operation on a sequence's cache or states takes its rows at an offset in a pass over several sequences (`Dyn::rows_before`, `Dyn::cache`, `Dyn::state`): the attention
/// layer kind, the gated delta rule, Kimi delta attention, Mamba-2, DeepSeek-V4's compressed attention.
fn by_sequence(op: &Op) -> bool {
    matches!(
        op,
        Op::Attn(_)
            | Op::GdnConv { .. }
            | Op::GdnDelta { .. }
            | Op::KdaConv { .. }
            | Op::KdaDelta { .. }
            | Op::MambaConv { .. }
            | Op::MambaScan { .. }
            | Op::CompKv { .. }
            | Op::Compress { .. }
            | Op::IdxKeys { .. }
            | Op::IdxPick(_)
            | Op::CompAttn { .. }
            | Op::IdxMean { .. }
            | Op::NgramRows(_)
            | Op::NgramAdd(_)
    )
}

/// The first operation of a description that keeps something of a sequence outside the rings and states each sequence holds (`Dyn::add_seqs`, `Dyn::select`
/// swaps those): neither one that acts row by row (it keeps nothing) nor one that takes a sequence's cache or states. None: an engine of the description
/// holds as many sequences as it has room for. A draft model's input (`Op::DraftIn`, the big model's last hidden state) is the one such today.
pub(super) fn one_sequence_op(d: &super::ModelDesc) -> Option<&Op> {
    std::iter::once(&d.first).chain(d.layers.iter().flatten()).find(|op| !row_wise(op) && !by_sequence(op))
}

/// An operation's kind as its error messages name it (`GdnConv`, `Attn`): its variant's name.
pub(super) fn op_name(op: &Op) -> String {
    format!("{op:?}").split(|c: char| !c.is_alphanumeric()).next().unwrap_or_default().to_string()
}

impl Paged {
    /// Sequence `slot`'s blocks in group `g`, as the host holds them.
    pub(super) fn group_table(&self, slot: usize, g: usize) -> &[u32] {
        self.pages.table(slot, g)
    }

    /// Sequence `slot`'s table in ring `ring`'s group as the GPU holds it (what the kernels read; the host's may have moved on since the last step).
    pub(super) fn gpu_table(&self, slot: usize, ring: usize) -> Result<Vec<u32>> {
        let mut t = vec![0i32; self.tlen];
        self.tabs[self.group[ring]].read(slot * self.tlen * 4, &mut t)?;
        Ok(t.into_iter().map(|x| x as u32).collect())
    }
}

impl Dyn {
    /// Bytes before the rows of the sequence an operation runs on now, in a slot of `width` values a row (0 outside a pass over several sequences).
    pub(super) fn rows_before(&self, width: usize) -> u64 {
        self.at.get().map_or(0, |k| (self.parts[k].row0 * width * 2) as u64)
    }

    /// Ring `ring`'s keys and values: the sequence's an operation runs on now in a pass over several, else the selected sequence's.
    pub(super) fn cache(&self, ring: usize) -> (u64, u64) {
        let (ko, vo) = self.ring_off[ring];
        match self.at.get() {
            Some(k) => (self.parts[k].kc + ko, self.parts[k].vc + vo),
            None => (self.kc.ptr + ko, self.vc.ptr + vo),
        }
    }

    /// Ring `ring`'s block table for the sequence an operation runs on now (0: the ring is a room of its own, addressed position modulo its capacity).
    pub(super) fn table(&self, ring: usize) -> u64 {
        self.table_of(self.at.get().map_or(self.seq, |k| self.parts[k].slot), ring)
    }

    /// Ring `ring`'s block table for the sequence in `slot` (0: none).
    pub(super) fn table_of(&self, slot: usize, ring: usize) -> u64 {
        let Some(pg) = &self.paged else { return 0 };
        pg.tabs[pg.group[ring]].ptr + (slot * pg.tlen * 4) as u64
    }

    /// Ring `ring`'s group's tables of the pass's rows that decode (`RowTabs`): the first row's, the entries a row's, and the copies of a row's table (`RowTabs::dup`).
    pub(super) fn row_tab(&self, ring: usize) -> (u64, usize, usize) {
        let pg = self.paged.as_ref().expect("a pass's decode rows are paged");
        let rt = pg.rows.as_ref().expect("a pass's decode rows have their tables");
        let g = pg.group[ring];
        (rt.tab[g].ptr, pg.tlen, rt.dup[g])
    }

    /// The tables of the pass's first `self.lead` rows, each its sequence's (`RowTabs`), from the slots written to `RowTabs::slot` and the rows' positions `pos`: a launch a
    /// group, on the GPU (a graph's replay makes them for the rows it is replayed with).
    pub(super) unsafe fn row_tables(&self, pos: u64) -> Result<()> {
        let pg = self.paged.as_ref().expect("a pass's decode rows are paged");
        let rt = pg.rows.as_ref().expect("a pass's decode rows have their tables");
        for (g, tab) in rt.tab.iter().enumerate().filter(|(g, _)| rt.dup[*g] > 0) {
            self.kern.row_tables(self.dev.stream, pg.tabs[g].ptr, pg.tlen, rt.slot_at(), pos, self.lead.get(), rt.win[g], rt.dup[g], tab.ptr)?;
        }
        Ok(())
    }

    /// A recurrent layer's operation (`rows_state`) on the pass's first `rows` rows, a token of a sequence each: one launch, each row through its own state (`RowTabs::states`),
    /// which is each row's own pass's arithmetic (the kernels are the one-sequence ones' on a token).
    pub(super) unsafe fn state_lead(&self, p: &super::Part, op: &Op, rows: usize) -> Result<()> {
        use super::Slot;
        let rt = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).expect("a pass's decode rows have their tables");
        let (s, ss, ru) = (self.dev.stream, rt.nst as i32, rows as u32);
        let sts = |i: usize| rt.states_at() + 8 * i as u64;
        match *op {
            Op::GdnConv { w, state, cd, row } => {
                self.k35.conv_rows(s, p.slot(Slot::Mix), row as i32, p.small[w].ptr, sts(state), ss, p.slot(Slot::Xc), ru, cd as i32)?;
                self.mark("linear attention: convolution")
            }
            Op::GdnDelta { a_log, dt_bias, state, cd, nk, nv } => {
                let (xc, ba, core) = (p.slot(Slot::Xc), p.slot(Slot::Ba), p.slot(Slot::Core));
                self.k35.delta_rows(
                    s,
                    xc,
                    cd as i32,
                    ba,
                    2 * nv as i32,
                    p.small[a_log].ptr,
                    p.small[dt_bias].ptr,
                    sts(state),
                    ss,
                    core,
                    ru,
                    nk as i32,
                    nv as i32,
                )?;
                self.mark("linear attention: delta rule")
            }
            Op::MambaConv { w, bias, state, cd, row, x_off } => {
                let x = p.slot(Slot::Mix) + (x_off * 2) as u64;
                self.kmamba.conv_rows(s, x, row as i32, p.small[w].ptr, p.small[bias].ptr, sts(state), ss, p.slot(Slot::Xc), ru, cd as i32)?;
                self.mark("Mamba-2: convolution")
            }
            Op::MambaScan { dt_bias, a_log, d, state, cd, row, dt_off, nh, ng } => {
                let (xc, dtr, core) = (p.slot(Slot::Xc), p.slot(Slot::Mix) + (dt_off * 2) as u64, p.slot(Slot::Core));
                let (dtb, al, dd) = (p.small[dt_bias].ptr, p.small[a_log].ptr, p.small[d].ptr);
                self.kmamba.ssm_rows(s, xc, cd as i32, dtr, row as i32, dtb, al, dd, sts(state), ss, core, ru, nh as i32, ng as i32, 0.0)?;
                self.mark("Mamba-2: scan")
            }
            _ => err(format!("{op:?}: no kernel takes a step's rows that decode, each through its own state")),
        }
    }

    /// Whether this description's keys and values can be paged (`pages_all`), with no draft model, the whole model on one GPU.
    fn pageable(&self) -> bool {
        self.draft.is_none() && self.stage.first && self.stage.last && pages_all(&self.big.d) && paging_asked()
    }

    /// The plan the loader took (`take_plan`): the sequences it holds and the bytes it gives their keys, values and states (`LoadPlan::users`, `kv`), which
    /// `serve_rows` holds them in and takes no more than; None: an engine no plan sized (a development check's), whose pools are what the GPU's free memory leaves.
    pub fn take_plan(&mut self, users: usize, kv: u64) {
        self.planned = Some((users.max(1), kv));
    }

    /// The rings' keys and values moved to pools of blocks shared by `users` sequences: as large as the plan says (`take_plan`: its bytes less what the sequences'
    /// states, marks and block 0 take, `SeqRoom::pool_budget`), else as the GPU's free memory leaves (`keep` of it kept); a sliding group no more than every user's window
    /// and a chunk, the full groups the rest. Each ring's room in kc and vc is then its group's blocks; the sequences' tables name every entry block 0 until they grow. A group
    /// is the rings of one window and one ratio: compressed entries' (`Ring::ratio`) are blocks of BLOCK entries, which cover BLOCK `ratio` tokens.
    fn page(&mut self, users: usize) -> Result<()> {
        let rings = &self.big.d.rings;
        let held: Vec<bool> = self.ring_cap.iter().map(|&c| c > 0).collect();
        let (group, windows) = ring_groups(rings, &held);
        let covers: Vec<usize> = windows.iter().map(|w| BLOCK * w.1).collect();
        // a token's bytes in each group (its keys and values, every ring of the group: bf16, or 12-bit pages where the model has tables a ring takes)
        let twelve: Vec<Option<f64>> = (0..rings.len())
            .map(|r| self.kv12.as_ref().filter(|_| self.ring_cap[r] > 0).and_then(|t| super::kvpages::Lossless::token_bytes(&self.lib, &self.big.d, t, r)))
            .collect();
        let mut per = vec![0usize; windows.len()];
        for (r, ring) in rings.iter().enumerate() {
            if self.ring_cap[r] > 0 {
                per[group[r]] += twelve[r].map_or(ring.token_bytes(), |b| b.ceil() as usize);
            }
        }
        let keep: Vec<Option<usize>> = windows.iter().map(|&(w, _)| (w > 0 && w != usize::MAX).then_some(w + self.chunk)).collect();
        let most: Vec<usize> = keep.iter().zip(&covers).map(|(&k, &c)| users * seq_blocks(k, self.ctx, c)).collect();
        // the old rooms given back first
        self.dev.sync()?;
        let graphed = self.graph.take().is_some();
        (self.kc, self.vc) = (Buf::zeroed(4)?, Buf::zeroed(4)?);
        let tab_bytes = users * windows.len() * self.ctx.div_ceil(BLOCK) * 4;
        // (12-bit pages: what their pools spend once whatever the sequences, the planner's `kv12::once_bytes`: the raw slabs one sequence alone may take, every page of it raw
        // (`Lossless::alone_slabs`, which the raw pools hold at the least), a prompt chunk's pages unpacked to bf16 (keys and values; a latent's alone, its row in heads
        // of 128), the escape pool's least)
        let once = match self.kv12.as_ref().filter(|_| twelve.iter().any(Option::is_some)) {
            Some(t) => super::kv12::once_bytes(&super::kvpages::Lossless::taken(&self.lib, &self.big.d, t, &self.ring_cap), self.ctx, self.chunk) as usize,
            None => 0,
        };
        let (free, _) = self.dev.mem_info()?;
        const KEEP: usize = 512 << 20;
        // (and what the vision tower takes for an image of the most pixels, where the model reads images: src/dynamic/images.rs)
        let tower = self.vis.as_ref().map_or(0, |v| v.reserve);
        let room = (free.saturating_sub(KEEP + tab_bytes + once + tower) as f64 * self.share) as usize;
        let budget = match self.planned {
            // (the plan's: its bytes less the sequences' states, their marks and the boundaries' pool (made before the pools) and each group's block 0; the later
            // sequences' states (`add_seqs`) are made after the pools, of what the GPU then has. A GPU that has less free than that refuses the plan, which promised
            // more than this engine holds; `serve --with`'s first model takes its share of it at most)
            Some((_, kv)) => {
                let sr = SeqRoom::of(&self.big.d, self.ctx, self.chunk);
                let b = sr.pool_budget(users, kv, once as u64);
                let later = (users - 1) * self.states.iter().map(|s| s.bytes).sum::<usize>();
                let zero: usize = sr.block.iter().sum::<u64>() as usize;
                if self.share >= 1.0 && b + zero + later > room {
                    return err(format!(
                        "the plan gives the pools of {users} sequences {:.1} MiB, and the GPU has {:.1} MiB free for them beside their states: the plan holds more than the GPU",
                        (b + zero) as f64 / (1u64 << 20) as f64,
                        room.saturating_sub(later) as f64 / (1u64 << 20) as f64
                    ));
                }
                b.min(room.saturating_sub(later + zero))
            }
            None => room,
        };
        // as many tokens' blocks in every group (a block of a group is its `covers` tokens of every layer in it), the sliding groups capped and what they leave to the full ones
        let least: Vec<usize> = keep.iter().zip(&covers).map(|(&k, &c)| seq_blocks(k, self.ctx, c)).collect();
        let mut blocks = pool_blocks(budget, &per, &covers, &keep, &most, &least);
        // (a plan's pools hold one sequence's context in every group: a sliding group's window and a chunk and the block it straddles, a full group's every token. A
        // plan that gives less promised more than its pools hold)
        if let Some((_, kv)) = self.planned {
            let need = |g: usize| if keep[g].is_some() { least[g] } else { self.ctx.div_ceil(covers[g]) };
            if let Some(g) = (0..blocks.len()).find(|&g| blocks[g] < need(g)) {
                return err(format!(
                    "the plan gives the keys, values and states of {users} sequences {:.1} MiB, where their pools hold {} blocks of {} tokens of a group whose one sequence takes {}",
                    kv as f64 / (1u64 << 20) as f64,
                    blocks[g],
                    covers[g],
                    need(g)
                ));
            }
        }
        // (a development build's GLYD_ENGINE_PAGES: at most that many blocks a group, to run the memory out on purpose)
        if let Some(cap) = crate::tune!("GLYD_ENGINE_PAGES").and_then(|v| v.parse::<usize>().ok()) {
            blocks.iter_mut().for_each(|b| *b = (*b).min(cap));
        }
        if blocks.iter().any(|&b| b < 2) {
            return err(format!("paged keys and values: {:.1} GiB of GPU memory free leave too few blocks ({blocks:?})", free as f64 / (1u64 << 30) as f64));
        }
        // each ring's room: its group's blocks (one more: block 0, which every entry no sequence holds names), in bf16 or in 12-bit pages
        self.lossless = None;
        let (mut off, mut off_v, mut ring_off) = (0u64, 0u64, Vec::with_capacity(rings.len()));
        for (r, ring) in rings.iter().enumerate() {
            ring_off.push((off, off_v));
            if self.ring_cap[r] > 0 && twelve[r].is_none() {
                let room = ((blocks[group[r]] + 1) * BLOCK * ring.nk * ring.d * 2) as u64;
                off += room;
                off_v += if ring.values { room } else { 0 };
            }
        }
        (self.kc, self.vc) = (Buf::zeroed(off.max(4) as usize)?, Buf::zeroed(off_v.max(4) as usize)?);
        self.ring_off = ring_off;
        if let Some(t) = self.kv12.as_ref().filter(|_| twelve.iter().any(Option::is_some)) {
            let rows = self.chunk.max(self.split_rows).max(users);
            self.lossless = Some(super::kvpages::Lossless::new(
                &self.lib,
                &self.big.d,
                t,
                &group,
                &blocks,
                &self.ring_cap,
                rows,
                self.split_rows.max(users),
                self.ctx,
                self.chunk,
                self.ldec.as_ref().map(|d| d.splits),
            )?);
        }
        let tlen = self.ctx.div_ceil(BLOCK);
        let tabs = (0..windows.len()).map(|_| Buf::zeroed(users * tlen * 4)).collect::<Result<Vec<_>>>()?;
        let rows = self.row_tabs_for(&group, windows.len(), users, tlen)?;
        let counts: Vec<u32> = blocks.iter().map(|&b| (b + 1) as u32).collect();
        self.paged = Some(Paged { pages: Pages::covering(&counts, &keep, &covers, users, self.ctx), group, tabs, tlen, rows });
        self.rgraphs.clear();
        self.pieces.clear();
        // (sequence 0's graph again, on the pools and its tables)
        if graphed {
            self.capture_graph()?;
        }
        Ok(())
    }

    /// The tables of a pass's decode rows (`RowTabs`) for `users` sequences, the `groups` groups `group` makes of the rings, tables of `tlen` entries: None where an attention layer's decode cannot take them (a layer that reads its picks, a call's sequences a row no group's copies divide).
    fn row_tabs_for(&self, group: &[usize], groups: usize, users: usize, tlen: usize) -> Result<Option<RowTabs>> {
        let d = &self.big.d;
        let attns: Vec<&super::AttnOp> = d.layers.iter().flatten().filter_map(|op| if let Op::Attn(a) = op { Some(a) } else { None }).collect();
        if attns.is_empty() || attns.iter().any(|a| a.picked) {
            return Ok(None);
        }
        // (a layer's calls of the library's decode: its query heads in dparts() sequences a row, over a bf16 cache and 12-bit pages alike)
        let sp = |a: &super::AttnOp| if self.ldec.is_some() && a.lib_decode() { a.dparts() } else { 1 };
        let pages12 = self.lossless.iter().flat_map(|lx| lx.rings.iter().enumerate()).filter_map(|(r, rp)| rp.as_ref().map(|rp| (group[r], rp.parts)));
        let dup = row_dups(groups, attns.iter().map(|a| (group[a.ring], sp(a))).chain(pages12));
        // (a group's tables are copied from the first block its widest reader reads: none sliding, from the first)
        let mut win: Vec<Option<usize>> = vec![None; groups];
        let readers = attns.iter().map(|a| (group[a.ring], a.window));
        let readers12 = self.lossless.iter().flat_map(|lx| lx.rings.iter().enumerate()).filter_map(|(r, rp)| rp.as_ref().map(|rp| (group[r], rp.shape.window)));
        for (g, w) in readers.chain(readers12) {
            win[g] = Some(match (win[g], w) {
                (Some(0), _) | (_, 0) => 0,
                (Some(v), w) => v.max(w),
                (None, w) => w,
            });
        }
        let win: Vec<usize> = win.into_iter().map(|w| w.unwrap_or(0)).collect();
        let tab = dup.iter().map(|&k| Buf::zeroed((users * k * tlen * 4).max(4))).collect::<Result<Vec<_>>>()?;
        // (a step of such rows alone in a graph: `rows_graphable`; each row's recurrent states through the table of their addresses)
        let graphs = self.rgraph_on && rows_graphable(std::iter::once(&d.first).chain(d.layers.iter().flatten()));
        let (nst, pcap) = (self.states.len(), self.chunk);
        let meta = Buf::zeroed(meta_states(pcap, users) + 8 * users * nst.max(1))?;
        Ok(Some(RowTabs { tab, dup, win, meta, pcap, nst, rows: users, graphs }))
    }

    /// The entries of sequence `slot`'s tables changed since they were last copied, copied to the GPU (nothing runs on them then: the steps before have been waited for);
    /// the next step after the copies out that still read the blocks taken (`fence_wait`).
    pub(super) fn upload_tables(&mut self, slot: usize) -> Result<()> {
        self.fence_wait()?;
        let Some(pg) = &mut self.paged else { return Ok(()) };
        for g in 0..pg.pages.groups() {
            if let Some((lo, hi)) = pg.pages.take_dirty(slot, g) {
                let v: Vec<i32> = pg.pages.table(slot, g)[lo..hi].iter().map(|&b| b as i32).collect();
                pg.tabs[g].write((slot * pg.tlen + lo) * 4, &v)?;
            }
        }
        Ok(())
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
        self.draft.is_none() && self.stage.first && self.stage.last && one_sequence_op(&self.big.d).is_none()
    }

    /// Whether a step of several rows gives each row the sums a one-row step does: not where the description has experts (a product over the experts' rows groups them by the step's rows; the GPU job's
    /// usercheck found 3 of 4 Muse Spark replies the same). With GLYD_ENGINE_BATCH_PREFILL=0 such a description's decode steps run one sequence each, so every reply is the reply alone.
    pub(crate) fn rows_exact(&self) -> bool {
        let d = &self.big.d;
        !d.layers.iter().flatten().any(|op| matches!(op, Op::Moe(_)))
    }

    /// The least window of the layers whose attention slides (none: every layer reads the whole sequence).
    #[cfg(feature = "dev")]
    pub(crate) fn least_window(&self) -> Option<usize> {
        self.big.d.rings.iter().map(|r| r.window).filter(|&w| w > 0 && w != usize::MAX).min()
    }

    /// The tokens a start found by name is a whole number of: the blocks of every group whole (BLOCK, or a compressed group's BLOCK `ratio`).
    #[cfg(feature = "dev")]
    pub(crate) fn whole_start(&self) -> usize {
        let gcd = |mut a: usize, mut b: usize| {
            while b > 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        self.big.d.rings.iter().map(|r| BLOCK * r.ratio.max(1)).fold(BLOCK, |a, b| a / gcd(a, b) * b)
    }

    /// Whether a kept sequence can be cut back before its end (`ModelStep::resume`): no recurrent state, which is what it is after its last token. (A latent cache is
    /// a ring as any keys are.)
    pub(crate) fn cuts(&self) -> bool {
        self.states.is_empty()
    }

    /// Sequence `slot`'s tokens in its cache.
    pub(super) fn seq_n(&self, slot: usize) -> usize {
        if slot == self.seq {
            self.n
        } else {
            self.seqs[slot].as_ref().map_or(0, |s| s.n)
        }
    }

    pub(super) fn set_seq_n(&mut self, slot: usize, n: usize) {
        if slot == self.seq {
            self.n = n;
        } else if let Some(s) = self.seqs[slot].as_mut() {
            s.n = n;
        }
    }

    /// Sequence `slot`'s caches and states (none of them copied).
    pub(super) fn part_of(&self, slot: usize, row0: usize, rows: usize, start: usize) -> Result<Part> {
        let (kc, vc, states) = if slot == self.seq {
            (self.kc.ptr, self.vc.ptr, self.states.iter().map(|b| b.ptr).collect())
        } else if self.paged.is_some() {
            // (paged: every sequence's blocks are in the pools; its tables say which)
            let s = self
                .seqs
                .get(slot)
                .and_then(Option::as_ref)
                .ok_or_else(|| crate::ffi::Error(format!("sequence {slot}: this engine holds {}", self.seqs.len())))?;
            (self.kc.ptr, self.vc.ptr, s.states.iter().map(|b| b.ptr).collect())
        } else {
            let s = self
                .seqs
                .get(slot)
                .and_then(Option::as_ref)
                .ok_or_else(|| crate::ffi::Error(format!("sequence {slot}: this engine holds {}", self.seqs.len())))?;
            (s.kc.ptr, s.vc.ptr, s.states.iter().map(|b| b.ptr).collect())
        };
        Ok(Part { slot, row0, rows, start, kc, vc, states })
    }

    /// Sequence `slot` empty again: no tokens, its states zeroed (its cache is overwritten as it runs). A slot swapped out: after the copy that reads its states.
    pub(super) fn empty(&mut self, slot: usize) -> Result<()> {
        self.unsettled_wait(slot)?;
        if let Some(m) = self.snap_n.get_mut(slot) {
            *m = 0;
        }
        if slot == self.seq {
            self.n = 0;
            self.pending = None;
            for b in &self.states {
                b.fill_async(0, self.dev.stream)?;
            }
        } else if let Some(s) = self.seqs.get_mut(slot).and_then(Option::as_mut) {
            s.n = 0;
            s.pending = None;
            for b in &s.states {
                b.fill_async(0, self.dev.stream)?;
            }
        }
        Ok(())
    }

    /// Sequence `slot`'s prompt chunk `tokens` (at its place in the sequence), as `prefill` runs one chunk: the head on its last row where `last`.
    fn chunk_alone(&mut self, slot: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        self.settle_slot(slot)?;
        self.select(slot)?;
        let n = tokens.len();
        if n == 0 || n > self.chunk || self.n + n + 1 > self.ctx {
            return err(format!("a chunk of {n} tokens after {}, where a chunk holds {} and the cache {}", self.n, self.chunk, self.ctx));
        }
        let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        self.rope_write(&pos.iter().map(|&p| (slot, p as i64)).collect::<Vec<_>>())?;
        let pp = self.pos.ptr;
        self.start.set(self.n);
        // SAFETY: live buffers of the engine, sized for a chunk.
        unsafe { self.embed(tokens)? };
        self.stage_images(&[(slot, 0, self.n, n)])?;
        // SAFETY: as above.
        unsafe {
            self.pieces_pass(n, (pp, pp), last as usize)?;
            // (a draft's part of the chunk, as `prefill` hands it: a draft model's first input, a draft head's rows)
            self.draft_after_chunk(self.n, tokens, last)?;
        }
        self.n += n;
        self.prof_pass(0, n)?;
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
        let (mut spans, mut ropes) = (Vec::new(), Vec::new());
        for &(slot, t, last) in jobs {
            self.settle_slot(slot)?;
            let start = self.seq_n(slot);
            if t.is_empty() || start + t.len() + 1 > self.ctx {
                return err(format!("{} tokens after {start}, where the cache holds {}", t.len(), self.ctx));
            }
            parts.push(self.part_of(slot, tokens.len(), t.len(), start)?);
            spans.push((slot, tokens.len(), start, t.len()));
            ropes.extend((0..t.len()).map(|i| (slot, (start + i) as i64)));
            pos.extend((0..t.len()).map(|i| (start + i) as i32));
            tokens.extend_from_slice(t);
            if last {
                head_rows.push(tokens.len() - 1);
            }
        }
        // (the leading rows that decode, a token of a sequence each: their attention one launch, or one call of the library's decode, a layer, each row through its own table
        // (`RowTabs`); a step of nothing else, padded to a graph's rows, that graph's replay (`rows_bucket`): its padding rows at position -1, which write and read no key)
        let room = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).map_or(0, |rt| rt.rows);
        let lead = jobs.iter().take_while(|j| j.1.len() == 1).count().min(room);
        let graph = if lead == jobs.len() && asked == lead { self.rows_bucket(n) } else { None };
        let rows = graph.unwrap_or(n);
        if let Some(k) = graph {
            pos.resize(k, -1);
            ropes.resize(k, (0, -1));
            tokens.resize(k, tokens[0]);
            head_rows = (0..k).collect();
        }
        self.dev.sync()?;
        self.rope_write(&ropes)?;
        // (the step's metadata in one copy where the rows' tables are (`RowTabs::meta`): the positions; the slots of the rows that decode, a graph's padding rows -1, the rows
        // past the lead, a prompt's, with no entry; each such row's recurrent states' addresses, a padding row's 0)
        let pp = match self.paged.as_ref().and_then(|pg| pg.rows.as_ref()) {
            Some(rt) => {
                let height = if graph.is_some() { rows } else { lead };
                let mut slots: Vec<i32> = jobs[..lead].iter().map(|j| j.0 as i32).collect();
                slots.resize(height, -1);
                let lead_states: Vec<&[u64]> = parts[..lead].iter().map(|p| &p.states[..]).collect();
                let states = if rt.nst > 0 && height > 0 { state_table(&lead_states, height, rt.nst) } else { Vec::new() };
                rt.write(&pos, &slots, &states)?;
                rt.pos_at()
            }
            None => {
                self.pos.write(0, &pos)?;
                self.pos.ptr
            }
        };
        (self.parts, self.head_rows) = (parts, head_rows);
        self.lead.set(if graph.is_some() { rows } else { lead });
        // (a pass that is no whole graph's in pieces (src/dynamic/pieces.rs): one that holds a prompt's tokens, or rows that decode of a description that takes no whole graph of
        // them; such rows of one that does run eager where its graph is not taken (GLYD_ENGINE_ROWS_GRAPH=0))
        let whole = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).is_some_and(|rt| rt.graphs);
        let pieces = !whole || jobs.iter().any(|j| j.1.len() > 1 || !j.2);
        // SAFETY: live buffers of the engine and of each sequence, the pass's rows within a chunk's (a graph's padded rows within the head's).
        let r = unsafe { self.embed(&tokens) }.and_then(|_| self.stage_images(&spans)).and_then(|_| unsafe {
            match graph {
                Some(k) => self.rows_replay(k, false),
                None if pieces => self.pieces_pass(n, (pp, pp), asked),
                None => self.forward(&self.big, n, (pp, pp), asked),
            }
        });
        self.parts.clear();
        self.head_rows.clear();
        self.lead.set(0);
        r?;
        for &(slot, t, _) in jobs {
            let to = self.seq_n(slot) + t.len();
            self.set_seq_n(slot, to);
        }
        let (heads, _) = self.head_out(asked, false)?;
        // (GLYD_PROF=1: the rows that decode, a token each with its sample, and the prompts' tokens)
        let rows = jobs.iter().filter(|j| j.1.len() == 1 && j.2).count();
        self.prof_pass(rows, n - rows)?;
        let mut heads = heads.into_iter();
        Ok(jobs.iter().map(|j| j.2.then(|| heads.next().map(|(token, logit)| Sample { token, logit, logprob: None })).flatten()).collect())
    }

    /// The rows of the CUDA graph a step of `n` rows that decode (and nothing else) replays (vLLM's full graphs of uniform decode batches at a ladder of
    /// padded sizes): `bucket`'s ladder up to the head's rows, where the description takes such a graph (`RowTabs::graphs`) and the engine captures graphs at all; None: the
    /// step runs eager. GLYD_ENGINE_ROWS_GRAPH=0 (a development build's) runs every such step eager, to compare.
    fn rows_bucket(&self, n: usize) -> Option<usize> {
        let rt = self.paged.as_ref()?.rows.as_ref().filter(|rt| rt.graphs)?;
        // (a traced pass runs eager: no graph holds a layer's end)
        if self.graphs_off.get()
            || super::tp::comm().is_some()
            || crate::hb_usage::recording()
            || self.tracing()
            || crate::tune!("GLYD_ENGINE_ROWS_GRAPH").is_some_and(|v| v == "0")
        {
            return None;
        }
        bucket(n, self.head_room().min(rt.rows).min(self.chunk))
    }

    /// Whether steps of several rows that decode replay CUDA graphs (`rows_bucket`).
    #[cfg(feature = "dev")]
    pub(crate) fn rows_graphs(&self) -> bool {
        self.rows_bucket(2).is_some()
    }

    /// The graphs of steps of rows that decode, every size of `rows_bucket`'s ladder, captured before the first request (as vLLM captures its at start), so that no
    /// step waits on a capture: each after a warm-up pass of padding rows alone (position -1: no key read or written, no sequence's state touched).
    fn capture_rows(&mut self) -> Result<()> {
        if self.rows_bucket(2).is_none() {
            return Ok(());
        }
        let most = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).map_or(0, |rt| rt.rows).min(self.head_room()).min(self.chunk);
        let mut sizes: Vec<usize> = (2..=most).filter_map(|n| bucket(n, most)).collect();
        sizes.dedup();
        for k in sizes {
            if self.rgraphs.iter().any(|g| g.rows == k) {
                continue;
            }
            self.dev.sync()?;
            // (positions and slots -1, no row's state: no key read or written, the recurrent layers' kernels skip every row)
            let rt = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).expect("room for the rows' tables");
            rt.write(&vec![-1i32; k], &vec![-1i32; k], &vec![0u64; k * rt.nst])?;
            self.rope_write(&vec![(0, -1); k])?;
            // (a part of no row of its own: the pass is one over several sequences, its rows all the lead's)
            (self.parts, self.head_rows) = (vec![self.part_of(self.seq, 0, 1, 0)?], (0..k).collect());
            self.lead.set(k);
            // SAFETY: the engine's buffers, sized for the head's rows (k at most); the padding rows write no cache.
            let r = unsafe { self.embed(&vec![0u32; k]).and_then(|_| self.rows_replay(k, true)) };
            self.parts.clear();
            self.head_rows.clear();
            self.lead.set(0);
            r?;
        }
        // (--prof: the warm-ups' events are no request's)
        self.prof_take()?;
        self.dev.sync()
    }

    /// A step of `k` rows that decode (padded: `pass`), its embedding queued: the graph of k rows replayed; where there is none yet, the step run eager and then captured
    /// (its first run is the warm-up a capture needs: every buffer and kernel it takes made). The pass's rows' positions, slots and tokens are in the buffers the graph reads.
    /// `start`: the capture is `capture_rows`' (else one in a request's step, counted: `rows_late`).
    unsafe fn rows_replay(&mut self, k: usize, start: bool) -> Result<()> {
        if let Some(i) = self.rgraphs.iter().position(|g| g.rows == k) {
            self.rgraphs[i].graph.launch(self.dev.stream)?;
            self.replayed.set(true);
            self.rplayed.set(Some(i));
            self.dcalls.set(self.rgraphs[i].calls);
            self.rows_replays.set(self.rows_replays.get() + 1);
            return Ok(());
        }
        if !start {
            self.rows_late.set(self.rows_late.get() + 1);
        }
        // (the positions in the step's metadata, which every replay reads: `RowTabs::meta`)
        let pp = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).map_or(self.pos.ptr, |rt| rt.pos_at());
        self.forward(&self.big, k, (pp, pp), k)?;
        // (--prof: the eager pass's events kept for its profile, the capture's kept with the graph)
        let eager = self.prof.as_ref().map(|p| std::mem::take(&mut *p.borrow_mut()));
        let queued = self.queued.get();
        self.capturing.set(true);
        let g = self.dev.capture(|| {
            // SAFETY: as the eager pass above, on the same buffers.
            unsafe { self.forward(&self.big, k, (pp, pp), k) }
        });
        self.capturing.set(false);
        self.queued.set(queued);
        let marks = match (&self.prof, eager) {
            (Some(p), Some(e)) => std::mem::replace(&mut *p.borrow_mut(), e),
            _ => Vec::new(),
        };
        match g {
            Ok(graph) => self.rgraphs.push(super::RowsGraph { rows: k, graph, marks, calls: self.dcalls.get() }),
            Err(e) => {
                // (the step ran; a capture that fails leaves such steps eager from here, as they were)
                eprintln!("glyd: a decode step of {k} rows could not be captured as a CUDA graph ({e}): such steps run eager");
                if let Some(rt) = self.paged.as_mut().and_then(|pg| pg.rows.as_mut()) {
                    rt.graphs = false;
                }
            }
        }
        Ok(())
    }

    /// Ready to be a scheduler's step for up to `users` sequences: the head's room for a row each, and as many more sequences as the GPU's free memory holds (the first is this one's). None where this
    /// engine runs one sequence only (a draft model, a stage of a split model, a description `add_seqs` refuses).
    pub fn serve_rows(&mut self, users: usize) -> Result<Option<usize>> {
        if self.draft.is_some() && self.stage.first && self.stage.last {
            // (a drafted model: its one sequence, each decode step a drafting round, `ModelStep::draft`)
            self.reset()?;
            self.make_lanes()?;
            return Ok(Some(1));
        }
        // (a stage of a model across GPUs: the run of the stages is the step, src/dynamic/split.rs)
        if self.draft.is_some() || !self.stage.first || !self.stage.last {
            return Ok(None);
        }
        if one_sequence_op(&self.big.d).is_some() {
            // (an operation that keeps a sequence's data outside the rings and states each sequence holds: one sequence, the scheduler's step of one slot)
            self.reset()?;
            self.make_lanes()?;
            return Ok(Some(1));
        }
        if self.seqs.len() > 1 {
            // (made ready already)
            return Ok(Some(self.seqs.len()));
        }
        // (a step's sequences begin empty: what a reply before this left in the one sequence is not a scheduler's, which would find its first slot holding tokens)
        self.reset()?;
        // (no more than the plan's sequences, where a plan sized this engine: `take_plan`)
        let users = self.planned.map_or(users, |p| users.min(p.0)).clamp(1, self.chunk.min(64));
        if users > 1 {
            self.set_head_rows(users)?;
        }
        // (a model with recurrent states and no latent cache keeps each sequence's states at its prompt's end too: a second copy of them a sequence)
        let marks = ModelStep::keeps(self) && self.can_mark();
        let states: usize = self.states.iter().map(|b| b.bytes).sum();
        if self.pageable() {
            // (keys and values in pools of blocks: a user takes the blocks its tokens fill, so the users asked for all have a sequence, and the memory decides how many tokens they hold together;
            // the marks' copies first, so the pools are sized by what is left)
            if marks {
                self.mark_room(users)?;
                self.snap_room()?;
            }
            self.page(users)?;
            self.add_seqs(users - 1)?;
            // (the host's swap area for the sequences sent back for room: as the plan sizes it, from what the computer has free now that the experts' host tier is in it)
            self.swap_room(crate::plan::swap_area(self.swap_backs() as u64, crate::offload::host_memory()) as usize)?;
            self.make_lanes()?;
            self.capture_rows()?;
            self.capture_pieces()?;
            return Ok(Some(users));
        }
        let per = (self.kc.bytes + self.vc.bytes + states * (1 + marks as usize)).max(1);
        let (free, _) = self.dev.mem_info()?;
        // (room kept for the products' workspaces and the driver)
        const KEEP: usize = 512 << 20;
        let fit = ((free.saturating_sub(KEEP + states * marks as usize) as f64 * self.share) as usize / per).min(users - 1);
        // (the plan's sequences, each its own rings (`SeqRoom::own_seqs`): a GPU that holds fewer refuses the plan, which promised more than this engine holds)
        let more = match self.planned {
            Some((_, kv)) => {
                let want = SeqRoom::of(&self.big.d, self.ctx, self.chunk).own_seqs(users, kv) - 1;
                if want > fit && self.share >= 1.0 {
                    return err(format!("the plan gives {} sequences rings of their own, and the GPU's free memory holds {}", want + 1, fit + 1));
                }
                want.min(fit)
            }
            None => fit,
        };
        if marks {
            self.mark_room(more + 1)?;
        }
        self.add_seqs(more)?;
        self.make_lanes()?;
        self.capture_pieces()?;
        Ok(Some(more + 1))
    }

    /// Room for `slots` sequences' states at their marks.
    fn mark_room(&mut self, slots: usize) -> Result<()> {
        self.snaps = (0..slots).map(|_| self.states.iter().map(|b| Buf::zeroed(b.bytes)).collect::<Result<Vec<_>>>()).collect::<Result<_>>()?;
        self.snap_n = vec![0; slots];
        Ok(())
    }

    /// The pool of states kept at block boundaries (src/dynamic/snaps.rs): sized by `snaps::count`, boundaries every `snaps::every` tokens (or as `ask_state_every` asked).
    fn snap_room(&mut self) -> Result<()> {
        let bytes: Vec<usize> = self.states.iter().map(|b| b.bytes).collect();
        let n = snaps::count(bytes.iter().sum());
        let every = if self.named.every > 0 { self.named.every } else { snaps::every(self.chunk, BLOCK) };
        self.named = if n > 0 { snaps::Snaps::new(n, &bytes, every)? } else { snaps::Snaps::none() };
        Ok(())
    }

    /// States kept at boundaries every `every` tokens (a multiple of a block), where the pool is made (`as_step`): a check's short prompts reach boundaries a chunk apart
    /// would not.
    #[cfg(feature = "dev")]
    pub(crate) fn ask_state_every(&mut self, every: usize) {
        self.named = snaps::Snaps { every: (every / BLOCK).max(1) * BLOCK, ..snaps::Snaps::none() };
    }

    /// The states of the sequence in `slot` kept under `key`, the name of its block `b`, which ends at its last token (a boundary): where that block is the one the name
    /// finds and no copy of this run's is kept yet.
    fn keep_states_at(&mut self, t: usize, b: usize, key: u128) -> Result<()> {
        let pg = self.paged.as_ref().expect("paged");
        let blk = pg.pages.table(t, 0)[b];
        if blk == 0 || pg.pages.named(0, key) != Some(blk) {
            return Ok(());
        }
        let Some(i) = self.named.lru.place(key, blk) else { return Ok(()) };
        let from = self.part_of(t, 0, 0, 0)?.states;
        for (j, &src) in from.iter().enumerate() {
            // SAFETY: the sequence's state and its kept copy's buffer, of the state's bytes.
            unsafe { copy_d2d_async(self.named.bufs[i][j].ptr, src, self.states[j].bytes, self.dev.stream)? };
        }
        Ok(())
    }

    /// Whether the pool of states kept at boundaries is made (boundaries asked for, and copies to keep them in).
    fn pooled(&self) -> bool {
        self.named.every > 0 && !self.named.bufs.is_empty()
    }

    /// The copy of the pool that holds the states after the first `j` blocks named `keys`: kept for the block their name finds now.
    fn kept_copy(&self, keys: &[u128], j: usize) -> Option<usize> {
        let pg = self.paged.as_ref()?;
        let key = *keys.get(j.checked_sub(1)?)?;
        self.named.lru.find(key, pg.pages.named(0, key)?)
    }

    /// Whether a served model's sequences have their states copied aside at a mark (`ModelStep::mark`), for a kept sequence to go on from: recurrent states (a latent cache
    /// is a ring, whose blocks a mark keeps as any keys'; so are DeepSeek-V4's compressed entries, whose states are its compressors' open windows alone).
    pub(crate) fn can_mark(&self) -> bool {
        !self.states.is_empty()
    }

    fn batch_prefill() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_none_or(|v| v != "0"))
    }

    /// A round is one step of the rows that decode and the prompts' chunks (`ModelStep::round`, docs/design/runtime.md section 5), each weight read once for both: a reply does not stop
    /// while a prompt that came in is run. GLYD_ENGINE_ONE_STEP=0: the round of two passes, a prompt pass and then a decode step. The record that set it:
    /// benchmarks/gpu/rtx4080-one-step-round-2026-11-22 (gemma-4-12B, 8 requests coming in: the gap between a reply's tokens, 99th percentile, 282 to 144 ms with 1k prompts and 465 to
    /// 168 ms with 4k, the prompts' first token and the total tokens/s the same or better). What it costs: a row that rides in a prompt's step is computed by the kernel of a step of that
    /// many rows, not its own pass's (the module's note), so a reply under load is not the reply alone to the last bit, as it already is not when more than 16 sequences decode together.
    fn one_step_asked() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_ONE_STEP").is_none_or(|v| v != "0"))
    }

    /// The rows of a step that holds both rows that decode and a prompt's chunk, where GLYD_ENGINE_STEP_TOKENS says: by default a step holds what a pass holds (a chunk: 512 rows for
    /// Gemma 4), which the record above found level with the round of two passes for the prompt's first token and the total. Fewer rows make the gaps between a reply's tokens shorter and
    /// the prompts slower: at 256 the gaps were 82 and 97 ms and the prompts' first token 8% to 10% later, at 64 the first token was 2 to 3 times later and the total a fifth to a third less.
    fn step_tokens() -> Option<usize> {
        static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *N.get_or_init(|| std::env::var("GLYD_ENGINE_STEP_TOKENS").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0))
    }
}

/// The prompt tokens a step takes beside any number of rows, so that a prompt moves while many sequences decode.
const STEP_LEAST: usize = 16;

/// The most rows of a pass whose products give every row the bits of its own pass (the kernels change past it: `glyd-engine roundcheck` measures it on every family).
pub(crate) const EXACT_ROWS: usize = 16;

/// `GLYD_ENGINE_STEP_PROF=1` (a development build's): where a step's time goes, every 256 steps on stderr. The host's time in the step's call (`decode`, `round`; a launched
/// step's `launch` and `finish`) and between one step's call and the next one's (the scheduler, the sampler, the server); the GPU's from the step's first work to its last, and
/// its idle time before the step (from the step before's last work: `Gaps`). With a step that waits for its tokens, the GPU is idle for the host's work between steps.
struct StepProf {
    at: std::time::Instant,
}

#[derive(Default)]
struct StepSums {
    n: usize,
    rows: usize,
    prompt: usize,
    call: f64,
    host: f64,
    hosts: usize,
    gpu: f64,
    idle: f64,
    idles: usize,
    idle_each: Vec<f32>,
    last: Option<std::time::Instant>,
}

thread_local!(static STEP_SUMS: std::cell::RefCell<StepSums> = std::cell::RefCell::new(StepSums::default()));

impl StepProf {
    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::tune!("GLYD_ENGINE_STEP_PROF").is_some_and(|v| v == "1"))
    }

    fn begin() -> Option<StepProf> {
        Self::on().then(|| StepProf { at: std::time::Instant::now() })
    }

    /// The step `p` began has its samples: `call` ms of the host's time in its calls (None: since `begin`), the GPU's times where `Gaps` took them.
    fn end(p: Option<StepProf>, rows: usize, prompt: usize, call: Option<f64>, times: Option<(f64, Option<f64>)>) {
        let Some(p) = p else { return };
        let now = std::time::Instant::now();
        let call = call.unwrap_or_else(|| now.duration_since(p.at).as_secs_f64() * 1e3);
        STEP_SUMS.with(|c| {
            let mut c = c.borrow_mut();
            if let Some(last) = c.last.filter(|&l| l <= p.at) {
                (c.host, c.hosts) = (c.host + p.at.duration_since(last).as_secs_f64() * 1e3, c.hosts + 1);
            }
            c.last = Some(now);
            (c.n, c.rows, c.prompt, c.call) = (c.n + 1, c.rows + rows, c.prompt + prompt, c.call + call);
            if let Some((gpu, idle)) = times {
                c.gpu += gpu;
                if let Some(i) = idle {
                    (c.idle, c.idles) = (c.idle + i, c.idles + 1);
                    c.idle_each.push(i as f32);
                }
            }
            if c.n == 256 {
                let n = c.n as f64;
                eprintln!(
                    "step profile, 256 steps of {:.1} rows and {:.0} prompt tokens on average: the call {:.3} ms (the GPU from its first work to its last {:.3} ms), the host between steps {:.3} ms, the GPU idle before a step {:.3} ms (the median; {:.3} the mean)",
                    c.rows as f64 / n,
                    c.prompt as f64 / n,
                    c.call / n,
                    c.gpu / n,
                    c.host / c.hosts.max(1) as f64,
                    super::prof::median(&c.idle_each),
                    c.idle / c.idles.max(1) as f64
                );
                *c = StepSums { last: c.last, ..Default::default() };
            }
        });
    }
}

impl Dyn {
    /// Err where a later step has been launched over the last step's head rows (their logits are that step's now): a step's rows are read before the next is launched.
    fn rows_readable(&self) -> Result<()> {
        if self.lanes.as_ref().is_some_and(|l| l.stale) {
            return err("a step's logits read after a later step was launched over them");
        }
        Ok(())
    }

    /// A served step's figures: the host's (`StepProf`, its call `call` ms where given) and the GPU's (`Gaps`: the step of slot `k`, or the last begun, done by now; GLYD_PROF=1's too).
    fn step_times(&self, p: Option<StepProf>, rows: usize, prompt: usize, call: Option<f64>) {
        self.step_times_of(p, rows, prompt, call, None)
    }

    fn step_times_of(&self, p: Option<StepProf>, rows: usize, prompt: usize, call: Option<f64>, k: Option<usize>) {
        let times = self.gaps.as_ref().and_then(|g| g.read(k.unwrap_or_else(|| g.last())).ok());
        if let Some((busy, idle)) = times.filter(|_| super::prof::on()) {
            super::prof::gpu_times(rows, prompt, busy, idle);
        }
        StepProf::end(p, rows, prompt, call, times);
    }

    /// A pass begins its first work on the GPU (`Gaps`; not inside a graph's capture): the slot of its times.
    pub(super) fn gap_begin(&self) -> Result<Option<usize>> {
        match &self.gaps {
            Some(g) if !self.capturing.get() => g.begin(self.dev.stream).map(Some),
            _ => Ok(None),
        }
    }

    /// The pass begun last has queued its last work (recorded again where it queues more).
    pub(super) fn gap_end(&self) -> Result<()> {
        match &self.gaps {
            Some(g) if !self.capturing.get() => g.end(g.last(), self.dev.stream),
            _ => Ok(()),
        }
    }

    /// `Lanes` for a served model whose steps can be launched before the step before them is read (`ModelStep::overlaps`: the conditions are there), where the library
    /// looks up its embedding's rows by tokens on the GPU (asked once here, of the embedding's layout); and `Gaps` where a step profile is asked for. Its blocks of experts
    /// are the ones a decode graph captures (src/dynamic/mod.rs): none reads the router's choices on the host, so an expert cache's are its layers planned on the GPU.
    fn make_lanes(&mut self) -> Result<()> {
        if StepProf::on() || super::prof::on() {
            self.gaps = Some(Gaps::new()?);
        }
        let Some(e) = self.big.mats[self.big.d.embed].packed_opt().map(|p| p.pack) else { return Ok(()) };
        if !crate::sched::overlap_forced()
            || !ModelStep::one_step(self)
            || !self.big.moe.iter().all(|b| !b.reads_host() && (b.resident().is_some() || b.graphable()))
            || self.big.ram.is_some()
            || self.reads_host_tables()
            || !self.kern.has_step_ids()
            || self.tok.as_ref().is_some_and(|b| b.bytes < 4 * self.chunk)
        {
            return Ok(());
        }
        let tab_words = self.paged.as_ref().map_or(0, |pg| self.seqs.len() * pg.pages.groups() * self.ctx.div_ceil(BLOCK));
        let status_words = self.lossless.as_ref().map_or(0, |lx| lx.status.len() * super::kvpages::VOID_WORDS);
        let lanes = Lanes::new(self.chunk, self.head_room(), tab_words, status_words)?;
        lanes.ids.write(0, &[0i64])?;
        // SAFETY: the embedding's pack; ids holds one id (0, a row of it), the scratch room a chunk's rows of the hidden width.
        let ok = unsafe { self.lib.rows(&e, lanes.ids.ptr, 1, self.big.slot(super::Slot::X), self.dev.stream) }.is_ok() && self.dev.sync().is_ok();
        if ok {
            self.lanes = Some(lanes);
        }
        Ok(())
    }

    /// The entries of sequence `slot`'s tables changed since they were last copied, copied to the GPU on the stream from the room of the lane the next step takes (`Lanes`): after
    /// the steps queued before it, which read the tables as they were, and before the next step's work; no wait. Without lanes: `upload_tables` (the steps before waited for).
    pub(super) fn upload_tables_queued(&mut self, slot: usize) -> Result<()> {
        // (the copies out that still read the blocks taken: before the step that writes them)
        self.fence_wait()?;
        let s = self.dev.stream;
        let len = self.ctx.div_ceil(BLOCK);
        let (Some(pg), Some(lanes)) = (&mut self.paged, &mut self.lanes) else {
            self.dev.sync()?;
            return self.upload_tables(slot);
        };
        let groups = pg.pages.groups();
        let lane = &mut lanes.lane[lanes.next];
        for g in 0..groups {
            if let Some((lo, hi)) = pg.pages.take_dirty(slot, g) {
                let at = (slot * groups + g) * len;
                let host = lane.tabs.as_slice_mut::<i32>();
                for (i, &b) in pg.pages.table(slot, g)[lo..hi].iter().enumerate() {
                    host[at + lo + i] = b as i32;
                }
                // SAFETY: the table's entries lo..hi of its device buffer (len entries), from the lane's pinned room at the same entries, which is not written again until
                // this lane's next step is launched three steps on (its copies have run by then).
                unsafe {
                    crate::dev::copy_h2d_async(
                        pg.tabs[g].ptr + ((slot * pg.tlen + lo) * 4) as u64,
                        (lane.tabs.ptr as *const i32).add(at + lo) as *const _,
                        (hi - lo) * 4,
                        s,
                    )?
                };
            }
        }
        Ok(())
    }

    /// `ModelStep::launch` on `Lanes`: the step's work queued on the stream with no wait, its inputs copied from the next lane's pinned room (positions, and the tokens' sources,
    /// which `step_ids` makes the embedding's ids of, a row's token the step before chose read from that step's head), its samples copied back into the lane, and the lane's
    /// event recorded after them. One decode row alone is its sequence's graph's step where it has one (the decode step a reply alone takes); else one pass of every row and
    /// chunk, as `pass` makes it. The sequences' places move on now, as `pass` moves them.
    fn queue_step(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<crate::sched::Launched> {
        let called = std::time::Instant::now();
        for (slot, at) in rows.iter().map(|r| (r.pages[0] as usize, r.pos)).chain(jobs.iter().map(|j| (j.pages[0] as usize, j.start))) {
            if self.seq_n(slot) != at {
                return err(format!("tokens at {at} of sequence {slot}, which holds {}", self.seq_n(slot)));
            }
        }
        let n = rows.len() + jobs.iter().map(|j| j.tokens.len()).sum::<usize>();
        let asked = rows.len() + jobs.iter().filter(|j| j.last).count();
        if n == 0 || n > self.chunk || asked > self.head_room() || jobs.iter().any(|j| j.tokens.is_empty()) {
            return err(format!("a step of {n} rows and {asked} samples, where a pass holds {} and the head {}", self.chunk, self.head_room()));
        }
        let (mut pos, mut src) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut ropes: Vec<(usize, i64)> = rows.iter().map(|r| (r.pages[0] as usize, r.pos as i64)).collect();
        let mut spans = Vec::with_capacity(jobs.len());
        for j in jobs {
            spans.push((j.pages[0] as usize, ropes.len(), j.start, j.tokens.len()));
            ropes.extend((0..j.tokens.len()).map(|i| (j.pages[0] as usize, (j.start + i) as i64)));
        }
        for r in rows {
            pos.push(r.pos as i32);
            src.push(match r.after {
                Some(h) if h < self.head_room() => -1 - h as i32,
                Some(h) => return err(format!("a row after head row {h} of a head of {}", self.head_room())),
                None => r.token as i32,
            });
        }
        for j in jobs {
            pos.extend((0..j.tokens.len()).map(|i| (j.start + i) as i32));
            src.extend(j.tokens.iter().map(|&t| t as i32));
        }
        if let Some(&t) = src.iter().find(|&&t| t >= 0 && t as usize >= self.big.d.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.big.d.vocab));
        }
        if pos.iter().any(|&p| p as usize + 1 > self.ctx) {
            return err(format!("a step past the cache's {} tokens", self.ctx));
        }
        let one = rows.len() == 1 && jobs.is_empty();
        if one {
            // (its graph and its own buffers selected: no wait where it is selected already, the sequences' steps before it waited for where not)
            self.select(rows[0].pages[0] as usize)?;
        }
        let graph = one && self.graph.is_some() && !self.verifying.get();
        let (s, chunk) = (self.dev.stream, self.chunk);
        let lanes = self.lanes.as_mut().expect("lanes");
        if lanes.out >= LANES - 1 {
            return err("a step launched while two are not finished");
        }
        let k = lanes.next;
        let (ids, src_dev, ins) = (lanes.ids.ptr, lanes.src.ptr, lanes.lane[k].ins.ptr as *mut i32);
        // SAFETY: the lane's pinned room holds a chunk's positions, then a chunk's sources (n <= chunk); it was last read by copies three steps ago, finished since.
        unsafe {
            std::ptr::copy_nonoverlapping(pos.as_ptr(), ins, n);
            std::ptr::copy_nonoverlapping(src.as_ptr(), ins.add(chunk), n);
        }
        let gap = self.gap_begin()?;
        let e = self.big.mats[self.big.d.embed].packed_opt().map(|p| p.pack).ok_or_else(|| crate::ffi::Error("the embedding is not packed".into()))?;
        let pp = if graph { self.st.ptr } else { self.pos.ptr };
        // SAFETY: the engine's live buffers: pos (or st) and the ids' room hold a chunk's rows, the head's tok its rows (each source below the head's rows, checked
        // above); the copies are ordered on the stream before the work that reads them and after the steps before (whose head this step's ids read).
        unsafe {
            crate::dev::copy_h2d_async(pp, ins as *const _, 4 * n, s)?;
            crate::dev::copy_h2d_async(src_dev, ins.add(chunk) as *const _, 4 * n, s)?;
            Dyn::mark(self, "start")?;
            self.kern.step_ids(s, src_dev, self.big.tok.ptr, ids, self.tok.as_ref().map_or(0, |b| b.ptr), n)?;
            self.lib.rows(&e, ids, n as i64, self.big.slot(super::Slot::X), s)?;
        }
        self.rope_write(&ropes)?;
        self.stage_images(&spans)?;
        let mut lasts = Vec::with_capacity(jobs.len());
        if graph {
            // SAFETY: the graph was captured on these buffers (the selected sequence's).
            unsafe { self.graph.as_ref().expect("a graph").launch(s)? };
            self.replayed.set(true);
            let slot = rows[0].pages[0] as usize;
            self.set_seq_n(slot, rows[0].pos + 1);
        } else {
            let (mut parts, mut head_rows, mut row0) = (Vec::new(), Vec::new(), 0);
            for r in rows {
                parts.push(self.part_of(r.pages[0] as usize, row0, 1, r.pos)?);
                head_rows.push(row0);
                row0 += 1;
            }
            for j in jobs {
                parts.push(self.part_of(j.pages[0] as usize, row0, j.tokens.len(), j.start)?);
                row0 += j.tokens.len();
                if j.last {
                    head_rows.push(row0 - 1);
                }
                lasts.push(j.last);
            }
            (self.parts, self.head_rows) = (parts, head_rows);
            // SAFETY: live buffers of the engine and of each sequence, the pass's rows within a chunk's (checked above).
            let r = unsafe { self.forward(&self.big, n, (pp, pp), asked) };
            self.parts.clear();
            self.head_rows.clear();
            r?;
            for r in rows {
                self.set_seq_n(r.pages[0] as usize, r.pos + 1);
            }
            for j in jobs {
                self.set_seq_n(j.pages[0] as usize, j.start + j.tokens.len());
            }
        }
        if graph {
            lasts.clear();
        }
        // (what the step's growths promised of the 12-bit pools: in their counts once the stream is waited for, as `kv12_room` does before it reads them)
        self.kv12_stepped();
        let head = self.head_room();
        let evs = self.prof.as_ref().map(|p| std::mem::take(&mut *p.borrow_mut())).unwrap_or_default();
        let lanes = self.lanes.as_mut().expect("lanes");
        let lane = &mut lanes.lane[k];
        // SAFETY: the head's tok and val hold `asked` rows (written by this step's head); the lane's pinned room holds `head` of each, read after its event.
        unsafe {
            crate::dev::copy_d2h_async(lane.outs.ptr, self.big.tok.ptr, 4 * asked, s)?;
            crate::dev::copy_d2h_async((lane.outs.ptr as *mut u8).add(4 * head) as *mut _, self.big.val.ptr, 4 * asked, s)?;
        }
        // (a step whose rows draw from their candidates: those queued with its work, and the sums of exponentials its rows ask for, read once its event is waited for; the
        // top-k kernel takes the vocabulary)
        let v = self.big.d.vocab;
        let asks = std::mem::take(&mut self.draws);
        let drawn = asks.iter().any(Option::is_some) && !self.top.off && v.is_multiple_of(8) && v >= crate::kern::TOPK;
        if drawn {
            self.queue_cands(asked)?;
            if asks.iter().flatten().any(|&t| t > 0.0) && self.sums.as_ref().is_none_or(|b| b.bytes < 8 * head) {
                self.sums = Some(Buf::zeroed(8 * head)?);
            }
            let out = self.sums.as_ref().map_or(0, |b| b.ptr);
            for (r, &t) in asks.iter().enumerate().take(asked) {
                if let Some(t) = t.filter(|&t| t > 0.0) {
                    // SAFETY: head row r of v logits and its largest (f32) in val; its f64 sum at row r of the sums' room (a head's rows).
                    unsafe {
                        self.kern.exp_sum_rows(
                            s,
                            self.big.logits.ptr + (r * v * 2) as u64,
                            v as i32,
                            1,
                            self.big.val.ptr + (4 * r) as u64,
                            t,
                            out + (8 * r) as u64,
                        )?
                    };
                }
            }
        }
        let lanes = self.lanes.as_mut().expect("lanes");
        let lane = &mut lanes.lane[k];
        if drawn && asks.iter().flatten().any(|&t| t > 0.0) {
            if let Some(b) = &self.sums {
                // SAFETY: the sums' room holds `asked` f64 (the head's rows); the lane's pinned room `head` of them, read after its event.
                unsafe { crate::dev::copy_d2h_async(lane.sums.ptr, b.ptr, 8 * asked, s)? };
            }
        }
        lane.asks = if drawn { asks } else { Vec::new() };
        if let Some(lx) = &self.lossless {
            // (the 12-bit pages' status after the step: its void ring, read when it lands, not as the next step writes it)
            for (g, b) in lx.status.iter().enumerate() {
                let w = super::kvpages::VOID_WORDS;
                // SAFETY: the group's status holds at least VOID_WORDS words; the lane's pinned room VOID_WORDS a group, read after its event.
                unsafe { crate::dev::copy_d2h_async((lane.status.ptr as *mut u32).add(g * w) as *mut _, b.ptr, 4 * w, s)? };
            }
        }
        if let (Some(g), Some(k)) = (&self.gaps, gap) {
            g.end(k, s)?;
        }
        lane.done.record(s)?;
        lane.drawn = drawn;
        (lane.rows, lane.lasts, lane.prompt, lane.graph, lane.evs) = (rows.len(), lasts, n - rows.len(), graph, evs);
        lane.queue_ms = called.elapsed().as_secs_f64() * 1e3;
        lanes.gap[k] = gap;
        lanes.next = (k + 1) % LANES;
        lanes.out += 1;
        lanes.stale = true;
        Ok(crate::sched::Launched { done: None, tag: k })
    }

    /// `ModelStep::finish` of a step `queue_step` launched in lane `k`: its event waited for (the step's own work, not what was queued after it), its samples read from the
    /// lane: the decode rows', then each chunk's that ends its prompt. Its head rows' logits are the scheduler's to read (`top`, `logits`) only where no later step is out.
    fn land_step(&mut self, k: usize) -> Result<crate::sched::Samples> {
        let called = std::time::Instant::now();
        let head = self.head_room();
        let lanes = self.lanes.as_mut().ok_or_else(|| crate::ffi::Error("a launched step's samples on an engine with no lanes".into()))?;
        lanes.lane[k].done.wait_host()?;
        lanes.out = lanes.out.saturating_sub(1);
        lanes.stale = lanes.out > 0;
        let later = lanes.out > 0;
        let gap = lanes.gap[k];
        let lane = &mut lanes.lane[k];
        let asked = lane.rows + lane.lasts.iter().filter(|&&l| l).count();
        // SAFETY: the lane's pinned room, which the step's copies (done: its event) wrote `asked` tokens and logits of.
        let (tok, val) = unsafe {
            (
                std::slice::from_raw_parts(lane.outs.ptr as *const i32, asked).to_vec(),
                std::slice::from_raw_parts((lane.outs.ptr as *const u8).add(4 * head) as *const f32, asked).to_vec(),
            )
        };
        let mut it = tok.into_iter().zip(val).map(|(t, v)| Sample { token: t as u32, logit: v, logprob: None });
        let decoded: Vec<Sample> = it.by_ref().take(lane.rows).collect();
        let prompted: Vec<Option<Sample>> = lane.lasts.iter().map(|&l| if l { it.next() } else { None }).collect();
        let (rows, prompt, graph, queue_ms) = (lane.rows, lane.prompt, lane.graph, lane.queue_ms);
        let evs = std::mem::take(&mut lane.evs);
        if graph && !later {
            self.replayed.set(false);
        }
        let drawn = lane.drawn;
        // SAFETY: the lane's pinned sums, written by the step's copies (its event waited for) for the rows that asked.
        let sums: Vec<Option<(f32, f64)>> =
            lane.asks.iter().enumerate().map(|(r, a)| a.filter(|&t| t > 0.0).map(|t| (t, unsafe { *(lane.sums.ptr as *const f64).add(r) }))).collect();
        // (the step's own status for the lost blocks the scheduler asks for next, `kv12_lost`)
        self.status_from = Some(lane.status.ptr as *const u32);
        self.cands.fresh(asked);
        // (made with the step where it draws: its event covers their copy)
        self.cands.made = drawn;
        self.cands.sums = sums;
        // (the expert cache's end of a step reads the GPU's counters and writes the plan's next settings: never while a later step is queued, whose calls read them; owed until
        // a step lands with none after it, which `quiet` asks for every QUIET_EVERY steps)
        if later {
            self.owed += 1;
        } else {
            self.end_expert_step()?;
            self.owed = 0;
        }
        // (GLYD_PROF=1: the pass's operations, by its own events; a graph's replay only where no later replay has recorded its events again)
        if super::prof::on() && self.prof.is_some() {
            fn refs<'a>(v: &'a [(&'static str, crate::dev::Ev)]) -> Vec<(&'static str, &'a crate::dev::Ev)> {
                v.iter().map(|(l, e)| (*l, e)).collect()
            }
            let ops = match graph {
                true if !later => super::prof::between(&refs(&self.gmarks))?,
                true => Vec::new(),
                false => super::prof::between(&refs(&evs))?,
            };
            let gpu = ops.iter().map(|x| x.1).sum();
            if !ops.is_empty() {
                super::prof::pass(rows, prompt, &ops, graph, (!graph).then_some(queue_ms), self.dcalls.get(), 0);
            }
            super::prof::launched(rows, prompt, queue_ms + called.elapsed().as_secs_f64() * 1e3, gpu);
        }
        self.spare.borrow_mut().extend(evs.into_iter().map(|x| x.1));
        let at = called - std::time::Duration::from_secs_f64(queue_ms / 1e3);
        let p = StepProf::on().then_some(StepProf { at });
        self.step_times_of(p, rows, prompt, Some(queue_ms + called.elapsed().as_secs_f64() * 1e3), gap);
        Ok((decoded, prompted))
    }

    /// Head row `row`'s candidates of the last step (`Cands`): None where the step wrote no such row, the vocabulary is not one the top-k kernel takes, or
    /// GLYD_ENGINE_HOST_SAMPLING=1 asks for the row on the host.
    fn row_top(&mut self, row: usize) -> Result<Option<Vec<(u32, f32)>>> {
        use crate::kern::TOPK;
        let v = self.big.d.vocab;
        if self.top.off || row >= self.cands.rows || !v.is_multiple_of(8) || v < TOPK {
            return Ok(None);
        }
        if !self.cands.made {
            self.queue_cands(self.cands.rows)?;
            self.dev.sync()?;
            self.cands.made = true;
        }
        let (_, _, host, cap) = self.cands.bufs.as_ref().expect("made with the candidates");
        // SAFETY: the pinned bytes were copied (the stream synchronized since, or the step's event waited for), TOPK for each row below `rows`.
        Ok(Some(unsafe { crate::kern::read_candidates(host.ptr as *const u8, 0, 4 * TOPK * cap, row) }))
    }

    /// The candidates of the last pass's first `n` head rows (`Cands`) queued on the stream: one top-k launch for all of them and their copy to the host's pinned room, read
    /// once the stream has run them (a sync, or the event of a launched step recorded after them).
    fn queue_cands(&mut self, n: usize) -> Result<()> {
        use crate::kern::TOPK;
        let v = self.big.d.vocab;
        let cap = self.head_room().max(n);
        if self.cands.bufs.as_ref().is_none_or(|b| b.3 < cap) {
            self.cands.bufs = Some((Buf::new(4 * TOPK * cap)?, Buf::new(4 * TOPK * cap)?, crate::dev::Pinned::new(8 * TOPK * cap)?, cap));
        }
        let (tok, val, host, cap) = self.cands.bufs.as_ref().map(|b| (b.0.ptr, b.1.ptr, b.2.ptr as *mut u8, b.3)).expect("made above");
        let s = self.dev.stream;
        // (the grammar masks of the step's sampled rows, given before it: the candidates of those rows made again with the mask put on the row before the top-k)
        let masks = std::mem::take(&mut self.row_masks);
        let masked = masks.iter().any(Option::is_some);
        if masked && self.masker.is_none() {
            self.masker = crate::kern::MaskGpu::new(v, cap).ok();
        }
        if masked && self.cands.raw.as_ref().is_none_or(|r| r.bytes < 8 * TOPK * cap) {
            self.cands.raw = Some(crate::dev::Pinned::new(8 * TOPK * cap)?);
        }
        // SAFETY: the pass's head rows of v logits each (n of them, within the head's room); TOPK of each to the buffers made for `cap` rows; the pinned rooms hold the
        // tokens, then from 4 TOPK cap on the logits, read after the stream has run the copies; the masker's staging was read by a copy the stream has run (a step's
        // candidates are made once its step before landed).
        unsafe {
            self.kern.topk_rows(s, self.big.logits.ptr, v as i32, n as u32, tok, val)?;
            if let Some(g) = self.masker.as_ref().filter(|_| masked) {
                // (the rows' own candidates copied first, then made again with the masks over them)
                let raw = self.cands.raw.as_ref().expect("made above").ptr as *mut u8;
                crate::dev::copy_d2h_async(raw as *mut _, tok, 4 * TOPK * n, s)?;
                crate::dev::copy_d2h_async(raw.add(4 * TOPK * cap) as *mut _, val, 4 * TOPK * n, s)?;
                if g.apply(&self.kern, &self.dev, &masks[..masks.len().min(n)], self.big.logits.ptr, tok, val)? {
                    self.cands.raw_rows = n;
                }
            }
            crate::dev::copy_d2h_async(host as *mut _, tok, 4 * TOPK * n, s)?;
            crate::dev::copy_d2h_async(host.add(4 * TOPK * cap) as *mut _, val, 4 * TOPK * n, s)?;
        }
        Ok(())
    }

    /// `ModelStep::hist`: row `row`'s HIST_BUCKETS buckets of its weights at `temperature` (`Kern::exp_hist`), and the row's largest logit they are about.
    fn row_hist(&mut self, row: usize, temperature: f32) -> Result<Option<crate::sampler::Hist>> {
        use crate::sampler::{HIST_BUCKETS, HIST_PER};
        let v = self.big.d.vocab;
        if self.top.off || row >= self.cands.rows || !self.kern.has_hist() || self.lanes.as_ref().is_some_and(|l| l.stale) {
            return Ok(None);
        }
        let (sums, counts) = (Buf::new(8 * HIST_BUCKETS)?, Buf::new(4 * HIST_BUCKETS)?);
        let s = self.dev.stream;
        // SAFETY: the step's head row `row` of v logits and its largest (f32) in val; HIST_BUCKETS sums and counts; read after the synchronize.
        unsafe {
            self.kern.exp_hist(
                s,
                self.big.logits.ptr + (row * v * 2) as u64,
                v as i32,
                self.big.val.ptr + (4 * row) as u64,
                temperature,
                HIST_BUCKETS as i32,
                HIST_PER,
                sums.ptr,
                counts.ptr,
            )?
        };
        self.dev.sync()?;
        let (mut h, mut c, mut m) = (vec![0f64; HIST_BUCKETS], vec![0u32; HIST_BUCKETS], [0f32; 1]);
        sums.read(0, &mut h)?;
        counts.read(0, &mut c)?;
        self.big.val.read(4 * row, &mut m)?;
        Ok(Some(crate::sampler::Hist { max: m[0], per: HIST_PER, sums: h, counts: c }))
    }

    /// `ModelStep::hist_bucket`: the tokens of bucket `bucket` of row `row`'s `row_hist` (`Kern::hist_bucket`), at most `cap`.
    fn row_bucket(&mut self, row: usize, temperature: f32, bucket: usize, cap: usize) -> Result<Option<Vec<(u32, f32)>>> {
        use crate::sampler::{HIST_BUCKETS, HIST_PER};
        let v = self.big.d.vocab;
        if self.top.off || row >= self.cands.rows || !self.kern.has_hist() || self.lanes.as_ref().is_some_and(|l| l.stale) {
            return Ok(None);
        }
        let (want, n, tok, val) = (Buf::new(4)?, Buf::new(4)?, Buf::new(4 * cap.max(1))?, Buf::new(4 * cap.max(1))?);
        want.write(0, &[bucket as i32])?;
        let s = self.dev.stream;
        // SAFETY: as `row_hist`; want holds the bucket, n one count, tok and val room for `cap`; read after the synchronize.
        unsafe {
            self.kern.hist_bucket(
                s,
                self.big.logits.ptr + (row * v * 2) as u64,
                v as i32,
                self.big.val.ptr + (4 * row) as u64,
                temperature,
                HIST_BUCKETS as i32,
                HIST_PER,
                want.ptr,
                cap as i32,
                n.ptr,
                tok.ptr,
                val.ptr,
            )?
        };
        self.dev.sync()?;
        let mut k = [0u32; 1];
        n.read(0, &mut k)?;
        let k = k[0] as usize;
        if k > cap {
            return Ok(None);
        }
        let (mut t, mut x) = (vec![0i32; k], vec![0f32; k]);
        tok.read(0, &mut t)?;
        val.read(0, &mut x)?;
        Ok(Some(t.into_iter().map(|t| t as u32).zip(x).collect()))
    }

    /// `ModelStep::mass`: row `row`'s sum of exp((logit - largest) / temperature) on the GPU, about the head's largest (`val`), read back (8 bytes, not the row).
    fn row_mass(&mut self, row: usize, temperature: f32) -> Result<Option<f64>> {
        let v = self.big.d.vocab;
        if self.top.off || row >= self.cands.rows || !v.is_multiple_of(8) || self.lanes.as_ref().is_some_and(|l| l.stale) {
            return Ok(None);
        }
        if self.sums.is_none() {
            self.sums = Some(Buf::zeroed(8 * self.head_room())?);
        }
        let out = self.sums.as_ref().expect("made above").ptr;
        // SAFETY: the step's head row `row` of v logits and its largest (f32) in val; one f64 out; read after the synchronize.
        unsafe {
            self.kern.exp_sum_rows(
                self.dev.stream,
                self.big.logits.ptr + (row * v * 2) as u64,
                v as i32,
                1,
                self.big.val.ptr + (4 * row) as u64,
                temperature,
                out,
            )?
        };
        self.dev.sync()?;
        let mut m = [0f64; 1];
        self.sums.as_ref().expect("made above").read(0, &mut m)?;
        Ok(Some(m[0]))
    }

    /// `ModelStep::round` with prompt chunks beside the rows.
    fn round_rows(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<(Vec<Sample>, Vec<Option<Sample>>)> {
        if rows.is_empty() {
            return Ok((Vec::new(), ModelStep::prefill_many(self, jobs)?));
        }
        for (slot, at) in rows.iter().map(|r| (r.pages[0] as usize, r.pos)).chain(jobs.iter().map(|j| (j.pages[0] as usize, j.start))) {
            if self.seq_n(slot) != at {
                return err(format!("tokens at {at} of sequence {slot}, which holds {}", self.seq_n(slot)));
            }
        }
        let ones: Vec<(usize, [u32; 1])> = rows.iter().map(|r| (r.pages[0] as usize, [r.token])).collect();
        let parts: Vec<(usize, &[u32], bool)> =
            ones.iter().map(|(s, t)| (*s, &t[..], true)).chain(jobs.iter().map(|j| (j.pages[0] as usize, j.tokens, j.last))).collect();
        let mut out = self.pass(&parts)?.into_iter();
        let decoded: Vec<Sample> = out.by_ref().take(rows.len()).map(|s| s.expect("every row samples")).collect();
        Ok((decoded, out.collect()))
    }
    /// `ModelStep::decode`.
    fn decode_rows(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        for r in rows {
            let slot = r.pages[0] as usize;
            if self.seq_n(slot) != r.pos {
                return err(format!("a token at {} of sequence {slot}, which holds {}", r.pos, self.seq_n(slot)));
            }
        }
        if rows.len() > self.head_room() {
            return err(format!("a step of {} rows, where the head holds {}", rows.len(), self.head_room()));
        }
        if rows.len() == 1 || !self.batchable() || (!Self::batch_prefill() && !self.rows_exact()) {
            // (one sequence's step as a reply alone takes it, its graph where it has one, on the blocks the scheduler's growth took; several one after another, the last
            // first, each one's logits row moved to its place in the head's rows before the next step writes its own over the first)
            let mut out = vec![None; rows.len()];
            let w = self.big.d.vocab * 2;
            for (j, r) in rows.iter().enumerate().rev() {
                self.select(r.pages[0] as usize)?;
                out[j] = Some(self.step_with(r.token, false)?);
                if j > 0 {
                    // SAFETY: the head's logits hold head_room rows; the step's are row 0.
                    unsafe { copy_d2d_async(self.big.logits.ptr + (j * w) as u64, self.big.logits.ptr, w, self.dev.stream)? };
                }
            }
            return Ok(out.into_iter().map(|s| s.expect("every row stepped")).collect());
        }
        let jobs: Vec<(usize, [u32; 1])> = rows.iter().map(|r| (r.pages[0] as usize, [r.token])).collect();
        let jobs: Vec<(usize, &[u32], bool)> = jobs.iter().map(|(s, t)| (*s, &t[..], true)).collect();
        if !Self::batch_prefill() && rows.len() > EXACT_ROWS {
            // (GLYD_ENGINE_BATCH_PREFILL=0, where every reply is to be the reply alone bit for bit: passes of at most EXACT_ROWS rows, the last first, each one's logits rows moved to
            // their place in the head's rows before the next writes its own over the first)
            let mut out = vec![None; rows.len()];
            let w = self.big.d.vocab * 2;
            for (g, part) in jobs.chunks(EXACT_ROWS).enumerate().collect::<Vec<_>>().into_iter().rev() {
                let at = g * EXACT_ROWS;
                for (j, s) in self.pass(part)?.into_iter().enumerate() {
                    out[at + j] = s;
                }
                if at > 0 {
                    // SAFETY: the head's logits hold head_room rows; the pass's are its first rows, and they do not reach their place (a pass is at most EXACT_ROWS rows, `at` at least that).
                    unsafe { copy_d2d_async(self.big.logits.ptr + (at * w) as u64, self.big.logits.ptr, part.len() * w, self.dev.stream)? };
                }
            }
            return Ok(out.into_iter().map(|s| s.expect("every row samples")).collect());
        }
        Ok(self.pass(&jobs)?.into_iter().map(|s| s.expect("every row samples")).collect())
    }
}

impl ModelStep for Dyn {
    /// The context, or where the pools hold less for one sequence alone, that: a reply that reaches it ends there (its length), and a sequence alone always fits, so one sent back for room
    /// always runs again (an admitted request never fails for the memory).
    fn window(&self) -> usize {
        match self.paged.as_ref().and_then(|p| p.pages.capacity()) {
            Some(c) => self.ctx.min(c),
            None => self.ctx,
        }
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
    fn images(&mut self, slot: i32, prompt: &[u32], images: &[std::sync::Arc<crate::vision::Image>]) -> Result<()> {
        match self.vis.as_mut() {
            Some(v) => v.set(slot as usize, prompt, images),
            None if images.is_empty() => Ok(()),
            None => err("this model reads no images (its description has no vision tower)"),
        }
    }
    fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        let slot = pages[0] as usize;
        if self.seq_n(slot) != start {
            return err(format!("a chunk at {start} of sequence {slot}, which holds {}", self.seq_n(slot)));
        }
        let timed = super::prof::begin();
        let r = self.chunk_alone(slot, tokens, last)?;
        self.kv12_stepped();
        self.cands.fresh(last as usize);
        self.end_expert_step()?;
        super::prof::end(timed, 0, tokens.len());
        Ok(r)
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
        let timed = super::prof::begin();
        let r = self.pass(&jobs)?;
        self.kv12_stepped();
        self.cands.fresh(jobs.iter().filter(|j| j.2).count());
        self.end_expert_step()?;
        super::prof::end(timed, 0, jobs.iter().map(|j| j.1.len()).sum());
        Ok(r)
    }
    fn one_step(&self) -> bool {
        Self::one_step_asked() && self.batchable() && Self::batch_prefill()
    }
    fn prompt_budget(&self, rows: usize) -> usize {
        Self::step_tokens().unwrap_or(self.chunk).saturating_sub(rows).max(STEP_LEAST).min(self.chunk.saturating_sub(rows))
    }
    /// The rows that decode and the prompts' chunks in one pass (`pass`: a row is a chunk of one token whose sample is asked for): the rows' samples, then the samples of the chunks that are
    /// `last`, the head's rows in that order. Rows alone are the decode step as it is (one row: its graph), chunks alone the prompt pass.
    fn round(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<(Vec<Sample>, Vec<Option<Sample>>)> {
        if jobs.is_empty() {
            return Ok((ModelStep::decode(self, rows)?, Vec::new()));
        }
        let (p, timed) = (StepProf::begin(), super::prof::begin());
        let r = self.round_rows(rows, jobs);
        self.kv12_stepped();
        let prompt = jobs.iter().map(|j| j.tokens.len()).sum();
        self.step_times(p, rows.len(), prompt, None);
        self.cands.fresh(rows.len() + jobs.iter().filter(|j| j.last).count());
        self.end_expert_step()?;
        super::prof::end(timed, rows.len(), prompt);
        r
    }
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        let (p, timed) = (StepProf::begin(), super::prof::begin());
        let r = self.decode_rows(rows);
        self.kv12_stepped();
        self.step_times(p, rows.len(), 0, None);
        self.cands.fresh(rows.len());
        self.end_expert_step()?;
        super::prof::end(timed, rows.len(), 0);
        r
    }
    /// A served model whose description runs the round as one step on one GPU, bf16 pages, no expert cache and no layer in the computer's memory (their host work is in the
    /// pass): `Lanes`, made by `serve_rows`. A recurrent model's states are copied at a mark or a boundary as the next step is launched (the scheduler asks then: `launch`),
    /// and the operations that read the tokens (an n-gram embedding, routing by token) read the ids `step_ids` makes. Only with GLYD_ENGINE_OVERLAP=1 (`make_lanes`).
    fn overlaps(&self) -> bool {
        self.lanes.is_some()
    }
    /// The step queued (`queue_step`), its rows' tokens read on the GPU by the embedding's lookup: a row's token the step before chose (`Row::after`) is read where that
    /// step's head left it. A step that does not overlap runs `round` to its end.
    fn launch(&mut self, rows: &[Row], jobs: &[PrefillJob]) -> Result<crate::sched::Launched> {
        if self.lanes.is_none() {
            if rows.iter().any(|r| r.after.is_some()) {
                return err("a row whose token the step before chose, on a step that is not launched before it is read");
            }
            return Ok(crate::sched::Launched { done: Some(ModelStep::round(self, rows, jobs)?), tag: 0 });
        }
        self.queue_step(rows, jobs)
    }
    fn finish(&mut self, l: crate::sched::Launched) -> Result<crate::sched::Samples> {
        match l.done {
            Some(d) => Ok(d),
            None => self.land_step(l.tag),
        }
    }
    /// The steps out waited for (where the stream still runs; an error is the scheduler's already) and their lanes free.
    fn abandon(&mut self) {
        let _ = self.dev.sync();
        if let Some(l) = &mut self.lanes {
            (l.out, l.stale) = (0, false);
        }
    }
    fn draws(&mut self, asks: &[Option<f32>]) {
        self.draws = asks.to_vec();
    }
    /// An expert cache's end of a step owed for QUIET_EVERY steps launched ahead (`land_step`): the next step is launched after this one lands.
    fn quiet(&self) -> bool {
        self.owed >= QUIET_EVERY && !self.stores().is_empty()
    }
    fn cold(&mut self) -> Result<()> {
        self.owed = 0;
        self.stores().iter().try_for_each(|st| st.cold(&self.dev))
    }
    fn trace_choices(&mut self, on: bool) {
        self.trace_row_routes(on);
    }
    fn row_choices(&self, slot: i32, pos: usize) -> Option<Vec<super::RowRoute>> {
        self.row_route(slot as usize, pos)
    }
    fn masks(&mut self, masks: Vec<Option<crate::constrain::MaskHandle>>) {
        self.row_masks = masks;
    }
    /// Row `row`'s buckets on the GPU (`exp_hist`), about the head's largest (`val`), read back (12 KB, not the row).
    fn hist(&mut self, row: usize, temperature: f32) -> Option<crate::sampler::Hist> {
        self.row_hist(row, temperature).unwrap_or(None)
    }
    fn hist_bucket(&mut self, row: usize, temperature: f32, bucket: usize, cap: usize) -> Option<Vec<(u32, f32)>> {
        self.row_bucket(row, temperature, bucket, cap).unwrap_or(None)
    }
    /// The row's own candidates where a grammar's mask made `top`'s (copied before the mask was put on): those of `top` otherwise.
    fn top_raw(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        let top = ModelStep::top(self, row)?;
        if row >= self.cands.raw_rows {
            return Some(top);
        }
        let (raw, cap) = (self.cands.raw.as_ref()?, self.cands.bufs.as_ref()?.3);
        // SAFETY: the pinned room the unmasked candidates were copied into with the masked ones (the stream run since: `top` read those), TOPK a row below `raw_rows`.
        Some(unsafe { crate::kern::read_candidates(raw.ptr as *const u8, 0, 4 * crate::kern::TOPK * cap, row) })
    }
    fn mass(&mut self, row: usize, temperature: f32) -> Option<f64> {
        // (made with the step where it asked for this one; else made now)
        if let Some(&Some((t, m))) = self.cands.sums.get(row).filter(|_| self.lanes.as_ref().is_none_or(|l| !l.stale)) {
            if t == temperature {
                return Some(m);
            }
        }
        self.row_mass(row, temperature).unwrap_or(None)
    }
    /// The candidates the GPU makes of the step's head rows (`Cands`); where it cannot (an error), the row is read as without them.
    fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        if self.lanes.as_ref().is_some_and(|l| l.stale) {
            return None;
        }
        self.row_top(row).unwrap_or(None)
    }
    fn expert_stats(&self) -> Vec<crate::offload::LayerStats> {
        Dyn::expert_stats(self)
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        self.rows_readable()?;
        self.dev.sync()?;
        let v = self.big.d.vocab;
        let mut words = vec![0u16; v];
        self.big.logits.read(row * v * 2, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }
    fn gather(&mut self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
        self.rows_readable()?;
        let v = self.big.d.vocab;
        let toks: Vec<u32> = tokens.iter().copied().filter(|&t| (t as usize) < v).collect();
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        let idx: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        let (ib, ob) = (Buf::new(idx.len() * 4)?, Buf::new(idx.len() * 4)?);
        ib.write(0, &idx)?;
        // SAFETY: the step's logits rows of v bf16 each (row is one of them, as `logits` reads it); idx holds n tokens below v, out room for n f32; the
        // stream is synchronized before out is read.
        unsafe { self.kern.gather_row(self.dev.stream, self.big.logits.ptr, v, row, ib.ptr, idx.len(), ob.ptr)? };
        self.dev.sync()?;
        let mut vals = vec![0f32; idx.len()];
        ob.read(0, &mut vals)?;
        Ok(toks.into_iter().zip(vals).collect())
    }
    fn lost(&mut self) -> crate::kv::Lost {
        // (a status that cannot be read: every running sequence's step taken back)
        self.kv12_lost().unwrap_or(crate::kv::Lost { overflow: true, ..Default::default() })
    }
    fn release(&mut self, pages: &[i32]) -> Result<()> {
        for &p in pages {
            // (a slot being copied back from the host: the copy done first)
            self.swap_settle(p as usize)?;
            self.empty(p as usize)?;
            // (a check's trace of rows' choices: the sequence's rows forgotten with it; its images)
            self.trace_forget(p as usize);
            if let Some(v) = self.vis.as_mut() {
                v.forget(p as usize);
            }
            // (what this step's growths promised it of the 12-bit pools goes with it)
            self.kv12_unpromise(p as usize);
            if let Some(pg) = &mut self.paged {
                // (its 12-bit pages that no one reads again cleared: their chunks and slabs back to the pools, for the sequences that go on)
                let freed = if self.lossless.is_some() { pg.pages.sole_unnamed(p as usize) } else { Vec::new() };
                pg.pages.release(p as usize);
                if freed.iter().any(|f| !f.is_empty()) {
                    // (12-bit pages cleared: their chunks and slabs go back to pools a step in flight may write, so that step lands first)
                    self.dev.sync()?;
                    self.upload_tables(p as usize)?;
                    for (g, f) in freed.iter().enumerate() {
                        self.kv12_release_blocks(g, &f.iter().map(|x| x.1).collect::<Vec<u32>>())?;
                    }
                } else {
                    // (a step in flight may still read the tables as they were: the copy is queued after it, and a block given back is written only by the steps after)
                    self.upload_tables_queued(p as usize)?;
                }
            }
        }
        Ok(())
    }
    /// A finished sequence stays in its slot for the conversation's next turn. Not with GLYD_ENGINE_KEEP=0 (its slot is emptied at once, as before), nor with
    /// GLYD_ENGINE_BATCH_PREFILL=0, where every reply is to be the reply alone bit for bit: a turn that goes on in a kept slot runs the rest of its prompt in a pass of fewer rows
    /// than the whole prompt's, which is another kernel's (the module's note).
    fn keeps(&self) -> bool {
        marks_kept() && self.draft.is_none()
    }
    /// A drafted model's decode of its one sequence: a drafting round (`spec_round`), as a reply alone runs it.
    fn draft(&mut self, row: &Row, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Option<Vec<u32>>> {
        if self.draft.is_none() {
            return Ok(None);
        }
        self.select(row.pages[0] as usize)?;
        if self.n != row.pos {
            return err(format!("a token at {} of a sequence that holds {}", row.pos, self.n));
        }
        let (p, timed) = (StepProf::begin(), super::prof::begin());
        let toks = self.spec_round(row.token, sampler)?;
        self.step_times(p, 1, 0, None);
        self.cands.fresh(0);
        self.end_expert_step()?;
        super::prof::end(timed, 1, 0);
        Ok(Some(toks))
    }
    /// Full blocks are found by name where the keys and values are paged (`Pages::name`, `Pages::adopt`): where nothing of a sequence is a state, a start of any whole
    /// number of blocks; a model with recurrent states and the pool of them (`named`), a start that ends at a boundary whose states are kept.
    fn shares_starts(&self) -> bool {
        self.paged.is_some() && (self.cuts() || self.pooled())
    }
    fn snap_every(&self) -> usize {
        if self.paged.is_some() && !self.cuts() && self.pooled() {
            self.named.every
        } else {
            0
        }
    }
    fn name(&mut self, slot: i32, first: usize, keys: &[u128]) -> Result<()> {
        if !self.shares_starts() {
            return Ok(());
        }
        let t = slot as usize;
        let pg = self.paged.as_mut().expect("paged");
        for (i, &k) in keys.iter().enumerate() {
            pg.pages.name(t, first + i, k);
        }
        // (a model with recurrent states: the states, where the last block named ends at the sequence's last token and on a boundary, are the states there)
        let end = (first + keys.len()) * BLOCK;
        if let Some(&key) = keys.last().filter(|_| !self.cuts() && self.pooled() && end == self.seq_n(t) && end % self.named.every == 0) {
            self.keep_states_at(t, end / BLOCK - 1, key)?;
        }
        Ok(())
    }
    fn take_removed(&mut self) -> Vec<u128> {
        self.paged.as_mut().map_or_else(Vec::new, |pg| pg.pages.take_gone())
    }
    fn held_by_name(&self, keys: &[u128]) -> usize {
        match &self.paged {
            Some(pg) if self.cuts() => pg.pages.fit(keys, self.chunk) * BLOCK,
            Some(pg) if self.pooled() => {
                snaps::hit(keys, self.named.every, BLOCK, |j| self.kept_copy(keys, j).is_some(), |j| pg.pages.fit(&keys[..j], self.chunk))
            }
            _ => 0,
        }
    }
    /// The blocks are the pools' as they are (their bytes are on the GPU whether a sequence holds them or they wait by name): the slot's tables name them, and the
    /// sequence is as long as they are.
    fn adopt(&mut self, slot: i32, keys: &[u128]) -> Result<usize> {
        let t = slot as usize;
        if !self.shares_starts() || self.seq_n(t) != 0 {
            return Ok(0);
        }
        self.settle_slot(t)?;
        let chunk = self.chunk;
        if self.cuts() {
            let n = self.paged.as_mut().expect("paged").pages.adopt(t, keys, chunk);
            if n > 0 {
                self.set_seq_n(t, n);
                self.dev.sync()?;
                self.upload_tables(t)?;
            }
            return Ok(n);
        }
        // (recurrent states: the blocks to the newest boundary whose states are kept, and those states)
        let n = self.held_by_name(keys);
        let Some(i) = self.kept_copy(keys, n / BLOCK).filter(|_| n > 0) else { return Ok(0) };
        let pg = self.paged.as_mut().expect("paged");
        if pg.pages.adopt(t, &keys[..n / BLOCK], chunk) != n {
            pg.pages.release(t);
            return Ok(0);
        }
        let to = self.part_of(t, 0, 0, 0)?.states;
        for (j, &dst) in to.iter().enumerate() {
            // SAFETY: a kept copy's buffer and the sequence's state, of the state's bytes.
            unsafe { copy_d2d_async(dst, self.named.bufs[i][j].ptr, self.states[j].bytes, self.dev.stream)? };
        }
        self.named.lru.touch(i);
        self.set_seq_n(t, n);
        self.dev.sync()?;
        self.upload_tables(t)?;
        Ok(n)
    }
    /// A start shared with an empty sequence (`Pages::share`; paged keys and values only). With no recurrent state: the full blocks of the first `upto`
    /// tokens. With recurrent states kept at a mark: `from`'s mark, where it is within `upto`: the full blocks before it shared, the block the mark is in copied to one of
    /// `to`'s own (its tokens before the mark are `from`'s; what `to` writes from the mark on is its own), and the states as they were at the mark copied to `to`'s, and to
    /// its mark.
    fn share(&mut self, from: i32, to: i32, upto: usize) -> Result<usize> {
        let (f, t) = (from as usize, to as usize);
        if self.paged.is_none() || f == t || self.seq_n(t) != 0 || upto > self.seq_n(f) {
            return Ok(0);
        }
        self.settle_slot(t)?;
        let chunk = self.chunk;
        let to_n = if self.cuts() {
            self.paged.as_mut().expect("paged").pages.share(f, t, upto, chunk)
        } else {
            let m = match self.snap_n.get(f).copied() {
                Some(m) if m > 0 && m <= upto && self.can_mark() && !self.snaps.is_empty() => m,
                _ => return Ok(0),
            };
            let pg = self.paged.as_mut().expect("paged");
            if !pg.pages.holds(f, m, chunk) {
                return Ok(0);
            }
            pg.pages.share(f, t, m, chunk);
            // the block the mark is in, in each group where it is not a whole block (a group of larger blocks, compressed entries', may have one where the others do not): one of `to`'s
            // own, with `from`'s bytes in every ring of the group
            let parts: Vec<Option<usize>> = (0..pg.pages.groups()).map(|g| pg.pages.partial(g, m)).collect();
            if parts.iter().any(Option::is_some) {
                if !pg.pages.grow(t, m) {
                    pg.pages.release(t);
                    return Ok(0);
                }
                let blocks: Vec<(usize, u32, u32)> = (0..pg.group.len())
                    .filter_map(|r| parts[pg.group[r]].map(|b| (r, pg.pages.table(f, pg.group[r])[b], pg.pages.table(t, pg.group[r])[b])))
                    .collect();
                for (r, src, dst) in blocks {
                    let ring = &self.big.d.rings[r];
                    let bytes = BLOCK * ring.nk * ring.d * 2;
                    if self.ring_cap[r] == 0 || src == 0 || dst == 0 {
                        continue;
                    }
                    let (ko, vo) = self.ring_off[r];
                    for at in std::iter::once(self.kc.ptr + ko).chain(ring.values.then_some(self.vc.ptr + vo)) {
                        // SAFETY: two blocks of the ring's pool, a block's bytes each.
                        unsafe { copy_d2d_async(at + (dst as usize * bytes) as u64, at + (src as usize * bytes) as u64, bytes, self.dev.stream)? };
                    }
                }
            }
            // the states as they were at the mark: `to`'s now, and at its own mark
            let states = self.part_of(t, 0, 0, 0)?.states;
            for (i, &dst) in states.iter().enumerate() {
                let (src, bytes) = (self.snaps[f][i].ptr, self.states[i].bytes);
                // SAFETY: a state's copy, the other sequence's state and its copy's buffer, of the state's bytes.
                unsafe {
                    copy_d2d_async(dst, src, bytes, self.dev.stream)?;
                    copy_d2d_async(self.snaps[t][i].ptr, src, bytes, self.dev.stream)?;
                }
            }
            self.snap_n[t] = m;
            m
        };
        if to_n > 0 {
            self.set_seq_n(t, to_n);
            self.dev.sync()?;
            self.upload_tables(t)?;
            // (a check's trace of rows' choices: the start's rows are `from`'s)
            self.trace_share(f, t, to_n);
        }
        Ok(to_n)
    }
    /// The states of the sequence in `slot`, which now holds `n` tokens, copied aside (a model that keeps marks: `serve_rows`): a kept sequence can go on from here.
    fn mark(&mut self, slot: i32, n: usize) -> Result<()> {
        let s = slot as usize;
        if self.snaps.is_empty() || n == 0 || n != self.seq_n(s) {
            return Ok(());
        }
        let from = self.part_of(s, 0, 0, 0)?.states;
        for (i, &src) in from.iter().enumerate() {
            // SAFETY: the sequence's state and its copy's buffer, of the state's bytes.
            unsafe { copy_d2d_async(self.snaps[s][i].ptr, src, self.states[i].bytes, self.dev.stream)? };
        }
        self.snap_n[s] = n;
        Ok(())
    }
    /// The whole of what the slot holds goes on as it is. Before its end: at `upto` where the description has no recurrent state and no latent cache (those are what they are after
    /// the sequence's last token); else at the sequence's mark, where its states were copied aside, if that is within `upto`. Either way the keys and values a step there reads must
    /// still be held: paged, the blocks (`Pages::cut_len`; a cut into a block another sequence holds too is to that block's start, which a state at a mark cannot follow);
    /// a ring of its own, one that has not gone round. Where it cannot go on from any place nothing changes and the answer is 0.
    fn resume(&mut self, slot: i32, upto: usize) -> Result<usize> {
        let s = slot as usize;
        let n = self.seq_n(s);
        if upto > 0 && upto == n {
            return Ok(n);
        }
        let marked = self.snap_n.get(s).copied().filter(|&m| m > 0 && m <= upto);
        let to = match (upto > 0 && upto < n, self.cuts(), marked) {
            (true, true, _) => upto,
            (true, false, Some(m)) => m,
            _ => 0,
        };
        let chunk = self.chunk;
        let kept = match (&self.paged, to) {
            (_, 0) => 0,
            (Some(pg), _) => pg.pages.cut_len(s, to, chunk),
            (None, _) if self.ring_cap.iter().all(|&cap| cap == 0 || n <= cap) => to,
            (None, _) => 0,
        };
        if kept == 0 || (kept != to && !self.cuts()) {
            return Ok(0);
        }
        if let Some(pg) = &mut self.paged {
            pg.pages.shrink(s, kept, chunk);
        }
        if !self.cuts() {
            let states = self.part_of(s, 0, 0, 0)?.states;
            for (i, &dst) in states.iter().enumerate() {
                // SAFETY: the sequence's state and its copy's buffer, of the state's bytes.
                unsafe { copy_d2d_async(dst, self.snaps[s][i].ptr, self.states[i].bytes, self.dev.stream)? };
            }
        }
        self.set_seq_n(s, kept);
        self.dev.sync()?;
        self.upload_tables(s)?;
        Ok(kept)
    }
    /// From the tokens the slot holds: the room for a whole prompt, which the scheduler takes as it lets the prompt in, holds a sliding group's blocks from where its next pass reads
    /// (`Pages::grow_from`).
    fn grow(&mut self, slot: i32, upto: usize) -> Result<bool> {
        self.settle_slot(slot as usize)?;
        let (n, chunk) = (self.seq_n(slot as usize), self.chunk);
        // (12-bit pages: a slab for every page its next pass writes, else the newest sequence is paused: `kv12_room`)
        if !self.kv12_room(slot as usize, n, upto)? {
            return Ok(false);
        }
        let Some(pg) = &mut self.paged else { return Ok(true) };
        if !pg.pages.grow_from(slot as usize, n, upto, chunk) {
            self.kv12_unpromise(slot as usize);
            return Ok(false);
        }
        // (nothing to copy where no entry changed: most steps; else the entries copied on the stream, or after the steps before where the steps are not launched ahead)
        if pg.pages.dirty(slot as usize) {
            self.upload_tables_queued(slot as usize)?;
        }
        Ok(true)
    }
    /// Paged sequences are swapped to the host's area and back (src/dynamic/swap.rs), every family's the same way: their blocks in every group and their states.
    fn swap_cost(&self, slot: i32) -> Option<(usize, f64)> {
        self.swap_cost_of(slot as usize)
    }
    fn swap_out(&mut self, slot: i32) -> Result<Option<u64>> {
        self.swap_out_of(slot as usize)
    }
    fn swap_in(&mut self, slot: i32, ticket: u64, upto: usize) -> Result<Option<usize>> {
        self.swap_in_of(slot as usize, ticket, upto)
    }
    fn swap_ready(&mut self, slot: i32, wait: bool) -> Result<bool> {
        self.swap_ready_of(slot as usize, wait)
    }
    fn swap_drop(&mut self, ticket: u64) {
        self.swap_drop_of(ticket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step of rows that decode replays the graph of the least power of two of rows that holds it, the head's rows at most (a ladder of padded sizes, as vLLM's
    /// decode graphs); one row is its sequence's own graph, and more rows than the head holds none.
    #[test]
    fn a_step_takes_the_least_graph_that_holds_its_rows() {
        let got: Vec<Option<usize>> = (0..=10).map(|n| bucket(n, 8)).collect();
        assert_eq!(got, [None, None, Some(2), Some(4), Some(4), Some(8), Some(8), Some(8), Some(8), None, None]);
        // (a head of rows that is no power of two: the ladder ends at it)
        assert_eq!((bucket(5, 6), bucket(6, 6), bucket(3, 6)), (Some(6), Some(6), Some(4)));
        assert_eq!((bucket(33, 64), bucket(64, 64), bucket(17, 20)), (Some(64), Some(64), Some(20)));
    }

    /// A step of rows that decode alone is one graph where every operation acts row by row, is an attention layer, or is a recurrent layer's that takes such rows, each
    /// through its own state (the gated delta rule's and Mamba-2's); Kimi delta attention's operations and the latent attention keep it eager.
    #[test]
    fn steps_of_recurrent_layers_take_graphs() {
        use super::super::Slot;
        let lin = Op::Linear { w: 0, x: Slot::H, y: Slot::O, name: "product" };
        let gdn =
            [lin.clone(), Op::GdnConv { w: 0, state: 0, cd: 8, row: 8 }, Op::GdnDelta { a_log: 0, dt_bias: 0, state: 1, cd: 8, nk: 1, nv: 1 }, lin.clone()];
        assert!(rows_graphable(gdn.iter()));
        let mamba = [
            Op::MambaConv { w: 0, bias: 0, state: 0, cd: 8, row: 8, x_off: 0 },
            Op::MambaScan { dt_bias: 0, a_log: 0, d: 0, state: 1, cd: 8, row: 8, dt_off: 0, nh: 1, ng: 1 },
            Op::MambaNorm { w: 0, row: 8, di: 8, ng: 1 },
        ];
        assert!(rows_graphable(mamba.iter()));
        assert!(gdn.iter().chain(&mamba).filter(|op| rows_state(op)).count() == 4);
        let kda = Op::KdaConv { w: 0, state: 0, c: 8, row: 8, w32: false };
        assert!(!rows_state(&kda) && !rows_graphable([lin.clone(), kda].iter()));
    }

    /// The step's metadata in one copy: positions from 0, slots after the chunk's positions, the state table after them at 8 bytes; only as far as what is written.
    #[test]
    fn a_steps_metadata_is_one_copy() {
        let b = meta_bytes(4, 3, &[7, 8], &[2, -1], &[0xAA, 0, 0xBB, 0]);
        assert_eq!(b.len(), meta_states(4, 3) + 32);
        assert_eq!(meta_states(4, 3), 32, "28 bytes of positions and slots, then 8-byte entries");
        let i32_at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        assert_eq!((i32_at(0), i32_at(4), i32_at(16), i32_at(20)), (7, 8, 2, -1));
        assert_eq!((u64_at(32), u64_at(40), u64_at(48)), (0xAA, 0, 0xBB));
        assert_eq!(meta_bytes(4, 3, &[1, 2, 3], &[], &[]).len(), 12);
        assert_eq!(meta_bytes(4, 3, &[1], &[5], &[]).len(), 20);
    }

    /// The rows' state table: row r's sequence's states at r nst .. r nst + nst, padding rows 0 (their kernels then write zeros and touch no state).
    #[test]
    fn each_rows_states_are_at_its_entries_and_padding_rows_have_none() {
        let (a, b) = ([10u64, 11, 12], [20u64, 21, 22]);
        assert_eq!(state_table(&[&a, &b], 4, 3), [10, 11, 12, 20, 21, 22, 0, 0, 0, 0, 0, 0]);
        assert_eq!(state_table(&[&a], 1, 3), [10, 11, 12]);
    }

    /// A row's table is copied as many times as the decode's calls on the group make sequences of a row, every count of them dividing it (Gemma 4's global layers:
    /// 16 query heads over one kv head, two sequences a row; its sliding ones one). A call then finds row r's j-th sequence's table at the stride `dup / sp` tables,
    /// in row r's copies: the decode's launch passes that stride, and `kvshapes` holds what it reads on a GPU.
    #[test]
    fn a_rows_tables_are_copied_for_every_calls_sequences() {
        let dup = row_dups(3, [(0, 1), (1, 2), (1, 1), (0, 1)].into_iter());
        assert_eq!(dup, [1, 2, 0]);
        assert_eq!(row_dups(1, [(0, 2), (0, 3)].into_iter()), [6]);
    }

    /// The pools' blocks from a budget: groups of BLOCK tokens get as many blocks each (the sliding ones capped, what they leave to the full ones, as before groups of larger
    /// blocks), and a group whose blocks cover more tokens (compressed entries') as many tokens' worth: a fourth of the blocks where a block covers four times the tokens.
    #[test]
    fn every_group_holds_as_many_tokens() {
        // (bytes a token: a full group 2,048, a sliding one 1,024; 60 MiB)
        let budget = 60 << 20;
        let (per, keep) = ([2048, 1024], [None, Some(1024 + 256)]);
        let b = pool_blocks(budget, &per, &[BLOCK, BLOCK], &keep, &[usize::MAX, 4 * 81], &[0, 0]);
        // (as many blocks a group: 60 MiB / (3,072 B x 16) = 1,280, the sliding group capped at 324 and its rest to the full group)
        assert_eq!(b, vec![1280 + (1280 - 324) * 1024 / 2048, 324]);
        assert!(b[0] * BLOCK * 2048 + b[1] * BLOCK * 1024 <= budget);
        // a compressed group of 4 tokens an entry (256 bytes a token) beside them: its blocks hold 64 tokens, so a fourth as many hold as many tokens
        let b =
            pool_blocks(budget, &[2048, 1024, 256], &[BLOCK, BLOCK, 4 * BLOCK], &[None, Some(1024 + 256), None], &[usize::MAX, 4 * 81, usize::MAX], &[0, 0, 0]);
        // (18,904 tokens each: 1,181 blocks, 324, 295; the 6,108 tokens' bytes the sliding group leaves to the two full ones: 381 blocks and 95)
        assert_eq!(b, vec![1181 + 381, 324, 295 + 95]);
        assert!(b[0] * BLOCK - b[2] * 4 * BLOCK < 4 * BLOCK, "the full groups hold as many tokens, to a block");
        assert!(b[0] * BLOCK * 2048 + b[1] * BLOCK * 1024 + b[2] * 4 * BLOCK * 256 <= budget);
        // one sequence's blocks first: a sliding group of 64 times the full group's bytes a token, eight users, a budget short of every user's window: as many tokens'
        // blocks alone gave the full group 147 (2,352 tokens), less than one sequence's context of 4,096; with the sequence's 257 and 97 first, it holds the context
        let (per, keep, most, one) = ([2048, 64 * 2048], [None, Some(1024 + 512)], [8 * 257, 8 * 97], [257, 97]);
        let budget = 300 << 20;
        let b = pool_blocks(budget, &per, &[BLOCK, BLOCK], &keep, &most, &[0, 0]);
        assert!(b[0] * BLOCK < 4096, "{b:?}");
        let b = pool_blocks(budget, &per, &[BLOCK, BLOCK], &keep, &most, &one);
        assert!(b[0] >= 257 && b[1] >= 97 && b[0] * BLOCK * per[0] + b[1] * BLOCK * per[1] <= budget, "{b:?}");
        // (a budget that holds no sequence: as many tokens' blocks, as before)
        assert_eq!(pool_blocks(1 << 20, &per, &[BLOCK, BLOCK], &keep, &most, &one), pool_blocks(1 << 20, &per, &[BLOCK, BLOCK], &keep, &most, &[0, 0]));
    }
}
