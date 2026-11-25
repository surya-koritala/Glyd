//! The description executor as a scheduler's step (`sched::ModelStep`, docs/design/runtime.md section 2): several sequences, each with its own room of the context (`add_seqs`), run by the one
//! scheduler that runs the dense Qwen3 engine. A step of one sequence is that sequence's pass as a reply alone runs it (its prompt chunk, or its graph's one-token step). A step of several runs their rows
//! in one pass: the operations that act row by row (the products, norms, activations) on all the rows at once, so each weight is read once for the step, and the operations on a sequence's cache or
//! states (the attention's cache write and read, the recurrent layers) on each sequence's rows by themselves with the kernels and row counts its own pass has. What a row computes never turns
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

/// Paged keys and values: the blocks every sequence holds, each ring's layer group, and each sequence's block tables on the GPU ([slot][group], int32 a BLOCK of tokens).
pub(super) struct Paged {
    pub(super) pages: Pages,
    pub(super) group: Vec<usize>,
    tabs: Vec<Vec<Buf>>,
}

/// The sampler's candidates of a step's rows (`ModelStep::top`): the `kern::TOPK` largest logits of each head row of the last step, made by one launch for all
/// of them at the first ask after the step and copied to the host, where the scheduler would read the whole row (Qwen3's 151,936 logits, Gemma 4's 262,144) to
/// copy, widen and draw from on the host. The dense engine had them in its step (its `candidates`). `rows`: the last step's head rows; `made`: whether its
/// candidates are; `bufs`: the GPU's and the host's room (tokens, logits; for `cap` rows), made at the first ask.
#[derive(Default)]
pub(super) struct Cands {
    bufs: Option<(Buf, Buf, crate::dev::Pinned, usize)>,
    rows: usize,
    made: bool,
    /// Each head row's sum of exponentials made with the step (a launched one's, `ModelStep::draws`): the temperature and the sum.
    sums: Vec<Option<(f32, f64)>>,
}

impl Cands {
    /// A step that wrote `rows` head rows ran: its candidates are not made yet.
    fn fresh(&mut self, rows: usize) {
        (self.rows, self.made) = (rows, false);
        self.sums.clear();
    }
}

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
/// kind's, the latent attention kind's).
pub(super) fn pages_all(d: &super::ModelDesc) -> bool {
    std::iter::once(&d.first).chain(d.layers.iter().flatten()).all(|op| op.rings().is_empty() || matches!(op, Op::Attn(_) | Op::Mla(_)))
        && d.layers.iter().flatten().any(|op| !op.rings().is_empty())
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
            | Op::IdxPick(_)
            | Op::CompAttn { .. }
            | Op::IdxMean { .. }
            | Op::NgramRows(_)
            | Op::NgramAdd(_)
    )
}

impl Paged {
    /// Sequence `slot`'s blocks in group `g`, as the host holds them.
    pub(super) fn group_table(&self, slot: usize, g: usize) -> &[u32] {
        self.pages.table(slot, g)
    }

    /// Sequence `slot`'s table in ring `ring`'s group as the GPU holds it (what the kernels read; the host's may have moved on since the last step).
    pub(super) fn gpu_table(&self, slot: usize, ring: usize) -> Result<Vec<u32>> {
        let b = &self.tabs[slot][self.group[ring]];
        let mut t = vec![0i32; b.bytes / 4];
        b.read(0, &mut t)?;
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
        pg.tabs[slot][pg.group[ring]].ptr
    }

    /// Whether this description's keys and values can be paged (`pages_all`), with no draft model, the whole model on one GPU.
    fn pageable(&self) -> bool {
        self.draft.is_none() && self.stage.first && self.stage.last && pages_all(&self.big.d) && crate::tune!("GLYD_ENGINE_PAGED").is_none_or(|v| v != "0")
    }

    /// The rings' keys and values moved to pools of blocks shared by `users` sequences, as large as the GPU's free memory leaves (`keep` of it kept): a sliding group no more than every
    /// user's window and a chunk, the full groups the rest. Each ring's room in kc and vc is then its group's blocks; the sequences' tables name every entry block 0 until they grow.
    fn page(&mut self, users: usize) -> Result<()> {
        let rings = &self.big.d.rings;
        let mut windows: Vec<usize> = Vec::new();
        let mut group = Vec::with_capacity(rings.len());
        for (r, ring) in rings.iter().enumerate() {
            let w = if self.ring_cap[r] == 0 { usize::MAX } else { ring.window };
            if !windows.contains(&w) {
                windows.push(w);
            }
            group.push(windows.iter().position(|&x| x == w).expect("just added"));
        }
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
        // (the 12-bit pages' step buffers: a prompt chunk's pages unpacked to bf16, keys and values, a ring at a time)
        let unpacked: usize = rings
            .iter()
            .zip(&twelve)
            .filter(|(_, t)| t.is_some())
            .map(|(ring, _)| {
                2 * if ring.window > 0 { (ring.window + self.chunk).div_ceil(BLOCK) + 2 } else { self.ctx.div_ceil(BLOCK) + 1 } * BLOCK * ring.nk * ring.d * 2
            })
            .max()
            .unwrap_or(0);
        let keep: Vec<Option<usize>> = windows.iter().map(|&w| (w > 0 && w != usize::MAX).then_some(w + self.chunk)).collect();
        let most: Vec<usize> = keep.iter().map(|k| users * (k.map_or(self.ctx, |k| k.min(self.ctx)).div_ceil(BLOCK) + 1)).collect();
        // the old rooms given back first
        self.dev.sync()?;
        let graphed = self.graph.take().is_some();
        (self.kc, self.vc) = (Buf::zeroed(4)?, Buf::zeroed(4)?);
        let tab_bytes = users * windows.len() * self.ctx.div_ceil(BLOCK) * 4;
        let (free, _) = self.dev.mem_info()?;
        const KEEP: usize = 512 << 20;
        let budget = (free.saturating_sub(KEEP + tab_bytes + unpacked) as f64 * self.share) as usize;
        // as many blocks in every group (a block of a group is BLOCK tokens of every layer in it), the sliding groups capped and what they leave to the full ones
        let all: usize = per.iter().sum::<usize>().max(1) * BLOCK;
        let even = budget / all;
        let mut blocks: Vec<usize> = most.iter().map(|&m| even.min(m)).collect();
        let left = budget - blocks.iter().zip(&per).map(|(b, p)| b * p * BLOCK).sum::<usize>();
        let full_per: usize = keep.iter().zip(&per).filter(|(k, _)| k.is_none()).map(|(_, p)| p * BLOCK).sum();
        if full_per > 0 {
            let more = left / full_per;
            for (g, k) in keep.iter().enumerate() {
                if k.is_none() {
                    blocks[g] = (blocks[g] + more).min(most[g]);
                }
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
                self.split_rows,
                self.ctx,
                self.chunk,
                self.ldec.as_ref().map(|d| d.splits).ok_or_else(|| crate::ffi::Error("12-bit pages for a model the library's decode does not take".into()))?,
            )?);
        }
        let tabs = (0..users)
            .map(|_| (0..windows.len()).map(|_| Buf::zeroed(self.ctx.div_ceil(BLOCK) * 4)).collect::<Result<Vec<_>>>())
            .collect::<Result<Vec<_>>>()?;
        let counts: Vec<u32> = blocks.iter().map(|&b| (b + 1) as u32).collect();
        self.paged = Some(Paged { pages: Pages::new(&counts, &keep, users, self.ctx), group, tabs });
        // (sequence 0's graph again, on the pools and its tables)
        if graphed {
            self.capture_graph()?;
        }
        Ok(())
    }

    /// The entries of sequence `slot`'s tables changed since they were last copied, copied to the GPU (nothing runs on them then: the steps before have been waited for).
    pub(super) fn upload_tables(&mut self, slot: usize) -> Result<()> {
        let Some(pg) = &mut self.paged else { return Ok(()) };
        for g in 0..pg.pages.groups() {
            if let Some((lo, hi)) = pg.pages.take_dirty(slot, g) {
                let v: Vec<i32> = pg.pages.table(slot, g)[lo..hi].iter().map(|&b| b as i32).collect();
                pg.tabs[slot][g].write(lo * 4, &v)?;
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
        let d = &self.big.d;
        self.draft.is_none()
            && self.stage.first
            && self.stage.last
            && std::iter::once(&d.first).chain(d.layers.iter().flatten()).all(|op| row_wise(op) || by_sequence(op))
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

    /// Sequence `slot` empty again: no tokens, its states zeroed (its cache is overwritten as it runs).
    pub(super) fn empty(&mut self, slot: usize) -> Result<()> {
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
        self.select(slot)?;
        let n = tokens.len();
        if n == 0 || n > self.chunk || self.n + n + 1 > self.ctx {
            return err(format!("a chunk of {n} tokens after {}, where a chunk holds {} and the cache {}", self.n, self.chunk, self.ctx));
        }
        let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        let pp = self.pos.ptr;
        self.start.set(self.n);
        // SAFETY: live buffers of the engine, sized for a chunk.
        unsafe {
            self.embed(tokens)?;
            self.forward(&self.big, n, (pp, pp), last as usize)?;
            if let (true, Some(dr)) = (last, &self.draft) {
                // (the draft model's first input: the hidden state of the prompt's last position, as `prefill` hands it)
                self.kern.copy_row(self.dev.stream, self.big.slot(super::Slot::H), n as i32 - 1, self.big.d.hidden as i32, dr.p.slot(super::Slot::Hid))?;
            }
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
        for &(slot, t, last) in jobs {
            let start = self.seq_n(slot);
            if t.is_empty() || start + t.len() + 1 > self.ctx {
                return err(format!("{} tokens after {start}, where the cache holds {}", t.len(), self.ctx));
            }
            parts.push(self.part_of(slot, tokens.len(), t.len(), start)?);
            pos.extend((0..t.len()).map(|i| (start + i) as i32));
            tokens.extend_from_slice(t);
            if last {
                head_rows.push(tokens.len() - 1);
            }
        }
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        let pp = self.pos.ptr;
        (self.parts, self.head_rows) = (parts, head_rows);
        // SAFETY: live buffers of the engine and of each sequence, the pass's rows within a chunk's.
        let r = unsafe { self.embed(&tokens).and_then(|_| self.forward(&self.big, n, (pp, pp), asked)) };
        self.parts.clear();
        self.head_rows.clear();
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

    /// Ready to be a scheduler's step for up to `users` sequences: the head's room for a row each, and as many more sequences as the GPU's free memory holds (the first is this one's). None where this
    /// engine runs one sequence only (a draft model, a stage of a split model, a family `add_seqs` refuses).
    pub fn serve_rows(&mut self, users: usize) -> Result<Option<usize>> {
        // (Gemma 4, Qwen3.5's layout, which Qwen3.8 dense runs on, Muse Spark, Kimi-K3, GLM-5.3-Flash, Nemotron-H, DeepSeek-V4, Qwen3.8-Flash-Next and the Llama-style dense
        // models on one GPU)
        if self.draft.is_some() && self.stage.first && self.stage.last {
            // (a drafted model: its one sequence, each decode step a drafting round, `ModelStep::draft`)
            self.reset()?;
            self.make_lanes()?;
            return Ok(Some(1));
        }
        if self.draft.is_some()
            || !self.stage.first
            || !self.stage.last
            || !matches!(
                self.big.d.family,
                "Gemma 4"
                    | "Qwen3.5"
                    | "Muse Spark"
                    | "Kimi-K3"
                    | "GLM-5.3-Flash"
                    | "Nemotron-H"
                    | "DeepSeek-V4"
                    | super::dense::FAMILY
                    | super::qwen4x::FAMILY
            )
        {
            return Ok(None);
        }
        if self.seqs.len() > 1 {
            // (made ready already)
            return Ok(Some(self.seqs.len()));
        }
        // (a step's sequences begin empty: what a reply before this left in the one sequence is not a scheduler's, which would find its first slot holding tokens)
        self.reset()?;
        let users = users.clamp(1, self.chunk.min(64));
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
            return Ok(Some(users));
        }
        let per = (self.kc.bytes + self.vc.bytes + states * (1 + marks as usize)).max(1);
        let (free, _) = self.dev.mem_info()?;
        // (room kept for the products' workspaces and the driver)
        const KEEP: usize = 512 << 20;
        let more = ((free.saturating_sub(KEEP + states * marks as usize) as f64 * self.share) as usize / per).min(users - 1);
        if marks {
            self.mark_room(more + 1)?;
        }
        self.add_seqs(more)?;
        self.make_lanes()?;
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
    /// is a ring, whose blocks a mark keeps as any keys').
    /// (not DeepSeek-V4's: its compressed entries are states that grow with the sequence, so a copy at each prompt's end would be a second cache)
    pub(crate) fn can_mark(&self) -> bool {
        !self.states.is_empty() && self.v4.is_none()
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
    /// looks up its embedding's rows by tokens on the GPU (asked once here, of the embedding's layout); and `Gaps` where a step profile is asked for.
    fn make_lanes(&mut self) -> Result<()> {
        if StepProf::on() || super::prof::on() {
            self.gaps = Some(Gaps::new()?);
        }
        let Some(e) = self.big.mats[self.big.d.embed].packed_opt().map(|p| p.pack) else { return Ok(()) };
        if !crate::sched::overlap_asked()
            || !ModelStep::one_step(self)
            || !self.stores().is_empty()
            || self.big.ram.is_some()
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
                        pg.tabs[slot][g].ptr + (lo * 4) as u64,
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
        self.end_expert_step()?;
        // (GLYD_PROF=1: the pass's operations, by its own events; a graph's replay only where no later replay has recorded its events again)
        if super::prof::on() && self.prof.is_some() {
            let ops = match graph {
                true if !later => super::prof::between(&self.gmarks)?,
                true => Vec::new(),
                false => super::prof::between(&evs)?,
            };
            let gpu = ops.iter().map(|x| x.1).sum();
            if !ops.is_empty() {
                super::prof::pass(rows, prompt, &ops, graph, (!graph).then_some(queue_ms));
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
        if masks.iter().any(Option::is_some) && self.masker.is_none() {
            self.masker = crate::kern::MaskGpu::new(v, cap).ok();
        }
        // SAFETY: the pass's head rows of v logits each (n of them, within the head's room); TOPK of each to the buffers made for `cap` rows; the pinned room holds the
        // tokens, then from 4 TOPK cap on the logits, read after the stream has run the copies; the masker's staging was read by a copy the stream has run (a step's
        // candidates are made once its step before landed).
        unsafe {
            self.kern.topk_rows(s, self.big.logits.ptr, v as i32, n as u32, tok, val)?;
            if let Some(g) = self.masker.as_ref().filter(|_| masks.iter().any(Option::is_some)) {
                g.apply(&self.kern, &self.dev, &masks[..masks.len().min(n)], self.big.logits.ptr, tok, val)?;
            }
            crate::dev::copy_d2h_async(host as *mut _, tok, 4 * TOPK * n, s)?;
            crate::dev::copy_d2h_async(host.add(4 * TOPK * cap) as *mut _, val, 4 * TOPK * n, s)?;
        }
        Ok(())
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
            // (one sequence's step as a reply alone takes it, its graph where it has one; several one after another, the last first, each one's logits row moved to its place in the head's rows
            // before the next step writes its own over the first)
            let mut out = vec![None; rows.len()];
            let w = self.big.d.vocab * 2;
            for (j, r) in rows.iter().enumerate().rev() {
                self.select(r.pages[0] as usize)?;
                out[j] = Some(self.step(r.token)?);
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
    fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        let slot = pages[0] as usize;
        if self.seq_n(slot) != start {
            return err(format!("a chunk at {start} of sequence {slot}, which holds {}", self.seq_n(slot)));
        }
        let timed = super::prof::begin();
        let r = self.chunk_alone(slot, tokens, last)?;
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
        self.step_times(p, rows.len(), 0, None);
        self.cands.fresh(rows.len());
        self.end_expert_step()?;
        super::prof::end(timed, rows.len(), 0);
        r
    }
    /// A served model whose description runs the round as one step on one GPU, bf16 pages, no expert cache and no layer in the computer's memory (their host work is in the
    /// pass): `Lanes`, made by `serve_rows`. A recurrent model's states are copied at a mark or a boundary as the next step is launched (the scheduler asks then: `launch`),
    /// and the operations that read the tokens (an n-gram embedding, routing by token) read the ids `step_ids` makes. GLYD_ENGINE_OVERLAP=0: none.
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
    fn masks(&mut self, masks: Vec<Option<crate::constrain::MaskHandle>>) {
        self.row_masks = masks;
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
            if let Some(pg) = &mut self.paged {
                pg.pages.release(p as usize);
                // (a step in flight may still read the tables as they were: the copy is queued after it, and a block given back is written only by the steps after)
                self.upload_tables_queued(p as usize)?;
            }
        }
        Ok(())
    }
    /// A finished sequence stays in its slot for the conversation's next turn. Not with GLYD_ENGINE_KEEP=0 (its slot is emptied at once, as before), nor with
    /// GLYD_ENGINE_BATCH_PREFILL=0, where every reply is to be the reply alone bit for bit: a turn that goes on in a kept slot runs the rest of its prompt in a pass of fewer rows
    /// than the whole prompt's, which is another kernel's (the module's note).
    fn keeps(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_KEEP").is_none_or(|v| v != "0")) && Self::batch_prefill() && self.draft.is_none()
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
            if !m.is_multiple_of(BLOCK) {
                // the block the mark is in: one of `to`'s own, with `from`'s bytes in every ring
                if !pg.pages.grow(t, m) {
                    pg.pages.release(t);
                    return Ok(0);
                }
                let part = m / BLOCK;
                let blocks: Vec<(usize, u32, u32)> =
                    (0..pg.group.len()).map(|r| (r, pg.pages.table(f, pg.group[r])[part], pg.pages.table(t, pg.group[r])[part])).collect();
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
        let (n, chunk) = (self.seq_n(slot as usize), self.chunk);
        let Some(pg) = &mut self.paged else { return Ok(true) };
        if !pg.pages.grow_from(slot as usize, n, upto, chunk) {
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
