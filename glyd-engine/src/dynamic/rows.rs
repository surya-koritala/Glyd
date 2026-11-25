//! The description executor as a scheduler's step (`sched::ModelStep`, docs/design/runtime.md section 2): several sequences, each with its own room of the context (`add_seqs`), run by the one
//! scheduler that runs the dense Qwen3 engine. A step of one sequence is that sequence's pass as a reply alone runs it (its prompt chunk, or its graph's one-token step). A step of several runs their rows
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
/// of them at the first ask after the step and copied to the host, where the scheduler would read the whole row (Qwen3's 151,936 logits, Gemma 4's 262,144) to
/// copy, widen and draw from on the host. The dense engine had them in its step (its `candidates`). `rows`: the last step's head rows; `made`: whether its
/// candidates are; `bufs`: the GPU's and the host's room (tokens, logits; for `cap` rows), made at the first ask.
#[derive(Default)]
pub(super) struct Cands {
    bufs: Option<(Buf, Buf, crate::dev::Pinned, usize)>,
    rows: usize,
    made: bool,
}

impl Cands {
    /// A step that wrote `rows` head rows ran: its candidates are not made yet.
    fn fresh(&mut self, rows: usize) {
        (self.rows, self.made) = (rows, false);
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
            || matches!(op, Op::Attn(_) | Op::Mla(_) | Op::CompKv { .. } | Op::Compress { .. } | Op::IdxPick(_) | Op::CompAttn { .. } | Op::IdxMean { .. })
    }) && d.layers.iter().flatten().any(|op| !op.rings().is_empty())
}

/// The blocks of each layer group's pool, from `budget` bytes: every group as many tokens' blocks (a block of group g is `covers[g]` tokens of each of its rings, `per[g]` bytes a
/// token), none more than `most[g]`; what the groups capped there leave goes to the full groups (`keep[g]` None), as many tokens' again.
pub(super) fn pool_blocks(budget: usize, per: &[usize], covers: &[usize], keep: &[Option<usize>], most: &[usize]) -> Vec<usize> {
    let even = budget / per.iter().sum::<usize>().max(1);
    let mut blocks: Vec<usize> = (0..per.len()).map(|g| (even / covers[g]).min(most[g])).collect();
    let left = budget - (0..per.len()).map(|g| blocks[g] * covers[g] * per[g]).sum::<usize>();
    let full: usize = (0..per.len()).filter(|&g| keep[g].is_none()).map(|g| per[g]).sum();
    if full > 0 {
        let more = left / full;
        for g in (0..per.len()).filter(|&g| keep[g].is_none()) {
            blocks[g] = (blocks[g] + more / covers[g]).min(most[g]);
        }
    }
    blocks
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
                self.knh.conv_rows(s, x, row as i32, p.small[w].ptr, p.small[bias].ptr, sts(state), ss, p.slot(Slot::Xc), ru, cd as i32)?;
                self.mark("Mamba-2: convolution")
            }
            Op::MambaScan { dt_bias, a_log, d, state, cd, row, dt_off, nh, ng } => {
                let (xc, dtr, core) = (p.slot(Slot::Xc), p.slot(Slot::Mix) + (dt_off * 2) as u64, p.slot(Slot::Core));
                let (dtb, al, dd) = (p.small[dt_bias].ptr, p.small[a_log].ptr, p.small[d].ptr);
                self.knh.ssm_rows(s, xc, cd as i32, dtr, row as i32, dtb, al, dd, sts(state), ss, core, ru, nh as i32, ng as i32, 0.0)?;
                self.mark("Mamba-2: scan")
            }
            _ => err(format!("{op:?}: no kernel takes a step's rows that decode, each through its own state")),
        }
    }

    /// Whether this description's keys and values can be paged (`pages_all`), with no draft model, the whole model on one GPU.
    fn pageable(&self) -> bool {
        self.draft.is_none() && self.stage.first && self.stage.last && pages_all(&self.big.d) && crate::tune!("GLYD_ENGINE_PAGED").is_none_or(|v| v != "0")
    }

    /// The rings' keys and values moved to pools of blocks shared by `users` sequences, as large as the GPU's free memory leaves (`keep` of it kept): a sliding group no more than every
    /// user's window and a chunk, the full groups the rest. Each ring's room in kc and vc is then its group's blocks; the sequences' tables name every entry block 0 until they grow. A group is
    /// the rings of one window and one ratio: compressed entries' (`Ring::ratio`) are blocks of BLOCK entries, which cover BLOCK `ratio` tokens.
    fn page(&mut self, users: usize) -> Result<()> {
        let rings = &self.big.d.rings;
        let mut windows: Vec<(usize, usize)> = Vec::new();
        let mut group = Vec::with_capacity(rings.len());
        for (r, ring) in rings.iter().enumerate() {
            let w = (if self.ring_cap[r] == 0 { usize::MAX } else { ring.window }, ring.ratio.max(1));
            if !windows.contains(&w) {
                windows.push(w);
            }
            group.push(windows.iter().position(|&x| x == w).expect("just added"));
        }
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
        // (the 12-bit pages' step buffers: a prompt chunk's pages unpacked to bf16, keys and values (a latent's alone, its row in heads of 128), a ring at a time)
        let unpacked: usize = rings
            .iter()
            .zip(&twelve)
            .filter(|(_, t)| t.is_some())
            .map(|(ring, _)| {
                let blocks = if ring.window > 0 { (ring.window + self.chunk).div_ceil(BLOCK) + 2 } else { self.ctx.div_ceil(BLOCK) + 1 };
                (1 + ring.values as usize) * blocks * BLOCK * ring.nk * ring.d.div_ceil(128) * 128 * 2
            })
            .max()
            .unwrap_or(0);
        let keep: Vec<Option<usize>> = windows.iter().map(|&(w, _)| (w > 0 && w != usize::MAX).then_some(w + self.chunk)).collect();
        let most: Vec<usize> = keep.iter().zip(&covers).map(|(k, &c)| users * (k.map_or(self.ctx, |k| k.min(self.ctx)).div_ceil(c) + 1)).collect();
        // the old rooms given back first
        self.dev.sync()?;
        let graphed = self.graph.take().is_some();
        (self.kc, self.vc) = (Buf::zeroed(4)?, Buf::zeroed(4)?);
        let tab_bytes = users * windows.len() * self.ctx.div_ceil(BLOCK) * 4;
        // (12-bit pages: the raw slabs one sequence alone may take, every page of it raw: `Lossless::alone_slabs`, which the raw pools hold at the least)
        let alone: usize = match self.kv12.as_ref().filter(|_| twelve.iter().any(Option::is_some)) {
            Some(t) => super::kvpages::Lossless::alone_slabs(&self.lib, &self.big.d, t, &self.ring_cap, self.ctx, self.chunk)
                .iter()
                .map(|&(hp, n)| self.lib.kv_raw_bytes(n, hp).unwrap_or(0) as usize)
                .sum(),
            None => 0,
        };
        let (free, _) = self.dev.mem_info()?;
        const KEEP: usize = 512 << 20;
        let budget = (free.saturating_sub(KEEP + tab_bytes + unpacked + alone) as f64 * self.share) as usize;
        // as many tokens' blocks in every group (a block of a group is its `covers` tokens of every layer in it), the sliding groups capped and what they leave to the full ones
        let mut blocks = pool_blocks(budget, &per, &covers, &keep, &most);
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
        // (a step of such rows alone in a graph: every operation one that acts row by row or an attention layer, no recurrent state; the latent attention runs its sequences by
        // their own pointers, so not it)
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
        let pp = self.pos.ptr;
        self.start.set(self.n);
        // SAFETY: live buffers of the engine, sized for a chunk.
        unsafe {
            self.embed(tokens)?;
            self.pieces_pass(n, (pp, pp), last as usize)?;
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
            self.settle_slot(slot)?;
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
        // (the leading rows that decode, a token of a sequence each: their attention one launch, or one call of the library's decode, a layer, each row through its own table
        // (`RowTabs`); a step of nothing else, padded to a graph's rows, that graph's replay (`rows_bucket`): its padding rows at position -1, which write and read no key)
        let room = self.paged.as_ref().and_then(|pg| pg.rows.as_ref()).map_or(0, |rt| rt.rows);
        let lead = jobs.iter().take_while(|j| j.1.len() == 1).count().min(room);
        let graph = if lead == jobs.len() && asked == lead { self.rows_bucket(n) } else { None };
        let rows = graph.unwrap_or(n);
        if let Some(k) = graph {
            pos.resize(k, -1);
            tokens.resize(k, tokens[0]);
            head_rows = (0..k).collect();
        }
        self.dev.sync()?;
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
        let r = unsafe {
            self.embed(&tokens).and_then(|_| match graph {
                Some(k) => self.rows_replay(k, false),
                None if pieces => self.pieces_pass(n, (pp, pp), asked),
                None => self.forward(&self.big, n, (pp, pp), asked),
            })
        };
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
        if self.graphs_off.get()
            || super::tp::comm().is_some()
            || crate::hb_usage::recording()
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
        // (a whole model on one GPU whose every operation keeps a sequence's data in its rings and states, with no draft model)
        if self.draft.is_some() || !self.stage.first || !self.stage.last || one_sequence_op(&self.big.d).is_some() {
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
            self.capture_rows()?;
            self.capture_pieces()?;
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

/// `GLYD_ENGINE_STEP_PROF=1` (a development build's): where a decode step's time goes, every 256 steps on stderr. The call's wall time; the GPU's, from an event at the
/// call's start to one after its work (the time the GPU is busy with the step or waits for the host to queue the rest of it); and the host's between one step's return
/// and the next one's call (the scheduler, the sampler, the server). With a step that waits for its tokens, the GPU is idle for the host's share.
struct StepProf {
    start: crate::dev::Ev,
    at: std::time::Instant,
}

#[derive(Default)]
struct StepSums {
    n: usize,
    rows: usize,
    prompt: usize,
    call: f64,
    gpu: f64,
    host: f64,
    last: Option<std::time::Instant>,
}

thread_local!(static STEP_SUMS: std::cell::RefCell<StepSums> = std::cell::RefCell::new(StepSums::default()));

impl StepProf {
    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::tune!("GLYD_ENGINE_STEP_PROF").is_some_and(|v| v == "1"))
    }

    fn begin(s: crate::ffi::Stream) -> Option<StepProf> {
        if !Self::on() {
            return None;
        }
        let start = crate::dev::Ev::new().ok()?;
        start.record(s).ok()?;
        Some(StepProf { start, at: std::time::Instant::now() })
    }

    fn end(p: Option<StepProf>, s: crate::ffi::Stream, rows: usize, prompt: usize) {
        let Some(p) = p else { return };
        let Ok(stop) = crate::dev::Ev::new() else { return };
        if stop.record(s).is_err() {
            return;
        }
        let gpu = stop.since(&p.start).unwrap_or(f32::NAN) as f64;
        let now = std::time::Instant::now();
        let call = now.duration_since(p.at).as_secs_f64() * 1e3;
        STEP_SUMS.with(|c| {
            let mut c = c.borrow_mut();
            if let Some(last) = c.last {
                c.host += p.at.duration_since(last).as_secs_f64() * 1e3;
            }
            c.last = Some(now);
            (c.n, c.rows, c.prompt, c.call, c.gpu) = (c.n + 1, c.rows + rows, c.prompt + prompt, c.call + call, c.gpu + gpu);
            if c.n == 256 {
                let n = c.n as f64;
                eprintln!(
                    "step profile, 256 steps of {:.1} rows and {:.0} prompt tokens on average: the call {:.3} ms (the GPU from its first work to its last {:.3} ms), the host between steps {:.3} ms",
                    c.rows as f64 / n,
                    c.prompt as f64 / n,
                    c.call / n,
                    c.gpu / n,
                    c.host / (n - 1.0)
                );
                *c = StepSums { last: c.last, ..Default::default() };
            }
        });
    }
}

impl Dyn {
    /// Head row `row`'s candidates of the last step (`Cands`): None where the step wrote no such row, the vocabulary is not one the top-k kernel takes, or
    /// GLYD_ENGINE_HOST_SAMPLING=1 asks for the row on the host.
    fn row_top(&mut self, row: usize) -> Result<Option<Vec<(u32, f32)>>> {
        use crate::kern::TOPK;
        let v = self.big.d.vocab;
        if self.top.off || row >= self.cands.rows || !v.is_multiple_of(8) || v < TOPK {
            return Ok(None);
        }
        let cap = self.head_room().max(self.cands.rows);
        if self.cands.bufs.as_ref().is_none_or(|b| b.3 < cap) {
            self.cands.bufs = Some((Buf::new(4 * TOPK * cap)?, Buf::new(4 * TOPK * cap)?, crate::dev::Pinned::new(8 * TOPK * cap)?, cap));
        }
        let (tok, val, host, cap) = self.cands.bufs.as_ref().expect("made above");
        if !self.cands.made {
            let (n, s, h) = (self.cands.rows, self.dev.stream, host.ptr as *mut u8);
            // SAFETY: the step's head rows of v logits each (n of them, within the head's room); TOPK of each to the buffers made for `cap` rows; the pinned room
            // holds the tokens, then from 4 TOPK cap on the logits; read after the synchronize.
            unsafe {
                self.kern.topk_rows(s, self.big.logits.ptr, v as i32, n as u32, tok.ptr, val.ptr)?;
                crate::dev::copy_d2h_async(h as *mut _, tok.ptr, 4 * TOPK * n, s)?;
                crate::dev::copy_d2h_async(h.add(4 * TOPK * cap) as *mut _, val.ptr, 4 * TOPK * n, s)?;
            }
            self.dev.sync()?;
            self.cands.made = true;
        }
        // SAFETY: the pinned bytes were copied above (the stream synchronized since), TOPK for each row below `rows`.
        Ok(Some(unsafe { crate::kern::read_candidates(host.ptr as *const u8, 0, 4 * TOPK * cap, row) }))
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
        let (p, timed) = (StepProf::begin(self.dev.stream), super::prof::begin());
        let r = self.round_rows(rows, jobs);
        self.kv12_stepped();
        let prompt = jobs.iter().map(|j| j.tokens.len()).sum();
        StepProf::end(p, self.dev.stream, rows.len(), prompt);
        self.cands.fresh(rows.len() + jobs.iter().filter(|j| j.last).count());
        self.end_expert_step()?;
        super::prof::end(timed, rows.len(), prompt);
        r
    }
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        let (p, timed) = (StepProf::begin(self.dev.stream), super::prof::begin());
        let r = self.decode_rows(rows);
        self.kv12_stepped();
        StepProf::end(p, self.dev.stream, rows.len(), 0);
        self.cands.fresh(rows.len());
        self.end_expert_step()?;
        super::prof::end(timed, rows.len(), 0);
        r
    }
    /// The candidates the GPU makes of the step's head rows (`Cands`); where it cannot (an error), the row is read as without them.
    fn top(&mut self, row: usize) -> Option<Vec<(u32, f32)>> {
        self.row_top(row).unwrap_or(None)
    }
    fn expert_stats(&self) -> Vec<crate::offload::LayerStats> {
        Dyn::expert_stats(self)
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let v = self.big.d.vocab;
        let mut words = vec![0u16; v];
        self.big.logits.read(row * v * 2, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }
    fn gather(&mut self, row: usize, tokens: &[u32]) -> Result<Vec<(u32, f32)>> {
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
                // (its 12-bit pages that no one reads again cleared: their chunks and slabs back to the pools, for the sequences that go on)
                let freed = if self.lossless.is_some() { pg.pages.sole_unnamed(p as usize) } else { Vec::new() };
                pg.pages.release(p as usize);
                self.dev.sync()?;
                self.upload_tables(p as usize)?;
                for (g, f) in freed.iter().enumerate() {
                    self.kv12_release_blocks(g, &f.iter().map(|x| x.1).collect::<Vec<u32>>())?;
                }
            }
        }
        Ok(())
    }
    /// A finished sequence stays in its slot for the conversation's next turn. Not with GLYD_ENGINE_KEEP=0 (its slot is emptied at once, as before), nor with
    /// GLYD_ENGINE_BATCH_PREFILL=0, where every reply is to be the reply alone bit for bit: a turn that goes on in a kept slot runs the rest of its prompt in a pass of fewer rows
    /// than the whole prompt's, which is another kernel's (the module's note).
    fn keeps(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_KEEP").is_none_or(|v| v != "0")) && Self::batch_prefill()
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
        if !self.kv12_room(n, upto)? {
            return Ok(false);
        }
        let Some(pg) = &mut self.paged else { return Ok(true) };
        if !pg.pages.grow_from(slot as usize, n, upto, chunk) {
            return Ok(false);
        }
        self.dev.sync()?;
        self.upload_tables(slot as usize)?;
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
    /// through its own state (the gated delta rule's and Mamba-2's: Qwen3.5's and Nemotron-H's layers); Kimi delta attention's operations and the latent attention keep it eager.
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

    /// A row's table is copied as many times as the decode's calls on the group make sequences of a row, every count of them dividing it: the stride `dup / sp` tables
    /// finds row r's j-th sequence's table in row r's copies (Gemma 4's global layers: 16 query heads over one kv head, two sequences a row; its sliding ones one).
    #[test]
    fn a_rows_tables_are_copied_for_every_calls_sequences() {
        let dup = row_dups(3, [(0, 1), (1, 2), (1, 1), (0, 1)].into_iter());
        assert_eq!(dup, [1, 2, 0]);
        assert_eq!(row_dups(1, [(0, 2), (0, 3)].into_iter()), [6]);
        // (sequence b = r sp + j of a call at stride dup / sp tables: row r's copy j dup / sp, within its dup copies)
        let (t, d) = (100usize, 2usize);
        for sp in [1usize, 2] {
            for r in 0..5 {
                for j in 0..sp {
                    let at = (r * sp + j) * (d / sp) * t;
                    assert!(at >= r * d * t && at < (r + 1) * d * t, "sp {sp} row {r} part {j}");
                }
            }
        }
    }

    /// The pools' blocks from a budget: groups of BLOCK tokens get as many blocks each (the sliding ones capped, what they leave to the full ones, as before groups of larger
    /// blocks), and a group whose blocks cover more tokens (compressed entries') as many tokens' worth: a fourth of the blocks where a block covers four times the tokens.
    #[test]
    fn every_group_holds_as_many_tokens() {
        // (bytes a token: a full group 2,048, a sliding one 1,024; 60 MiB)
        let budget = 60 << 20;
        let (per, keep) = ([2048, 1024], [None, Some(1024 + 256)]);
        let b = pool_blocks(budget, &per, &[BLOCK, BLOCK], &keep, &[usize::MAX, 4 * 81]);
        // (as many blocks a group: 60 MiB / (3,072 B x 16) = 1,280, the sliding group capped at 324 and its rest to the full group)
        assert_eq!(b, vec![1280 + (1280 - 324) * 1024 / 2048, 324]);
        assert!(b[0] * BLOCK * 2048 + b[1] * BLOCK * 1024 <= budget);
        // a compressed group of 4 tokens an entry (256 bytes a token) beside them: its blocks hold 64 tokens, so a fourth as many hold as many tokens
        let b = pool_blocks(budget, &[2048, 1024, 256], &[BLOCK, BLOCK, 4 * BLOCK], &[None, Some(1024 + 256), None], &[usize::MAX, 4 * 81, usize::MAX]);
        // (18,904 tokens each: 1,181 blocks, 324, 295; the 6,108 tokens' bytes the sliding group leaves to the two full ones: 381 blocks and 95)
        assert_eq!(b, vec![1181 + 381, 324, 295 + 95]);
        assert!(b[0] * BLOCK - b[2] * 4 * BLOCK < 4 * BLOCK, "the full groups hold as many tokens, to a block");
        assert!(b[0] * BLOCK * 2048 + b[1] * BLOCK * 1024 + b[2] * 4 * BLOCK * 256 <= budget);
    }
}
