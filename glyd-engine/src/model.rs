//! The Qwen3 forward pass on the GPU: prefill in chunks (eager) and a decode step (one CUDA graph), greedy sampling.
//!
//! A step's layer (M tokens): norm (+ the residual add) -> qkv product -> query/key norm and rotary -> KV write ->
//! attention over the lossless pages -> o product -> add and norm -> gate/up product -> silu * up -> down product.
//! Attention in prefill is the library's decode kernel with a row for every token of the chunk, all reading the same
//! pages with lengths start + 1 ... start + n.

use crate::dev::{copy_d2h_async, copy_h2d_async, Buf, Dev, Ev, Graph, Pinned};
use crate::ffi::{err, Lib, Result};
use crate::kern::{Kern, TOPK};
use crate::kv::{self, Kv, KvTables, Lost, HEAD_WORDS};
use crate::sampler::{Need, Top};
use crate::weights::{PackBuf, Weights};
use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

/// What a calibrating forward pass calls at every layer, after the layer's keys and values are made: (the engine, the layer, the tokens).
pub type Hook<'a> = &'a mut dyn FnMut(&Engine, usize, usize) -> Result<()>;

pub struct Opts {
    /// The longest context (prompt and generated tokens) the cache holds, where `util` is not set.
    pub ctx: usize,
    /// Fill the GPU's memory up to this share with the KV cache instead (the cache's size is then what is left).
    pub util: Option<f64>,
    /// Tokens a prefill chunk takes at most (0: 4096 where cuBLAS is there, whose route decodes a matrix once a chunk, else 2048).
    pub chunk: usize,
    pub graph: bool,
    pub device: i32,
    /// The KV tables' file, where it is not kv-tables.safetensors in the model's directory.
    pub tables: Option<std::path::PathBuf>,
    /// The layout a Hub-format directory is packed in when it is packed here (None: the GPU's, `best_layout`'s).
    pub layout: Option<crate::weights::Layout>,
    /// The prompt's attention through the library's decode kernel, a row for every token (for checking the flash kernel against it), where false: the flash kernel on a bf16 staging cache.
    pub attn_decode: bool,
    /// Tokens the cache holds for all the sequences together, where it holds more than one window (0: the window, `ctx`).
    pub pool: usize,
    /// Sequences a decode step takes at most, 1 to MAX_ROWS (0: the batch's way is not set up; one sequence's, with `step`, is).
    pub users: usize,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts { ctx: 8192, util: None, chunk: 0, graph: true, device: 0, tables: None, layout: None, attn_decode: false, pool: 0, users: 0 }
    }
}

/// Values of the cache compared with a bf16 copy of every write (GLYD_ENGINE_VERIFY=1), and those that differed; and the values of pages the cache had named lost at that time (their holders were taken back from: not compared).
pub static VERIFY_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VERIFY_DIFFER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VERIFY_LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Where a decode step's time goes, summed over the steps (ns): the host's work before the GPU has the step (copies, the embedding, the launch), the wait for the GPU to finish it (its work and the copies back), and the host's work after.
pub static DECODE_PREP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DECODE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DECODE_POST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The GPU's own time for the decode graphs (events around the launch), summed (ns).
pub static DECODE_GPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The differing values by head and by key (even) or value (odd), for finding where a difference lies.
pub static VERIFY_BY: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// A bf16 copy of every layer's cache (the same slots), to check what the cache gives back against what was written.
struct Verify {
    shadow: Vec<Buf>,
    tmp: Buf,
    count: Buf,
}

/// The scheduler's types (src/sched.rs owns them; this engine is one `ModelStep` of several).
pub use crate::sched::{Finish, PrefillJob, Reply, Row, Sample};

/// Sequences a decode step takes at most.
pub const MAX_ROWS: usize = 8;

/// The most rows of a product the library's own route takes whatever the GPU (`decoded_way`: past it a product may take the decoded matrix's way). A prompt chunk of at most this many tokens is run with others in one pass
/// (a longer one is its own: it is compute-bound, and the weights are read once for it). Where the pass has more rows than this, its products take the route their row count gives, which is not the route each chunk's alone
/// took: the logits differ in the last bits (a few hundredths of a logprob at most), as they do between any two routes.
pub const FUSED_ROWS: usize = 64;

/// The pages `start` .. `start + n` (a sequence's room, for the checks that give a sequence pages of their own).
pub fn first_pages(start: usize, n: usize) -> Vec<i32> {
    (start..start + n).map(|p| p as i32).collect()
}
/// Where, in the pinned staging buffer, a decode step's block-table rows begin (the first bytes serve the single-sequence step and the samples).
const HOST_BATCH: usize = 8192;
/// Where the cache's status header is copied to after a step.
const HOST_HEAD: usize = 1024;
/// Where the candidates of each row of a step are copied to (the token indices, then their logits: TOPK of each a row).
const HOST_TOPK_TOK: usize = 2048;
const HOST_TOPK_VAL: usize = 4096;
/// Where a decode step's per-row arrays lie in `bstate` (positions, lengths: int32 [MAX_ROWS]; slots: int64 [MAX_ROWS]).
const B_POS: u64 = 0;
const B_LENS: u64 = 32;
const B_SLOTS: u64 = 64;

/// What follows the layers: nothing, the LM head and argmax on the last row (a prompt's end, one sequence's step), or on every row (a step of several sequences).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Head {
    None,
    Last,
    All,
    /// The rows `Engine::picked` names (the last rows of the sequences of a prompt batch), in that order.
    Rows,
}

/// One sequence's part of a prompt batch (`Engine::prefill_batch`): its tokens are rows `off` .. `off + n` of the batch, they start at position `start` of the sequence, the pages before them (`earlier` of them) are at
/// `gather_off` in the gather list.
#[derive(Clone, Copy)]
struct Part {
    off: usize,
    n: usize,
    start: usize,
    earlier: usize,
    gather_off: usize,
}

/// A prompt chunk's attention: the chunk starts at `start` in its sequence; the earlier pages are the list at `gather` (int32 on the device), the chunk's own keys go to the staging cache at the slots `lslots`.
#[derive(Clone, Copy)]
pub(crate) struct Flash {
    pub start: usize,
    pub gather: u64,
    pub lslots: u64,
}

/// Row `row`'s TOPK candidates (token, logit) from the pinned bytes at `p`: the token indices (int32) from byte `tok` on and the logits (f32) from byte `val` on, each row's TOPK after the row before's.
///
/// # Safety
/// `p` holds TOPK (row + 1) tokens at `tok` and as many logits at `val`, aligned for 4-byte reads.
unsafe fn read_candidates(p: *const u8, tok: usize, val: usize, row: usize) -> Vec<(u32, f32)> {
    (0..TOPK).map(|j| (*(p.add(tok + 4 * (row * TOPK + j)) as *const i32) as u32, *(p.add(val + 4 * (row * TOPK + j)) as *const f32))).collect()
}

/// The grammar masks of a step's rows on the GPU: `apply` waits for each row's mask (the workers make them beside the step), copies them to the device and, behind the step's work on the stream, makes the step's candidates again with
/// the masked top-k kernel, over the unmasked ones the step made (a graph's, whose kernel is fixed); the host then draws from candidates that hold the allowed tokens (`Sampler::sample_grammar`). No mask for a row: the row is as the step made it.
pub struct MaskGpu {
    dev: Buf,
    stage: Pinned,
    vocab: usize,
    /// A row's words: its flag and its V / 32 words of bits.
    words: usize,
    rows: usize,
}

impl MaskGpu {
    pub fn new(vocab: usize, rows: usize) -> Result<MaskGpu> {
        let words = 1 + vocab.div_ceil(32);
        Ok(MaskGpu { dev: Buf::zeroed(4 * words * rows)?, stage: Pinned::new(4 * words * rows)?, vocab, words, rows })
    }

    /// Whether the masks of rows can be put on the GPU (the kernels have the masked top-k, and take the vocabulary).
    pub fn can(&self, kern: &Kern) -> bool {
        kern.has_masked_topk() && self.vocab.is_multiple_of(8) && self.vocab >= TOPK
    }

    /// For the rows `masks` names (None: no mask): the masked candidates of every row into `tok` and `val` (as many rows as there are masks), after the work queued on the stream. Nothing is done where no row has a mask
    /// or a row's worker died (that row keeps its unmasked candidates, which the host's draw checks against the mask as it always did). Returns whether it made any.
    ///
    /// # Safety
    /// `logits` holds a row of the vocabulary for each of `masks`, `tok` and `val` TOPK of each, all live and on the engine's stream; the stream was synchronized since the last call (the staging bytes are free).
    pub unsafe fn apply(&self, kern: &Kern, dev: &Dev, masks: &[Option<crate::constrain::MaskHandle>], logits: u64, tok: u64, val: u64) -> Result<bool> {
        if masks.is_empty() || masks.len() > self.rows || !self.can(kern) {
            return Ok(false);
        }
        let stage = std::slice::from_raw_parts_mut(self.stage.ptr as *mut u32, self.words * masks.len());
        let mut any = false;
        for (r, m) in masks.iter().enumerate() {
            let row = &mut stage[r * self.words..(r + 1) * self.words];
            row.fill(0);
            if let Some(vob) = m.as_ref().and_then(|m| m.words()).filter(|v| v.len() == self.vocab) {
                row[1..1 + vob.as_slice().len().min(self.words - 1)].copy_from_slice(&vob.as_slice()[..vob.as_slice().len().min(self.words - 1)]);
                row[0] = 1;
                any = true;
            }
        }
        if !any {
            return Ok(false);
        }
        copy_h2d_async(self.dev.ptr, self.stage.ptr as *const _, 4 * self.words * masks.len(), dev.stream)?;
        kern.topk_rows_masked(dev.stream, logits, self.vocab as i32, masks.len() as u32, tok, val, self.dev.ptr, self.words as i32)?;
        Ok(true)
    }
}

/// The GPU's side of the sampler's candidates for an engine with one row of logits (Qwen3.5's and Nemotron 3 Super's, whose `Gen::reply` draws on the host): behind a step that `ask` names (`Sampler::top_need`), `queue` makes the row's
/// 64 largest logits and, for a top-p with no top-k, the sum of its exponentials, and copies them to the host in the step's one synchronize; `read` gives them. A draw then reads 520 bytes in place of the row (131,072 to
/// 248,000 logits to copy, sort and exponentiate on the host). `off` (GLYD_ENGINE_HOST_SAMPLING=1, to compare): none are made and the row is read.
pub struct TopGpu {
    tok: Buf,
    val: Buf,
    sum: Buf,
    stage: Pinned,
    vocab: usize,
    /// What the next step is asked for.
    pub ask: Option<Need>,
    /// What the last `queue` made.
    got: Option<Need>,
    pub off: bool,
    /// The mask of the next step's grammar (`Sampler::mask_handle`), and where it is put on the GPU.
    pub mask: Option<crate::constrain::MaskHandle>,
    masker: Option<MaskGpu>,
}

impl TopGpu {
    pub fn new(vocab: usize) -> Result<TopGpu> {
        Ok(TopGpu {
            tok: Buf::zeroed(4 * TOPK)?,
            val: Buf::zeroed(4 * TOPK)?,
            sum: Buf::zeroed(8)?,
            stage: Pinned::new(8 * TOPK + 8)?,
            vocab,
            ask: None,
            got: None,
            off: crate::tune!("GLYD_ENGINE_HOST_SAMPLING").is_some(),
            mask: None,
            masker: MaskGpu::new(vocab, 1).ok(),
        })
    }

    /// Behind the step's work on the stream: the candidates of the row `ask` names, with the copies to the host (the caller synchronizes). The kernels take a vocabulary of a multiple of 8 and at least TOPK; a smaller one has none.
    ///
    /// # Safety
    /// `logits` is a row of the vocabulary's size and `maxv` its largest (f32), both live and made on the engine's stream.
    pub unsafe fn queue(&mut self, kern: &Kern, dev: &Dev, logits: u64, maxv: u64) -> Result<()> {
        self.got = self.ask.filter(|_| !self.off && self.vocab.is_multiple_of(8) && self.vocab >= TOPK);
        let Some(need) = self.got else { return Ok(()) };
        let (s, host) = (dev.stream, self.stage.ptr as *mut u8);
        kern.topk_rows(s, logits, self.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        // (a grammar's mask: the candidates made again with it, the mask put on the row before the top-k)
        if let (Some(m), Some(g)) = (self.mask.take(), &self.masker) {
            g.apply(kern, dev, &[Some(m)], logits, self.tok.ptr, self.val.ptr)?;
        }
        copy_d2h_async(host as *mut _, self.tok.ptr, 4 * TOPK, s)?;
        copy_d2h_async(host.add(4 * TOPK) as *mut _, self.val.ptr, 4 * TOPK, s)?;
        if need.mass {
            kern.exp_sum_rows(s, logits, self.vocab as i32, 1, maxv, need.temperature, self.sum.ptr)?;
            copy_d2h_async(host.add(8 * TOPK) as *mut _, self.sum.ptr, 8, s)?;
        }
        Ok(())
    }

    /// What the last `queue` made (the stream synchronized since): None where it made nothing.
    pub fn read(&self) -> Option<Top> {
        let need = self.got?;
        let p = self.stage.ptr as *const u8;
        // SAFETY: the pinned bytes were copied behind the step (the stream synchronized since): TOPK tokens (int32), TOPK logits (f32), then the sum (f64) where asked.
        unsafe { Some(Top { cands: read_candidates(p, 0, 4 * TOPK, 0), mass: need.mass.then(|| *(p.add(8 * TOPK) as *const f64)) }) }
    }
}

/// 64-row groups of the embedding decoded at once for a prompt chunk.
const EMBED_GROUPS: usize = 32;

/// The 64-row groups of the embedding that `tokens` fall in, each once and in order, and for each batch of `per` groups the pair (destination row, row among the batch's decoded groups) of every token
/// whose group is in it.
pub fn embed_plan(tokens: &[u32], per: usize) -> (Vec<i64>, Vec<Vec<(i32, i32)>>) {
    let mut groups: Vec<i64> = tokens.iter().map(|&t| t as i64 / 64).collect();
    groups.sort_unstable();
    groups.dedup();
    let at: std::collections::HashMap<i64, usize> = groups.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let mut by_batch: Vec<Vec<(i32, i32)>> = vec![Vec::new(); groups.len().div_ceil(per)];
    for (i, &t) in tokens.iter().enumerate() {
        let g = at[&(t as i64 / 64)];
        by_batch[g / per].push((i as i32, (g % per * 64) as i32 + (t % 64) as i32));
    }
    (groups, by_batch)
}

pub struct Engine {
    pub dev: Dev,
    pub lib: Lib,
    pub(crate) kern: Kern,
    pub w: Weights,
    pub kv: Kv,
    pub(crate) chunk: usize,
    // activations
    pub(crate) x: Buf,
    pub(crate) h: Buf,
    pub(crate) qkv: Buf,
    pub(crate) qo: Buf,
    pub(crate) ko: Buf,
    pub(crate) attn: Buf,
    pub(crate) o: Buf,
    pub(crate) gu: Buf,
    pub(crate) m: Buf,
    pub(crate) d: Buf,
    hlast: Buf,
    logits: Buf,
    scratch: Buf,
    /// Several 64-row groups of the embedding decoded at once, and the (destination row, source row) pairs that gather them into x (a prompt chunk's embedding).
    embed_groups: Buf,
    embed_pairs: Buf,
    tok: Buf,
    val: Buf,
    /// The TOPK largest logits of each row of the last step and their token indices (the sampler's candidates).
    topk_tok: Buf,
    topk_val: Buf,
    /// The step's grammar masks, one for each row that has a sample drawn after the step (`ModelStep::masks`), and where they are put on the GPU.
    masks: RefCell<Vec<Option<crate::constrain::MaskHandle>>>,
    maskgpu: Option<MaskGpu>,
    lse: Buf,
    /// NVIDIA's cuBLAS where the machine has it, the scratch it multiplies a decoded matrix from (the largest matrix in bf16), and the GPU's code for the library's routes.
    blas: Option<crate::ffi::Blas>,
    /// The engine's own dense product for the prompt's long products (a matrix decoded to bf16 into `scratch_w`, then kernels/engine.cu's gemm3), where cuBLAS is not used.
    own: bool,
    /// Tokens from which a product takes the decoded matrix's way whatever the library's route table says (GLYD_ENGINE_DECODE_FROM: to see where the crossover lies).
    decode_from: Option<usize>,
    scratch_w: Buf,
    gpu_code: i64,
    pub(crate) pos: Buf,
    pub(crate) slots: Buf,
    lens: Buf,
    st: Buf,
    ws: Buf,
    done: Buf,
    kvws: Buf,
    /// The bf16 staging cache a prompt chunk's attention reads (the earlier pages decoded into it, the chunk's own keys and values put after) and the page numbers 0, 1, ... for gathering them.
    stage: Buf,
    stage_blocks: i64,
    iota: Buf,
    attn_decode: bool,
    splits_dec: i64,
    graph: Option<Graph>,
    /// The decode steps of several sequences: the per-row arrays a graph reads, the block-table rows, a graph for every number of rows, the pages to clear, and what a prompt's chunk needs besides the staging cache.
    bstate: Buf,
    btab: Buf,
    bt_stride: i64,
    graphs: Vec<Option<Graph>>,
    lslots: Buf,
    gpages: Buf,
    rel: Buf,
    steps: u64,
    lazy_release: bool,
    verify: Option<Verify>,
    /// What the cache lost or made heavy in the steps since the scheduler last asked (`take_lost`).
    lost_acc: Lost,
    /// Events around a decode graph's launch.
    graph_ev: (Ev, Ev),
    /// The sampler's candidates are not made on the GPU: a draw reads the row's logits to the host (GLYD_ENGINE_HOST_SAMPLING=1, to compare).
    host_sampling: bool,
    /// The longest sequence (the context asked for, to a whole page); the cache may hold more tokens for all the sequences together.
    window: usize,
    host: Pinned,
    /// Tokens in the cache.
    pub n: usize,
    /// The tokens whose keys and values are in the cache, in order (as many as `n`): what a next prompt shares a prefix with.
    cached: Vec<u32>,
    /// Device bytes of the activations and workspaces (not weights, not the cache).
    pub work_bytes: usize,
    /// A step's profile: an event after every launch, labelled (set by `profile_step`).
    marks: RefCell<Option<Vec<(&'static str, Ev)>>>,
    /// The sequences of the prompt batch being run (empty: one sequence's chunk, as `Flash` says), and the rows whose logits it is to give.
    parts: RefCell<Vec<Part>>,
    picked: RefCell<Vec<usize>>,
    /// A row of the chunk being run whose residual stream is dumped after every layer (debugging: GLYD_ENGINE_DUMP=position, in `score`).
    dump_row: std::cell::Cell<Option<usize>>,
    dumped: RefCell<Vec<Vec<u16>>>,
}

fn mib(b: usize) -> f64 {
    b as f64 / (1 << 20) as f64
}

impl Engine {
    pub fn new(dir: &Path, o: &Opts) -> Result<Engine> {
        let tables = KvTables::load(&o.tables.clone().unwrap_or_else(|| dir.join("kv-tables.safetensors")))?;
        Engine::build(dir, o, Some(tables))
    }

    /// An engine with the weights and the activations but no KV cache, for the calibration that makes the cache's tables (src/calib.rs).
    pub fn for_calibration(dir: &Path, o: &Opts) -> Result<Engine> {
        Engine::build(dir, o, None)
    }

    fn build(dir: &Path, o: &Opts, tables: Option<KvTables>) -> Result<Engine> {
        let t0 = Instant::now();
        let timing = crate::tune!("GLYD_ENGINE_TIMING").is_some();
        let lap = |what: &str| {
            if timing {
                eprintln!("  {:6.2} s  {what}", t0.elapsed().as_secs_f64());
            }
        };
        let dev = Dev::new(o.device)?;
        let lib = Lib::find()?;
        let kern = Kern::load(&dev)?;
        lap("device, library and kernels");
        let (free0, total) = dev.mem_info()?;
        let max_pos = if o.util.is_some() { None } else { Some(o.ctx.max(1) + 16) };
        let w = if dir.join("glyd.json").exists() {
            Weights::load(dir, max_pos, dev.stream)?
        } else {
            // a Hub-format directory: packed here, in memory, in the layout asked for or the GPU's
            let layout = crate::weights::choose_layout(o.layout, dev.cc_pair(), dev.total_mem as u64, crate::weights::Config::load(dir)?.matrix_bytes(), 0).0;
            let threads = crate::tune!("GLYD_ENGINE_THREADS")
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get().min(16)));
            Weights::load_bf16(dir, max_pos, threads, layout)?
        };
        lap("weights uploaded");
        let cfg = w.cfg.clone();
        let (hid, kvh, hd, inter) = (cfg.hidden, cfg.kv_heads, cfg.head_dim, cfg.inter);
        if let Some(tables) = tables.as_ref().filter(|t| o.util.is_none() && t.positions > 0 && o.ctx as u64 > t.positions) {
            return err(format!("--ctx {}, where this model is set up for {} positions: ask for less", o.ctx, tables.positions));
        }
        if hd != 128 {
            return err(format!("head size {hd}: not supported"));
        }
        // the long products: GLYD_ENGINE_GEMM=blas (cuBLAS where the machine has it), own (the engine's own kernels), lib (the library's routes); unset: the engine's own on Ada, else cuBLAS where it is, else the engine's own on Ampere and Hopper
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let blas = if (o.chunk == 0 || o.chunk > 64) && (want == "blas" || (want.is_empty() && dev.cc != (8, 9))) { crate::ffi::Blas::find() } else { None };
        let own = want == "own" || (want.is_empty() && blas.is_none() && matches!(dev.cc.0, 8 | 9));
        let decode_from = crate::tune!("GLYD_ENGINE_DECODE_FROM").and_then(|v| v.parse().ok());
        let c = if o.chunk > 0 {
            o.chunk
        } else if blas.is_some() || own {
            4096
        } else {
            2048
        };
        let qkv_w = (cfg.heads + 2 * kvh) * hd;
        let act = |cols: usize, rows: usize| Buf::new(rows * cols * 2);
        let (x, h, qkv) = (act(hid, c)?, act(hid, c)?, act(qkv_w, c)?);
        let (qo, ko, attn) = (act(cfg.heads * hd, c)?, act(kvh * hd, c)?, act(cfg.heads * hd, c)?);
        let (o_, gu, m, d) = (act(hid, c)?, act(2 * inter, c)?, act(inter, c)?, act(hid, c)?);
        let hlast = act(hid, MAX_ROWS)?;
        let logits = act(cfg.vocab, MAX_ROWS)?;
        let scratch = Buf::new(64 * hid * 2)?;
        let embed_groups = Buf::new(EMBED_GROUPS * 64 * hid * 2)?;
        let embed_pairs = Buf::new(c * 8)?;
        let (tok, val, lse) = (Buf::zeroed(4 * MAX_ROWS)?, Buf::zeroed(4 * MAX_ROWS)?, Buf::zeroed(4)?);
        let (topk_tok, topk_val) = (Buf::zeroed(4 * MAX_ROWS * TOPK)?, Buf::zeroed(4 * MAX_ROWS * TOPK)?);
        let (pos, slots, lens) = (Buf::zeroed(4 * c)?, Buf::zeroed(8 * c)?, Buf::zeroed(4 * c)?);
        let st = Buf::zeroed(16)?;
        // a long prompt's products by the library's routes: where it says DECODE (W decoded to bf16, then the caller's GEMM), cuBLAS where the machine has it
        let gpu_code = lib.gpu()? as i64;
        let scratch_w = match (&blas, own) {
            (Some(_), _) | (_, true) => Buf::new(
                w.layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gate_up, &l.down]).map(|p| (p.pack.rows * p.pack.cols) as usize * 2).max().unwrap_or(16),
            )?,
            _ => Buf::new(16)?,
        };
        // products' workspace: the most any matrix takes for any token count up to a chunk
        let shapes: Vec<(i64, i64)> = vec![
            (qkv_w as i64, hid as i64),
            (hid as i64, (cfg.heads * hd) as i64),
            (2 * inter as i64, hid as i64),
            (hid as i64, inter as i64),
            (cfg.vocab as i64, hid as i64),
        ];
        // (a pack is in the layout that was smaller for its matrix: the workspace is the most of the two layouts')
        let mut need = 0usize;
        for m_ in 1..=c as i64 {
            for &(oo, kk) in &shapes[..4] {
                for twelve in [false, true] {
                    need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
                }
            }
        }
        for &(oo, kk) in &shapes[4..] {
            for m_ in 1..=MAX_ROWS as i64 {
                for twelve in [false, true] {
                    need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
                }
            }
        }
        let ws = Buf::zeroed(need)?;
        let done = Buf::zeroed(4 << 18)?;
        lap("product workspaces");
        // the cache: a share of what is left, or the context asked for
        let group = cfg.heads / kvh;
        let max_pos = o.ctx.max(1);
        let work_so_far = free0.saturating_sub(dev.mem_info()?.0) - w.bytes;
        let per_token = match &tables {
            Some(t) => kv::bytes_per_token(&lib, t)?,
            None => 1.0,
        };
        let tokens = match o.util {
            Some(u) => {
                let (free, _) = dev.mem_info()?;
                let used = total - free;
                let budget = (u * total as f64) as usize;
                let room = budget.saturating_sub(used).saturating_sub(64 << 20);
                (room as f64 / per_token) as usize / 16 * 16
            }
            None => max_pos,
        };
        if tokens < 16 && tables.is_some() {
            return Err(crate::ffi::Error::of(crate::ffi::Kind::OutOfMemory, "no room for a KV cache"));
        }
        let max_pages = tokens.div_ceil(16) as i64;
        // attention workspace for decode (one row) and, where a chunk's rows go through the decode kernel, for a chunk's rows, at the most splits any of them gets
        let mut kvneed = 0usize;
        let mut splits_dec = 0;
        for b in 1..=(if o.attn_decode { c as i64 } else { 1 }) {
            let s = lib.kv_decode_splits(b, kvh as i64, max_pages)?;
            if b == 1 {
                splits_dec = s;
            }
            kvneed = kvneed.max(lib.kv_decode_workspace(b, kvh as i64, group as i64, hd as i64, s)?);
        }
        // (and for a decode step of up to MAX_ROWS sequences, all at the one number of splits: a sequence's attention is the same whatever it is batched with)
        for b in 1..=MAX_ROWS as i64 {
            kvneed = kvneed.max(lib.kv_decode_workspace(b, kvh as i64, group as i64, hd as i64, splits_dec)?);
        }
        let kvws = Buf::zeroed(kvneed)?;
        // the staging cache of a prompt's attention: the longest sequence the tables cover (or the context asked for), a layer at a time
        let positions = tables.as_ref().map_or(0, |t| t.positions as usize);
        let stage_tokens = if positions > 0 { positions } else { o.ctx.max(1) }.max(c).div_ceil(16) * 16;
        let stage_blocks = (stage_tokens / 16) as i64;
        let stage = if o.attn_decode || tables.is_none() { Buf::new(16)? } else { Buf::new(stage_tokens * kvh * 256 * 2)? };
        let iota = Buf::new(4 * stage_blocks as usize)?;
        iota.write(0, &(0..stage_blocks as i32).collect::<Vec<i32>>())?;
        lap("attention workspace");
        // the cache itself
        let tokens = if o.util.is_some() {
            // (the workspace above was sized for the most pages the budget allowed; the cache may have less)
            let (free, _) = dev.mem_info()?;
            let used = total - free;
            let room = ((o.util.unwrap_or(0.9) * total as f64) as usize).saturating_sub(used).saturating_sub(64 << 20);
            (room as f64 / per_token) as usize / 16 * 16
        } else {
            tokens
        };
        let window = tokens.div_ceil(16) * 16;
        // the room for several sequences: what was asked for, less where the GPU has not the memory (a reserve stays for the graphs and the steps' buffers)
        let pool_tokens = if o.util.is_some() {
            tokens
        } else {
            let want = o.pool.max(tokens).div_ceil(16) * 16;
            let (free, _) = dev.mem_info()?;
            let fits = (free.saturating_sub(768 << 20) as f64 / per_token) as usize / 16 * 16;
            if tables.is_some() {
                want.min(fits).max(window)
            } else {
                window
            }
        };
        let kv = match &tables {
            Some(t) => Kv::new(&lib, t, kvh, group, cfg.layers, pool_tokens, kv::ring_for(c))?,
            None => Kv::empty(kvh, group, cfg.layers)?,
        };
        lap("KV cache");
        let work_bytes = work_so_far + kvws.bytes + stage.bytes + iota.bytes;
        let bt_stride = (window / 16) as i64;
        let host = Pinned::new(HOST_BATCH + MAX_ROWS * bt_stride as usize * 4)?;
        let (bstate, btab) = (Buf::zeroed(128)?, Buf::zeroed(MAX_ROWS * bt_stride as usize * 4)?);
        let (lslots, gpages, rel) = (Buf::zeroed(8 * c)?, Buf::zeroed(4 * stage_blocks as usize * MAX_ROWS)?, Buf::zeroed(4 * kv.nblocks.max(1) as usize)?);
        let mut e = Engine {
            dev,
            lib,
            kern,
            w,
            kv,
            chunk: c,
            x,
            h,
            qkv,
            qo,
            ko,
            attn,
            o: o_,
            gu,
            m,
            d,
            hlast,
            logits,
            scratch,
            tok,
            val,
            topk_tok,
            topk_val,
            masks: RefCell::new(Vec::new()),
            maskgpu: MaskGpu::new(cfg.vocab, MAX_ROWS).ok(),
            lse,
            pos,
            slots,
            lens,
            st,
            ws,
            done,
            kvws,
            stage,
            stage_blocks,
            iota,
            attn_decode: o.attn_decode,
            splits_dec,
            graph: None,
            bstate,
            btab,
            bt_stride,
            graphs: (0..=MAX_ROWS).map(|_| None).collect(),
            lslots,
            gpages,
            rel,
            steps: 0,
            lazy_release: crate::tune!("GLYD_ENGINE_LAZY_RELEASE").is_some(),
            verify: None,
            lost_acc: Lost::default(),
            graph_ev: (Ev::new()?, Ev::new()?),
            host_sampling: crate::tune!("GLYD_ENGINE_HOST_SAMPLING").is_some(),
            window,
            blas,
            own,
            decode_from,
            scratch_w,
            embed_groups,
            embed_pairs,
            gpu_code,
            host,
            n: 0,
            cached: Vec::new(),
            work_bytes,
            marks: RefCell::new(None),
            parts: RefCell::new(Vec::new()),
            picked: RefCell::new(Vec::new()),
            dump_row: std::cell::Cell::new(None),
            dumped: RefCell::new(Vec::new()),
        };
        if crate::tune!("GLYD_ENGINE_VERIFY").is_some() && tables.is_some() {
            let per_layer = e.kv.nblocks as usize * 16 * kvh * 256 * 2;
            let shadow = (0..cfg.layers).map(|_| Buf::zeroed(per_layer)).collect::<Result<Vec<_>>>()?;
            e.verify = Some(Verify { shadow, tmp: Buf::new(64 * 16 * kvh * 256 * 2)?, count: Buf::zeroed(17 * 8)? });
        }
        if o.graph && tables.is_some() {
            e.capture_graph()?;
            if o.users > 0 {
                e.capture_graphs(o.users.min(MAX_ROWS))?;
            }
        }
        lap("decode graph");
        Ok(e)
    }

    pub fn describe(&self) -> String {
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        format!(
            "{} (sm_{}{}): weights {:.2} GiB ({:.1}% of bf16's {:.2}), cache {} tokens ({:.2} GiB, {} blocks a layer), workspaces {:.0} MiB, {:.2} of {:.2} GiB in use",
            self.dev.name,
            self.dev.cc.0,
            self.dev.cc.1,
            self.w.bytes as f64 / (1u64 << 30) as f64,
            100.0 * self.w.bytes as f64 / self.w.bf16_bytes as f64,
            self.w.bf16_bytes as f64 / (1u64 << 30) as f64,
            self.kv.tokens(),
            self.kv.bytes as f64 / (1u64 << 30) as f64,
            self.kv.nblocks,
            mib(self.work_bytes),
            (total - free) as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        )
    }

    /// A sequence of `n` tokens is within the positions the KV tables were made for.
    fn check_window(&self, n: usize) -> Result<()> {
        if n > self.kv.max_seq() {
            return err(format!("{n} tokens, where this model is set up for {} positions: send a shorter text", self.kv.max_seq()));
        }
        Ok(())
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    pub fn reset(&mut self) {
        self.n = 0;
        self.cached.clear();
        self.kv.void.clear();
    }

    /// The residual stream's row at `dump_row`, copied out (after the work queued so far).
    fn dump_x(&self) -> Result<()> {
        if let Some(r) = self.dump_row.get() {
            self.dev.sync()?;
            let hid = self.w.cfg.hidden;
            let mut v = vec![0u16; hid];
            self.x.read(r * hid * 2, &mut v)?;
            self.dumped.borrow_mut().push(v);
        }
        Ok(())
    }

    /// Where a profile is being taken: an event now, with a label for the work just queued.
    #[cfg(feature = "dev")]
    fn mark(&self, label: &'static str) -> Result<()> {
        if let Some(v) = self.marks.borrow_mut().as_mut() {
            let e = Ev::new()?;
            e.record(self.dev.stream)?;
            v.push((label, e));
        }
        Ok(())
    }

    /// (A release build takes no profile, and holds none of the labels.)
    #[cfg(not(feature = "dev"))]
    #[inline(always)]
    fn mark(&self, _label: &'static str) -> Result<()> {
        Ok(())
    }

    /// Whether a product of `m` rows by `p` takes the decoded matrix's way: the library's route says DECODE or AHEAD (a long prompt, where decoding W once and multiplying by a dense product is the faster, on
    /// the GPUs it is measured on), or `m` is past GLYD_ENGINE_DECODE_FROM.
    unsafe fn decoded_way(&self, p: &PackBuf, m: usize) -> Result<bool> {
        if (self.blas.is_none() && !self.own) || m <= FUSED_ROWS || p.pack.cols % 64 != 0 {
            return Ok(false);
        }
        if self.decode_from.is_some_and(|d| m >= d) {
            return Ok(true);
        }
        let route = self.lib.route(self.gpu_code, p.pack.is_twelve(), p.pack.rows, p.pack.cols, m as i64)?;
        Ok(route == 0 || route == 5)
    }

    unsafe fn lin(&self, p: &PackBuf, x: u64, m: usize, y: u64) -> Result<()> {
        if self.decoded_way(p, m)? {
            self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, self.dev.stream)?;
            if let Some(b) = &self.blas {
                return b.gemm_nt(self.dev.stream, self.scratch_w.ptr, x, y, p.pack.rows as i32, p.pack.cols as i32, m as i32);
            }
            let tile = self.kern.pick_tile(m, p.pack.rows as usize, p.pack.cols as usize);
            return self.kern.gemm(self.dev.stream, tile, x, self.scratch_w.ptr, y, m as i32, p.pack.rows as i32, p.pack.cols as i32);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, self.dev.stream)
    }

    /// The gate-and-up product and silu(gate) * up into `m`: one kernel where the engine's own product runs it (the same bits as the two steps), else the two.
    unsafe fn gate_up_silu(&self, p: &PackBuf, h: u64, n: usize, inter: usize) -> Result<()> {
        let s = self.dev.stream;
        if self.blas.is_none()
            && self.own
            && self.decoded_way(p, n)?
            && p.pack.rows as usize == 2 * inter
            && inter.is_multiple_of(128)
            && crate::tune!("GLYD_ENGINE_NO_FUSE").is_none()
        {
            self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, s)?;
            self.kern.gemm_silu(s, h, self.scratch_w.ptr, self.m.ptr, n as i32, p.pack.rows as i32, p.pack.cols as i32)?;
            self.mark("gate and up product with SiLU product")?;
            return Ok(());
        }
        self.lin(p, h, n, self.gu.ptr)?;
        self.mark("gate and up product")?;
        self.kern.silu_mul(s, self.gu.ptr, self.m.ptr, n as u32, inter as i32)?;
        self.mark("SiLU product")
    }

    /// Token rows into x. A prompt chunk: each 64-row group of the packed embedding the tokens fall in decoded once (EMBED_GROUPS of them at a time), the rows gathered by one launch a batch; a single
    /// token (a decode step): its group decoded and its row copied.
    pub(crate) unsafe fn embed(&self, tokens: &[u32]) -> Result<()> {
        let hid = self.w.cfg.hidden;
        let s = self.dev.stream;
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.w.cfg.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.w.cfg.vocab));
        }
        if tokens.len() == 1 {
            let t = tokens[0] as i64;
            let row0 = (t / 64) * 64;
            self.lib.unpack(&self.w.embed.pack, row0, 64, self.scratch.ptr, s)?;
            return self.kern.copy_row(s, self.scratch.ptr, (t - row0) as i32, hid as i32, self.x.ptr);
        }
        let (groups, by_batch) = embed_plan(tokens, EMBED_GROUPS);
        // the pairs of every batch, in the order of the batches, written once (the batches' kernels read their own parts)
        let counts: Vec<usize> = by_batch.iter().map(Vec::len).collect();
        let pairs: Vec<i32> = by_batch.iter().flatten().flat_map(|&(d, s_)| [d, s_]).collect();
        self.embed_pairs.write(0, &pairs)?;
        let mut done = 0usize;
        for (b, batch) in groups.chunks(EMBED_GROUPS).enumerate() {
            for (j, &g) in batch.iter().enumerate() {
                self.lib.unpack(&self.w.embed.pack, g * 64, 64, self.embed_groups.ptr + (j * 64 * hid * 2) as u64, s)?;
            }
            self.kern.copy_rows(s, self.embed_groups.ptr, self.embed_pairs.ptr + (done * 8) as u64, counts[b] as i32, hid as i32, self.x.ptr)?;
            done += counts[b];
        }
        Ok(())
    }

    /// The layers on the n rows in x: their cache slots, positions and lengths at the given device addresses; where `logits`,
    /// the LM head on the last row and its argmax follow. One sequence on the identity block table.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn forward(
        &self,
        n: usize,
        pos: u64,
        slots: u64,
        lens: u64,
        bt_stride: i64,
        splits: i64,
        logits: bool,
        flash_start: Option<usize>,
        hook: Option<Hook>,
    ) -> Result<()> {
        let flash = flash_start.map(|start| Flash { start, gather: self.iota.ptr, lslots: slots });
        self.forward_ex(n, pos, slots, lens, self.kv.block_table.ptr, bt_stride, splits, if logits { Head::Last } else { Head::None }, flash, hook)
    }

    /// `forward` for any sequences: the block table at `bt` (rows `bt_stride` apart, one for each row of x where the rows are sequences), the head on the rows `head` says, a prompt chunk's attention as `flash` says.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn forward_ex(
        &self,
        n: usize,
        pos: u64,
        slots: u64,
        lens: u64,
        bt: u64,
        bt_stride: i64,
        splits: i64,
        head: Head,
        flash: Option<Flash>,
        mut hook: Option<Hook>,
    ) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (hid, hd, kvh, inter) = (c.hidden, c.head_dim, c.kv_heads, c.inter);
        let qkv_w = (c.heads + 2 * kvh) * hd;
        let scale = 1.0 / (hd as f64).sqrt();
        let (nu, hid_i, eps) = (n as u32, hid as i32, c.eps);
        for (l, ly) in self.w.layers.iter().enumerate() {
            if l == 0 {
                self.kern.rmsnorm(s, self.x.ptr, ly.in_norm.ptr, self.h.ptr, nu, hid_i, eps)?;
            } else {
                self.kern.rmsnorm_add(s, self.x.ptr, self.d.ptr, ly.in_norm.ptr, self.h.ptr, nu, hid_i, eps)?;
            }
            self.mark("norm (input, with the residual)")?;
            if l > 0 {
                self.dump_x()?; // (the residual after layer l - 1's MLP)
            }
            self.lin(&ly.qkv, self.h.ptr, n, self.qkv.ptr)?;
            self.mark("qkv product")?;
            self.kern.qk_norm_rope(
                s,
                self.qkv.ptr,
                qkv_w as i32,
                ly.q_norm.ptr,
                ly.k_norm.ptr,
                self.w.cos.ptr,
                self.w.sin.ptr,
                pos,
                self.qo.ptr,
                self.ko.ptr,
                nu,
                c.heads as i32,
                kvh as i32,
                eps,
            )?;
            self.mark("query/key norm and rotary")?;
            if let Some(h) = hook.as_mut() {
                // calibrating: the keys (self.ko) and values (in self.qkv) of the layer go to the calibration, which does the attention on its own memory
                h(self, l, n)?;
                self.mark("layer hook")?;
            } else {
                self.lib.kv_write(
                    &self.kv.geom,
                    self.kv.pools[l].ptr,
                    self.kv.kbase[l].ptr,
                    self.kv.vbase[l].ptr,
                    l as i64,
                    self.ko.ptr,
                    (kvh * hd) as i64,
                    self.qkv.ptr + ((c.heads + kvh) * hd * 2) as u64,
                    qkv_w as i64,
                    slots,
                    n as i64,
                    self.kv.status.ptr,
                    self.kv.lost.ptr,
                    self.kv.nlost,
                    self.kv.ring,
                    s,
                )?;
                self.mark("KV write")?;
                if let Some(v) = &self.verify {
                    self.lib.kv_stage_put(
                        &self.kv.geom,
                        self.ko.ptr,
                        (kvh * hd) as i64,
                        self.qkv.ptr + ((c.heads + kvh) * hd * 2) as u64,
                        qkv_w as i64,
                        slots,
                        n as i64,
                        v.shadow[l].ptr,
                        self.kv.nblocks,
                        s,
                    )?;
                }
                match flash {
                    Some(Flash { .. }) if !self.parts.borrow().is_empty() => {
                        // a batch of prompts: each sequence's earlier pages decoded to the staging cache, its keys and values put after them, our causal kernel on that; the rows of every
                        // other layer's work (products, norms, rope, the cache write) were for all of them at once
                        let Some(Flash { gather, lslots, .. }) = flash else { unreachable!() };
                        let (hd2, kv2) = ((c.heads * hd * 2) as u64, (kvh * hd * 2) as u64);
                        for p in self.parts.borrow().iter() {
                            if p.earlier > 0 {
                                self.lib.kv_gather(
                                    &self.kv.geom,
                                    self.kv.pools[l].ptr,
                                    self.kv.kbase[l].ptr,
                                    self.kv.vbase[l].ptr,
                                    gather + 4 * p.gather_off as u64,
                                    self.iota.ptr,
                                    p.earlier as i64,
                                    self.stage.ptr,
                                    s,
                                )?;
                            }
                            self.lib.kv_stage_put(
                                &self.kv.geom,
                                self.ko.ptr + p.off as u64 * kv2,
                                (kvh * hd) as i64,
                                self.qkv.ptr + ((c.heads + kvh) * hd * 2) as u64 + (p.off * qkv_w * 2) as u64,
                                qkv_w as i64,
                                lslots + 8 * p.off as u64,
                                p.n as i64,
                                self.stage.ptr,
                                self.stage_blocks,
                                s,
                            )?;
                            self.kern.prefill_attn(
                                s,
                                self.qo.ptr + p.off as u64 * hd2,
                                (c.heads * hd) as i32,
                                self.stage.ptr,
                                self.attn.ptr + p.off as u64 * hd2,
                                (c.heads * hd) as i32,
                                p.n as i32,
                                p.start as i32,
                                c.heads as i32,
                                kvh as i32,
                            )?;
                        }
                        self.mark("prompt attention (a batch)")?;
                    }
                    Some(Flash { start, gather, lslots }) => {
                        // a prompt chunk: the earlier pages decoded to bf16, the chunk's own keys and values put after them, our causal kernel on that
                        let earlier = (start as i64 + 15) / 16;
                        if earlier > 0 {
                            self.lib.kv_gather(
                                &self.kv.geom,
                                self.kv.pools[l].ptr,
                                self.kv.kbase[l].ptr,
                                self.kv.vbase[l].ptr,
                                gather,
                                self.iota.ptr,
                                earlier,
                                self.stage.ptr,
                                s,
                            )?;
                        }
                        self.mark("KV gather")?;
                        self.lib.kv_stage_put(
                            &self.kv.geom,
                            self.ko.ptr,
                            (kvh * hd) as i64,
                            self.qkv.ptr + ((c.heads + kvh) * hd * 2) as u64,
                            qkv_w as i64,
                            lslots,
                            n as i64,
                            self.stage.ptr,
                            self.stage_blocks,
                            s,
                        )?;
                        self.mark("KV stage")?;
                        self.kern.prefill_attn(
                            s,
                            self.qo.ptr,
                            (c.heads * hd) as i32,
                            self.stage.ptr,
                            self.attn.ptr,
                            (c.heads * hd) as i32,
                            n as i32,
                            start as i32,
                            c.heads as i32,
                            kvh as i32,
                        )?;
                        self.mark("prompt attention")?;
                    }
                    None => {
                        self.lib.kv_decode(
                            &self.kv.geom,
                            self.kv.group,
                            self.kv.pools[l].ptr,
                            self.kv.kbase[l].ptr,
                            self.kv.vbase[l].ptr,
                            self.qo.ptr,
                            (c.heads * hd) as i64,
                            bt,
                            bt_stride,
                            lens,
                            n as i64,
                            splits,
                            scale,
                            self.attn.ptr,
                            (c.heads * hd) as i64,
                            self.kvws.ptr,
                            self.kvws.bytes,
                            s,
                        )?;
                        self.mark("attention over the cache")?;
                    }
                }
            }
            self.lin(&ly.o, self.attn.ptr, n, self.o.ptr)?;
            self.mark("o product")?;
            self.kern.rmsnorm_add(s, self.x.ptr, self.o.ptr, ly.post_norm.ptr, self.h.ptr, nu, hid_i, eps)?;
            self.mark("norm (after attention, with the residual)")?;
            self.dump_x()?; // (the residual after layer l's attention)
            self.gate_up_silu(&ly.gate_up, self.h.ptr, n, inter)?;
            self.lin(&ly.down, self.m.ptr, n, self.d.ptr)?;
            self.mark("down product")?;
        }
        if head == Head::Rows {
            // the last row of each sequence of a prompt batch
            let rows = self.picked.borrow().clone();
            for (j, &r) in rows.iter().enumerate() {
                let (at, to) = ((r * hid * 2) as u64, (j * hid * 2) as u64);
                self.kern.rmsnorm_add(s, self.x.ptr + at, self.d.ptr + at, self.w.norm.ptr, self.hlast.ptr + to, 1, hid_i, eps)?;
            }
            let k = rows.len();
            self.mark("final norm")?;
            self.lin(self.w.lm_head(), self.hlast.ptr, k, self.logits.ptr)?;
            self.mark("LM head product")?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, k as u32, self.tok.ptr, self.val.ptr)?;
            self.mark("argmax")?;
            if !self.host_sampling {
                self.kern.topk_rows(s, self.logits.ptr, c.vocab as i32, k as u32, self.topk_tok.ptr, self.topk_val.ptr)?;
                self.mark("top candidates")?;
            }
        } else if head != Head::None {
            // the last row, or every row (a step of several sequences)
            let (first, rows) = if head == Head::Last { ((n - 1) * hid * 2, 1) } else { (0, n) };
            let first = first as u64;
            self.kern.rmsnorm_add(s, self.x.ptr + first, self.d.ptr + first, self.w.norm.ptr, self.hlast.ptr, rows as u32, hid_i, eps)?;
            self.mark("final norm")?;
            self.lin(self.w.lm_head(), self.hlast.ptr, rows, self.logits.ptr)?;
            self.mark("LM head product")?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, rows as u32, self.tok.ptr, self.val.ptr)?;
            self.mark("argmax")?;
            if !self.host_sampling {
                self.kern.topk_rows(s, self.logits.ptr, c.vocab as i32, rows as u32, self.topk_tok.ptr, self.topk_val.ptr)?;
                self.mark("top candidates")?;
            }
        }
        Ok(())
    }

    /// The decode step as a graph (launched with the state at `st`: position, length and slot).
    fn capture_graph(&mut self) -> Result<()> {
        // one eager step first (lazy initializations stay out of the capture); it writes slot 0, which a prefill rewrites
        self.dev.sync()?;
        let st = self.st.ptr;
        self.st.write(0, &[0i32, 1i32])?;
        self.st.write(8, &[0i64])?;
        let (n_tok, bt) = (1usize, self.kv.nblocks);
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            self.embed(&[0u32])?;
            self.forward(n_tok, st, st + 8, st + 4, bt, self.splits_dec, true, None, None)?;
        }
        self.dev.sync()?;
        let g = self.dev.capture(|| {
            // SAFETY: as above; nothing runs during a capture.
            unsafe { self.forward(n_tok, st, st + 8, st + 4, bt, self.splits_dec, true, None, None) }
        })?;
        self.graph = Some(g);
        self.dev.sync()?;
        self.kv.status.zero()?; // (the warm-up's writes)
        Ok(())
    }

    /// A decode graph for each number of rows 1 to `rows`: the rows' positions, lengths and slots are read from `bstate`, their block-table rows from `btab`, and the head runs on every row.
    fn capture_graphs(&mut self, rows: usize) -> Result<()> {
        self.dev.sync()?;
        let bs = self.bstate.ptr;
        let mut st = [0u8; 128];
        for i in 0..MAX_ROWS {
            st[B_LENS as usize + 4 * i..][..4].copy_from_slice(&1i32.to_le_bytes());
        }
        self.bstate.write(0, &st[..])?;
        for b in 1..=rows {
            // one eager step first, as `capture_graph` does: every row at position 0 of page 0, which a sequence's first token rewrites
            // SAFETY: live buffers of the engine; the stream is synchronized after.
            unsafe {
                self.embed(&vec![0u32; b])?;
                self.forward_ex(b, bs + B_POS, bs + B_SLOTS, bs + B_LENS, self.btab.ptr, self.bt_stride, self.splits_dec, Head::All, None, None)?;
            }
            self.dev.sync()?;
            let g = self.dev.capture(|| {
                // SAFETY: as above; nothing runs during a capture.
                unsafe { self.forward_ex(b, bs + B_POS, bs + B_SLOTS, bs + B_LENS, self.btab.ptr, self.bt_stride, self.splits_dec, Head::All, None, None) }
            })?;
            self.graphs[b] = Some(g);
            self.dev.sync()?;
        }
        self.kv.status.zero()?; // (the warm-up's writes)
        Ok(())
    }

    /// With GLYD_ENGINE_VERIFY: the full pages `pages`, decoded from the cache in every layer, compared value for value with the bf16 copy of what was written (see `verify_tokens`).
    pub fn verify_pages(&mut self, pages: &[i32]) -> Result<()> {
        let all: Vec<(i32, usize)> = pages.iter().map(|&p| (p, 16)).collect();
        self.verify_tokens(&all)
    }

    /// With GLYD_ENGINE_VERIFY: for each (page, tokens) of `pages`, its first `tokens` tokens decoded from the cache in every layer and compared value for value with the bf16 copy of what was written; the counts go to VERIFY_CHECKED and
    /// VERIFY_DIFFER. A page the cache has named lost (what it holds is not what was written: the scheduler takes the step's tokens back from what holds it) is not compared, its values are counted in VERIFY_LOST.
    pub fn verify_tokens(&mut self, pages: &[(i32, usize)]) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(v) = &self.verify else { return Ok(()) };
        let per_token = 256 * self.kv.geom.heads as u64;
        let elems = 16 * per_token;
        let layers = self.kv.layers as u64;
        let (live, lost): (Vec<_>, Vec<_>) = pages.iter().copied().partition(|(p, _)| !self.kv.void.contains(p));
        VERIFY_LOST.fetch_add(lost.iter().map(|&(_, n)| n as u64).sum::<u64>() * per_token * layers, Relaxed);
        if live.is_empty() {
            return Ok(());
        }
        for part in live.chunks(64) {
            // (the previous part's kernels read `rel`; the write below does not wait for the stream, so they are waited for first: without this the second part's page numbers were read by the first part's gathers, and values
            // that were stored exactly were reported as different)
            self.dev.sync()?;
            let ids: Vec<i32> = part.iter().map(|&(p, _)| p).collect();
            self.rel.write(0, &ids)?;
            for l in 0..self.kv.layers {
                // SAFETY: live buffers of the engine; `rel` holds the page numbers, `iota` 0, 1, ...; the temporary holds 64 pages.
                unsafe {
                    self.lib.kv_gather(
                        &self.kv.geom,
                        self.kv.pools[l].ptr,
                        self.kv.kbase[l].ptr,
                        self.kv.vbase[l].ptr,
                        self.rel.ptr,
                        self.iota.ptr,
                        part.len() as i64,
                        v.tmp.ptr,
                        self.dev.stream,
                    )?;
                    for (j, &(p, n)) in part.iter().enumerate() {
                        self.kern.count_diff(
                            self.dev.stream,
                            v.tmp.ptr + j as u64 * elems * 2,
                            v.shadow[l].ptr + p as u64 * elems * 2,
                            n as u64 * per_token,
                            v.count.ptr,
                            self.kv.geom.heads as i32,
                            128,
                        )?;
                    }
                }
            }
        }
        self.dev.sync()?;
        let mut d = [0u64; 17];
        v.count.read(0, &mut d)?;
        v.count.zero()?;
        for (a, x) in VERIFY_BY.iter().zip(&d[1..]) {
            a.fetch_add(*x, Relaxed);
        }
        VERIFY_CHECKED.fetch_add(live.iter().map(|&(_, n)| n as u64).sum::<u64>() * per_token * layers, Relaxed);
        VERIFY_DIFFER.fetch_add(d[0], Relaxed);
        Ok(())
    }

    /// A copy of the cache's status header to the pinned buffer, queued on the stream (read by `read_head` once the stream is synchronized).
    fn queue_head(&self) -> Result<()> {
        // SAFETY: the pinned buffer holds HOST_BATCH bytes; the status buffer is longer than its header.
        unsafe { copy_d2h_async((self.host.ptr as *mut u8).add(HOST_HEAD) as *mut _, self.kv.status.ptr, 4 * HEAD_WORDS, self.dev.stream) }
    }

    /// The status header `queue_head` copied (the stream synchronized since).
    fn head_words(&self) -> [u32; HEAD_WORDS] {
        let mut w = [0u32; HEAD_WORDS];
        // SAFETY: the pinned bytes were copied by `queue_head`.
        unsafe { std::ptr::copy_nonoverlapping((self.host.ptr as *const u8).add(HOST_HEAD) as *const u32, w.as_mut_ptr(), HEAD_WORDS) };
        w
    }

    /// After a step of the scheduler's (the stream synchronized, the header queued): blocks written from their first token are new, the rings are read for what the step lost, and the written tokens are compared with the bf16 copy.
    fn after_step(&mut self, fresh: &[i32], touched: &[(i32, usize)]) -> Result<()> {
        self.kv.renew(fresh);
        let w = self.head_words();
        let l = self.kv.drain(&w)?;
        self.lost_acc.merge(l);
        self.verify_tokens(touched)
    }

    /// The sampler's candidates for row `row` of the last step (a decode step's rows, or row 0 after a prompt's last chunk): the TOPK largest logits of its row, largest first, equal ones by the smaller token index.
    pub fn candidates(&self, row: usize) -> Option<Vec<(u32, f32)>> {
        if self.host_sampling {
            return None;
        }
        let p = self.host.ptr as *const u8;
        // SAFETY: the pinned bytes were copied by the step (the stream synchronized since), MAX_ROWS rows of TOPK of each fit between the offsets.
        Some(unsafe { read_candidates(p, HOST_TOPK_TOK, HOST_TOPK_VAL, row) })
    }

    /// What the cache lost or made heavy in the steps since the last call.
    pub fn take_lost(&mut self) -> Lost {
        std::mem::take(&mut self.lost_acc)
    }

    /// The keys and values of tokens `from` .. `from + n` of two sequences whose pages are `a` and `b` (the same tokens, made two ways), decoded from the cache in every layer and compared value for value: (values compared, values that differ).
    pub fn compare_blocks(&mut self, a: &[i32], b: &[i32], from: usize, n: usize) -> Result<(u64, u64)> {
        let per = 256 * self.kv.geom.heads as usize;
        let np = (from + n).div_ceil(16);
        if np > a.len() || np > b.len() || np as i64 > self.stage_blocks {
            return err(format!("{} tokens in {} and {} blocks", from + n, a.len(), b.len()));
        }
        let (pa, pb) = (Buf::new(4 * np)?, Buf::new(4 * np)?);
        pa.write(0, &a[..np])?;
        pb.write(0, &b[..np])?;
        let (ta, tb) = (Buf::new(np * 16 * per * 2)?, Buf::new(np * 16 * per * 2)?);
        let (mut xa, mut xb) = (vec![0u16; np * 16 * per], vec![0u16; np * 16 * per]);
        let (mut checked, mut differ) = (0u64, 0u64);
        self.dev.sync()?;
        for l in 0..self.kv.layers {
            // SAFETY: live buffers of the engine; the stream is synchronized before they are read.
            unsafe {
                self.lib.kv_gather(
                    &self.kv.geom,
                    self.kv.pools[l].ptr,
                    self.kv.kbase[l].ptr,
                    self.kv.vbase[l].ptr,
                    pa.ptr,
                    self.iota.ptr,
                    np as i64,
                    ta.ptr,
                    self.dev.stream,
                )?;
                self.lib.kv_gather(
                    &self.kv.geom,
                    self.kv.pools[l].ptr,
                    self.kv.kbase[l].ptr,
                    self.kv.vbase[l].ptr,
                    pb.ptr,
                    self.iota.ptr,
                    np as i64,
                    tb.ptr,
                    self.dev.stream,
                )?;
            }
            self.dev.sync()?;
            ta.read(0, &mut xa)?;
            tb.read(0, &mut xb)?;
            for i in from * per..(from + n) * per {
                checked += 1;
                differ += (xa[i] != xb[i]) as u64;
            }
        }
        Ok((checked, differ))
    }

    /// Pages of the cache in all (the sequences' room together).
    pub fn pool_pages(&self) -> usize {
        self.kv.nblocks as usize
    }

    /// A prompt chunk of one sequence: `tokens` at positions `start` on, written into the sequence's `pages` (at least as many as they need); where `last`, the logits of the chunk's final row too, and the token
    /// chosen from them (the largest logit).
    pub fn prefill_chunk(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        let n = tokens.len();
        if n == 0 || n > self.chunk {
            return err(format!("{n} tokens in a chunk of at most {}", self.chunk));
        }
        if start + n > self.capacity() || pages.len() * 16 < start + n {
            return err(format!("{} tokens in {} blocks, where the window holds {}", start + n, pages.len(), self.capacity()));
        }
        if self.attn_decode {
            return err("the prompt's attention through the decode kernel is for checking one sequence");
        }
        self.chunk_attention(start, n)?;
        let slot = |p: usize| pages[p / 16] as i64 * 16 + (p % 16) as i64;
        let pos: Vec<i32> = (0..n).map(|i| (start + i) as i32).collect();
        let slots: Vec<i64> = (0..n).map(|i| slot(start + i)).collect();
        let lslots: Vec<i64> = (0..n).map(|i| (start + i) as i64).collect();
        let lens: Vec<i32> = (0..n).map(|i| (start + i + 1) as i32).collect();
        self.dev.sync()?; // (the arrays below are read by the previous chunk's kernels until then)
        self.pos.write(0, &pos)?;
        self.slots.write(0, &slots)?;
        self.lslots.write(0, &lslots)?;
        self.lens.write(0, &lens)?;
        let earlier = start.div_ceil(16);
        if earlier > 0 {
            self.gpages.write(0, &pages[..earlier])?;
        }
        let flash = Flash { start, gather: self.gpages.ptr, lslots: self.lslots.ptr };
        // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
        unsafe {
            self.embed(tokens)?;
            self.forward_ex(
                n,
                self.pos.ptr,
                self.slots.ptr,
                self.lens.ptr,
                self.kv.block_table.ptr,
                0,
                0,
                if last { Head::Last } else { Head::None },
                Some(flash),
                None,
            )?;
            self.queue_head()?;
        }
        self.dev.sync()?;
        // the blocks whose first token the chunk writes are new lives (the kernel cleared them first); of the pages it touches, the tokens up to its last hold what it wrote
        let fresh: Vec<i32> = (start.div_ceil(16)..(start + n).div_ceil(16)).map(|b| pages[b]).collect();
        let touched: Vec<(i32, usize)> = (start / 16..(start + n).div_ceil(16)).map(|b| (pages[b], (start + n - 16 * b).min(16))).collect();
        self.after_step(&fresh, &touched)?;
        if !last {
            return Ok(None);
        }
        Ok(Some(self.read_sample(false)?))
    }

    /// Prompt chunks of several sequences in one pass (1 to MAX_ROWS of them, `chunk` tokens in all): the products, norms, rope and cache writes run on all their rows at once (the weights are read once, not once a
    /// sequence), the attention on each sequence's rows as `prefill_chunk` does it. The sample after each chunk that is `last`, in order of the jobs (None for the rest); the logits of a sample are row `j` of `read_logits_row`
    /// where it is the j-th `last` (as the rows of a decode step are).
    pub fn prefill_batch(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
        if let [j] = jobs {
            return Ok(vec![self.prefill_chunk(j.pages, j.start, j.tokens, j.last)?]);
        }
        if jobs.is_empty() || jobs.len() > MAX_ROWS {
            return err(format!("{} sequences in a prompt batch: 1 to {MAX_ROWS}", jobs.len()));
        }
        if self.attn_decode {
            return err("the prompt's attention through the decode kernel is for checking one sequence");
        }
        if let Some(j) = jobs.iter().find(|j| j.tokens.len() > FUSED_ROWS) {
            return err(format!("a chunk of {} tokens in a batch: up to {FUSED_ROWS} run together", j.tokens.len()));
        }
        let total: usize = jobs.iter().map(|j| j.tokens.len()).sum();
        if total == 0 || total > self.chunk || jobs.iter().any(|j| j.tokens.is_empty()) {
            return err(format!("{total} tokens in a batch of chunks of at most {}", self.chunk));
        }
        let (mut tokens, mut pos, mut slots, mut lslots, mut lens, mut gather) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut parts, mut picked, mut fresh, mut touched) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for j in jobs {
            let (start, n) = (j.start, j.tokens.len());
            if start + n > self.capacity() || j.pages.len() * 16 < start + n {
                return err(format!("{} tokens in {} blocks, where the window holds {}", start + n, j.pages.len(), self.capacity()));
            }
            if (start + n).div_ceil(16) as i64 > self.stage_blocks {
                return err(format!("{} tokens, where the work cache of a prompt's attention holds {}", start + n, self.stage_blocks * 16));
            }
            let earlier = start.div_ceil(16);
            parts.push(Part { off: tokens.len(), n, start, earlier, gather_off: gather.len() });
            gather.extend_from_slice(&j.pages[..earlier]);
            for i in 0..n {
                let p = start + i;
                pos.push(p as i32);
                slots.push(j.pages[p / 16] as i64 * 16 + (p % 16) as i64);
                lslots.push(p as i64);
                lens.push((p + 1) as i32);
            }
            tokens.extend_from_slice(j.tokens);
            if j.last {
                picked.push(tokens.len() - 1);
            }
            fresh.extend((start.div_ceil(16)..(start + n).div_ceil(16)).map(|b| j.pages[b]));
            touched.extend((start / 16..(start + n).div_ceil(16)).map(|b| (j.pages[b], (start + n - 16 * b).min(16))));
        }
        if gather.len() > self.stage_blocks as usize * MAX_ROWS {
            return err(format!("{} earlier blocks in a batch", gather.len()));
        }
        self.dev.sync()?; // (the arrays below are read by the previous step's kernels until then)
        self.pos.write(0, &pos)?;
        self.slots.write(0, &slots)?;
        self.lslots.write(0, &lslots)?;
        self.lens.write(0, &lens)?;
        if !gather.is_empty() {
            self.gpages.write(0, &gather)?;
        }
        let k = picked.len();
        *self.parts.borrow_mut() = parts;
        *self.picked.borrow_mut() = picked;
        let flash = Flash { start: 0, gather: self.gpages.ptr, lslots: self.lslots.ptr };
        let p = self.host.ptr as *mut u8;
        // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step; the pinned buffer holds what the copies of a decode step do.
        let run = unsafe {
            (|| -> Result<()> {
                self.embed(&tokens)?;
                self.forward_ex(
                    total,
                    self.pos.ptr,
                    self.slots.ptr,
                    self.lens.ptr,
                    self.kv.block_table.ptr,
                    0,
                    0,
                    if k > 0 { Head::Rows } else { Head::None },
                    Some(flash),
                    None,
                )?;
                if k > 0 {
                    copy_d2h_async(p.add(512) as *mut _, self.tok.ptr, 4 * k, self.dev.stream)?;
                    copy_d2h_async(p.add(768) as *mut _, self.val.ptr, 4 * k, self.dev.stream)?;
                    if !self.host_sampling {
                        self.mask_candidates(k)?;
                        copy_d2h_async(p.add(HOST_TOPK_TOK) as *mut _, self.topk_tok.ptr, 4 * TOPK * k, self.dev.stream)?;
                        copy_d2h_async(p.add(HOST_TOPK_VAL) as *mut _, self.topk_val.ptr, 4 * TOPK * k, self.dev.stream)?;
                    }
                }
                self.queue_head()
            })()
        };
        self.parts.borrow_mut().clear();
        run?;
        self.dev.sync()?;
        self.after_step(&fresh, &touched)?;
        let mut row = 0;
        Ok(jobs
            .iter()
            .map(|j| {
                j.last.then(|| {
                    // SAFETY: the pinned bytes were just written.
                    let s =
                        unsafe { Sample { token: *(p.add(512 + 4 * row) as *const i32) as u32, logit: *(p.add(768 + 4 * row) as *const f32), logprob: None } };
                    row += 1;
                    s
                })
            })
            .collect())
    }

    /// One decode step of several sequences (1 to MAX_ROWS): each row's token at the sequence's next position, its keys and values written into the sequence's pages; the token that follows each, by the largest logit
    /// (the logits stay for `read_logits_row` until the next step).
    pub fn decode_batch(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        let b = rows.len();
        if b == 0 || b > MAX_ROWS {
            return err(format!("{b} sequences in a step: 1 to {MAX_ROWS}"));
        }
        let t_start = Instant::now();
        let mut timed = false;
        let mut st = [0u8; 128];
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        for (i, r) in rows.iter().enumerate() {
            if r.pos + 1 > self.capacity() || r.pages.len() <= r.pos / 16 {
                return err(format!("position {} in {} blocks, where the window holds {}", r.pos, r.pages.len(), self.capacity()));
            }
            st[B_POS as usize + 4 * i..][..4].copy_from_slice(&(r.pos as i32).to_le_bytes());
            st[B_LENS as usize + 4 * i..][..4].copy_from_slice(&((r.pos + 1) as i32).to_le_bytes());
            let slot = r.pages[r.pos / 16] as i64 * 16 + (r.pos % 16) as i64;
            st[B_SLOTS as usize + 8 * i..][..8].copy_from_slice(&slot.to_le_bytes());
        }
        let bs = self.bstate.ptr;
        let p = self.host.ptr as *mut u8;
        // SAFETY: the pinned buffer holds HOST_BATCH bytes and a row of the block table for each of MAX_ROWS; the copies are queued before the step on the stream, which was synchronized at the last step's end.
        unsafe {
            std::ptr::copy_nonoverlapping(st.as_ptr(), p.add(256), 128);
            copy_h2d_async(bs, p.add(256) as *const _, 128, self.dev.stream)?;
            for (i, r) in rows.iter().enumerate() {
                let np = r.pos / 16 + 1;
                let at = p.add(HOST_BATCH + i * self.bt_stride as usize * 4);
                std::ptr::copy_nonoverlapping(r.pages.as_ptr() as *const u8, at, np * 4);
                copy_h2d_async(self.btab.ptr + (i * self.bt_stride as usize * 4) as u64, at as *const _, np * 4, self.dev.stream)?;
            }
            self.embed(&tokens)?;
            // (a profile times the launches one by one: no graph)
            match self.graphs.get(b).and_then(|g| g.as_ref()).filter(|_| self.marks.borrow().is_none()) {
                Some(g) => {
                    self.graph_ev.0.record(self.dev.stream)?;
                    g.launch(self.dev.stream)?;
                    self.graph_ev.1.record(self.dev.stream)?;
                    timed = true;
                }
                None => self.forward_ex(b, bs + B_POS, bs + B_SLOTS, bs + B_LENS, self.btab.ptr, self.bt_stride, self.splits_dec, Head::All, None, None)?,
            }
            copy_d2h_async(p.add(512) as *mut _, self.tok.ptr, 4 * b, self.dev.stream)?;
            copy_d2h_async(p.add(768) as *mut _, self.val.ptr, 4 * b, self.dev.stream)?;
            if !self.host_sampling {
                self.mask_candidates(b)?;
                copy_d2h_async(p.add(HOST_TOPK_TOK) as *mut _, self.topk_tok.ptr, 4 * TOPK * b, self.dev.stream)?;
                copy_d2h_async(p.add(HOST_TOPK_VAL) as *mut _, self.topk_val.ptr, 4 * TOPK * b, self.dev.stream)?;
            }
            self.queue_head()?;
        }
        let t_queued = Instant::now();
        self.dev.sync()?;
        let t_done = Instant::now();
        // SAFETY: the pinned bytes were just written.
        let out = (0..b)
            .map(|i| unsafe { Sample { token: *(p.add(512 + 4 * i) as *const i32) as u32, logit: *(p.add(768 + 4 * i) as *const f32), logprob: None } })
            .collect();
        self.steps += 1;
        // (a row at the first token of a block starts a new life of it; every row's token is the last of its page so far)
        let fresh: Vec<i32> = rows.iter().filter(|r| r.pos % 16 == 0).map(|r| r.pages[r.pos / 16]).collect();
        let touched: Vec<(i32, usize)> = rows.iter().map(|r| (r.pages[r.pos / 16], r.pos % 16 + 1)).collect();
        self.after_step(&fresh, &touched)?;
        use std::sync::atomic::Ordering::Relaxed;
        DECODE_PREP_NS.fetch_add(t_queued.duration_since(t_start).as_nanos() as u64, Relaxed);
        DECODE_WAIT_NS.fetch_add(t_done.duration_since(t_queued).as_nanos() as u64, Relaxed);
        DECODE_POST_NS.fetch_add(t_done.elapsed().as_nanos() as u64, Relaxed);
        if timed {
            if let Ok(ms) = self.graph_ev.1.since(&self.graph_ev.0) {
                DECODE_GPU_NS.fetch_add((ms as f64 * 1e6) as u64, Relaxed);
            }
        }
        Ok(out)
    }

    /// Row `row` of the last decode step's logits (or the last prompt chunk's final row, as row 0), on the host as f32.
    pub fn read_logits_row(&mut self, row: usize) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let v = self.w.cfg.vocab;
        let mut raw = vec![0u16; v];
        self.logits.read(row * v * 2, &mut raw)?;
        Ok(raw.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// The pages of sequences that have ended and that nothing else names, cleared in every layer (their chunks and slabs back to the extra pools).
    pub fn release_pages(&mut self, pages: &[i32]) -> Result<()> {
        // (GLYD_ENGINE_LAZY_RELEASE=1 leaves the clearing to the next write of the page, to see what the clearing costs: the extra pools are then sized for what the pages of natural text need at once, and a cache that is mostly pages of
        // other sequences' random tokens runs out of them)
        if self.lazy_release {
            return Ok(());
        }
        for part in pages.chunks(self.kv.nblocks.max(1) as usize) {
            self.dev.sync()?;
            self.rel.write(0, part)?;
            for l in 0..self.kv.layers {
                // SAFETY: live buffers of the engine; `rel` holds the page numbers.
                unsafe {
                    self.lib.kv_release(
                        &self.kv.geom,
                        self.kv.pools[l].ptr,
                        self.kv.nblocks,
                        self.rel.ptr,
                        part.len() as i64,
                        l as i64,
                        self.kv.status.ptr,
                        self.kv.lost.ptr,
                        self.kv.nlost,
                        self.dev.stream,
                    )?
                };
            }
        }
        self.dev.sync()?;
        self.kv.renew(pages);
        Ok(())
    }

    /// How a prompt chunk of `n` tokens after `start` earlier ones does its attention: (the decode kernel's splits, the flash kernel's start). The flash kernel on the
    /// staging cache, or (Opts::attn_decode) the library's decode kernel with a row for every token, all reading the same pages with lengths start + 1 ... start + n.
    fn chunk_attention(&self, start: usize, n: usize) -> Result<(i64, Option<usize>)> {
        if !self.attn_decode {
            if (start + n).div_ceil(16) as i64 > self.stage_blocks {
                return err(format!("{} tokens, where the work cache of a prompt's attention holds {}", start + n, self.stage_blocks * 16));
            }
            return Ok((0, Some(start)));
        }
        let splits = self.lib.kv_decode_splits(n as i64, self.kv.geom.heads, (start + n).div_ceil(16) as i64)?;
        let need = self.lib.kv_decode_workspace(n as i64, self.kv.geom.heads, self.kv.group, 128, splits)?;
        if need > self.kvws.bytes {
            return err(format!("attention workspace: {need} bytes for {n} rows, the buffer has {}", self.kvws.bytes));
        }
        Ok((splits, None))
    }

    /// The masks the scheduler named for this step's `rows` rows (`ModelStep::masks`, taken): the rows' candidates made again with them, behind the step's work. Nothing where it named none.
    fn mask_candidates(&self, rows: usize) -> Result<()> {
        let masks = std::mem::take(&mut *self.masks.borrow_mut());
        if masks.len() != rows {
            return Ok(());
        }
        if let Some(g) = &self.maskgpu {
            // SAFETY: the rows' logits and candidates are the step's, on its stream; the stream was synchronized after the last step, whose copies read the staging bytes.
            unsafe { g.apply(&self.kern, &self.dev, &masks, self.logits.ptr, self.topk_tok.ptr, self.topk_val.ptr)? };
        }
        Ok(())
    }

    /// The grammar masks of the rows the next step takes samples from (one for each, None for a row with no grammar), for the candidates (`mask_candidates`).
    pub fn set_masks(&self, masks: Vec<Option<crate::constrain::MaskHandle>>) {
        *self.masks.borrow_mut() = masks;
    }

    /// The sampled token (and logit) after the work queued: read back; with `logprob`, its logprob too.
    fn read_sample(&mut self, logprob: bool) -> Result<Sample> {
        let tp = self.host.ptr as *mut i32;
        // SAFETY: the pinned buffer holds 4096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(self.host.ptr, self.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((self.host.ptr as *mut u8).add(8) as *mut _, self.val.ptr, 4, self.dev.stream)?;
            if !self.host_sampling {
                copy_d2h_async((self.host.ptr as *mut u8).add(HOST_TOPK_TOK) as *mut _, self.topk_tok.ptr, 4 * TOPK, self.dev.stream)?;
                copy_d2h_async((self.host.ptr as *mut u8).add(HOST_TOPK_VAL) as *mut _, self.topk_val.ptr, 4 * TOPK, self.dev.stream)?;
            }
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*tp, *((self.host.ptr as *const u8).add(8) as *const f32)) };
        let mut s = Sample { token: token as u32, logit, logprob: None };
        if logprob {
            // SAFETY: live buffers; the lse is read after the synchronize.
            unsafe {
                self.kern.logsumexp(self.dev.stream, self.logits.ptr, self.w.cfg.vocab as i32, logit, self.lse.ptr)?;
            }
            self.dev.sync()?;
            let mut l = [0f32; 1];
            self.lse.read(0, &mut l)?;
            s.logprob = Some(logit - l[0]);
        }
        Ok(s)
    }

    /// The prompt through the model; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32], logprob: bool) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() > self.kv.tokens() {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.kv.tokens()));
        }
        self.check_window(self.n + tokens.len())?;
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let start = self.n;
            let last = at + n == tokens.len();
            let pos: Vec<i32> = (0..n).map(|i| (start + i) as i32).collect();
            let slots: Vec<i64> = (0..n).map(|i| (start + i) as i64).collect();
            let lens: Vec<i32> = (0..n).map(|i| (start + i + 1) as i32).collect();
            self.dev.sync()?; // (the arrays below are read by the previous chunk's kernels until then)
            self.pos.write(0, &pos)?;
            self.slots.write(0, &slots)?;
            self.lens.write(0, &lens)?;
            let (splits, flash) = self.chunk_attention(start, n)?;
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, self.slots.ptr, self.lens.ptr, 0, splits, last, flash, None)?;
            }
            self.n += n;
            self.cached.extend_from_slice(&tokens[at..at + n]);
            at += n;
        }
        let s = self.read_sample(logprob)?;
        if let Err(e) = self.kv.check_status() {
            self.reset();
            return Err(e);
        }
        Ok(s)
    }

    /// Teacher-forced scoring of `tokens` from an empty cache: for each position i before the last, the model's largest-logit token and the
    /// logprob it gives token i + 1 (the proofs' top-1 agreement and logprob difference over a long text).
    pub fn score(&mut self, tokens: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        if tokens.len() < 2 || tokens.len() > self.kv.tokens() {
            return err(format!("{} tokens to score, the cache holds {}", tokens.len(), self.kv.tokens()));
        }
        self.check_window(tokens.len())?;
        self.reset();
        let (hid, v) = (self.w.cfg.hidden, self.w.cfg.vocab);
        let all = Buf::new(self.chunk * v * 2)?;
        let need = self.lib.linear_workspace(self.w.lm_head().pack.is_twelve(), v as i64, hid as i64, self.chunk as i64)?;
        let sws = Buf::zeroed(need.max(self.ws.bytes))?;
        let tgt = Buf::new(4 * self.chunk)?;
        let (top1, lse, tl) = (Buf::new(4 * self.chunk)?, Buf::new(4 * self.chunk)?, Buf::new(4 * self.chunk)?);
        let (mut top, mut lp) = (Vec::new(), Vec::new());
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let start = self.n;
            let pos: Vec<i32> = (0..n).map(|i| (start + i) as i32).collect();
            let slots: Vec<i64> = (0..n).map(|i| (start + i) as i64).collect();
            let lens: Vec<i32> = (0..n).map(|i| (start + i + 1) as i32).collect();
            let targets: Vec<i32> = (0..n).map(|i| tokens.get(at + i + 1).map_or(-1, |&t| t as i32)).collect();
            self.dev.sync()?;
            self.pos.write(0, &pos)?;
            self.slots.write(0, &slots)?;
            self.lens.write(0, &lens)?;
            tgt.write(0, &targets)?;
            let want: Option<usize> = crate::tune!("GLYD_ENGINE_DUMP").and_then(|v| v.parse().ok());
            self.dump_row.set(want.filter(|&p| p >= start && p < start + n).map(|p| p - start));
            self.dumped.borrow_mut().clear();
            let (splits, flash) = self.chunk_attention(start, n)?;
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, self.slots.ptr, self.lens.ptr, 0, splits, false, flash, None)?;
                // the last layer's down product added, every row normalized, the LM head on all rows
                self.kern.rmsnorm_add(self.dev.stream, self.x.ptr, self.d.ptr, self.w.norm.ptr, self.h.ptr, n as u32, hid as i32, self.w.cfg.eps)?;
                self.dump_x()?; // (the residual after the last layer)
                self.lib.linear(&self.w.lm_head().pack, self.h.ptr, n as i64, all.ptr, sws.ptr, sws.bytes, self.done.ptr, self.dev.stream)?;
                self.kern.row_stats(self.dev.stream, all.ptr, v as i32, tgt.ptr, top1.ptr, lse.ptr, tl.ptr, n as u32)?;
            }
            self.dev.sync()?;
            let (mut a, mut b, mut c) = (vec![0i32; n], vec![0f32; n], vec![0f32; n]);
            top1.read(0, &mut a)?;
            lse.read(0, &mut b)?;
            tl.read(0, &mut c)?;
            if let Some(path) = crate::tune!("GLYD_ENGINE_DUMP_FILE") {
                let d = self.dumped.borrow();
                if !d.is_empty() {
                    let bytes: Vec<u8> = d.iter().flat_map(|row| row.iter().flat_map(|x| x.to_le_bytes())).collect();
                    std::fs::write(&path, bytes).map_err(|e| crate::ffi::Error(format!("{path}: {e}")))?;
                    eprintln!("{} residual rows ({} values each) written to {path}", d.len(), self.w.cfg.hidden);
                }
            }
            self.dump_row.set(None);
            let used = if at + n == tokens.len() { n - 1 } else { n };
            for i in 0..used {
                top.push(a[i] as u32);
                lp.push(c[i] - b[i]);
            }
            self.n += n;
            at += n;
        }
        self.kv.check_status()?;
        Ok((top, lp))
    }

    /// One prompt chunk (at most `chunk` tokens, from an empty cache), each launch followed by an event: the GPU's time by kind of work as (label, milliseconds, launches), slowest first, and the total.
    pub fn profile_prefill(&mut self, tokens: &[u32]) -> Result<Vec<(String, f64, usize)>> {
        let n = tokens.len();
        if n == 0 || n > self.chunk {
            return err(format!("{n} tokens to profile, a chunk takes 1 to {}", self.chunk));
        }
        self.reset();
        let pos: Vec<i32> = (0..n as i32).collect();
        let slots: Vec<i64> = (0..n as i64).collect();
        let lens: Vec<i32> = (1..=n as i32).collect();
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        self.slots.write(0, &slots)?;
        self.lens.write(0, &lens)?;
        let (splits, flash) = self.chunk_attention(0, n)?;
        let start = Ev::new()?;
        // SAFETY: as `prefill`.
        unsafe {
            *self.marks.borrow_mut() = Some(Vec::new());
            start.record(self.dev.stream)?;
            self.embed(tokens)?;
            self.mark("embedding rows")?;
            self.forward(n, self.pos.ptr, self.slots.ptr, self.lens.ptr, 0, splits, true, flash, None)?;
        }
        self.dev.sync()?;
        self.n = n;
        self.cached = tokens.to_vec();
        let marks = self.marks.borrow_mut().take().unwrap_or_default();
        let mut acc: Vec<(String, f64, usize)> = Vec::new();
        let mut total = 0.0;
        let mut prev = &start;
        for (label, e) in &marks {
            let ms = e.since(prev)? as f64;
            prev = e;
            total += ms;
            match acc.iter_mut().find(|a| a.0 == *label) {
                Some(a) => {
                    a.1 += ms;
                    a.2 += 1;
                }
                None => acc.push((label.to_string(), ms, 1)),
            }
        }
        acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        acc.push(("total".to_string(), total, 0));
        Ok(acc)
    }

    /// Eager decode steps (the cache holds the prompt), each launch followed by an event: the GPU's time by kind of work, over `steps` steps,
    /// as (label, milliseconds a step, launches a step), slowest first, with the embedding's two launches and the step's total.
    pub fn profile_steps(&mut self, mut token: u32, steps: usize) -> Result<Vec<(String, f64, usize)>> {
        let mut acc: Vec<(String, f64, usize)> = Vec::new();
        let mut total = 0.0;
        for _ in 0..steps {
            self.dev.sync()?;
            let state = [self.n as i32, (self.n + 1) as i32];
            let slot = [self.n as i64];
            let start = Ev::new()?;
            // SAFETY: as `step`.
            unsafe {
                let p = self.host.ptr as *mut u8;
                std::ptr::copy_nonoverlapping(state.as_ptr() as *const u8, p.add(64), 8);
                std::ptr::copy_nonoverlapping(slot.as_ptr() as *const u8, p.add(72), 8);
                copy_h2d_async(self.st.ptr, p.add(64) as *const _, 16, self.dev.stream)?;
                *self.marks.borrow_mut() = Some(Vec::new());
                start.record(self.dev.stream)?;
                self.embed(&[token])?;
                self.mark("embedding row")?;
                self.forward(1, self.st.ptr, self.st.ptr + 8, self.st.ptr + 4, self.kv.nblocks, self.splits_dec, true, None, None)?;
            }
            self.dev.sync()?;
            self.n += 1;
            let marks = self.marks.borrow_mut().take().unwrap_or_default();
            let mut prev = &start;
            for (label, e) in &marks {
                let ms = e.since(prev)? as f64;
                prev = e;
                total += ms;
                match acc.iter_mut().find(|a| a.0 == *label) {
                    Some(a) => {
                        a.1 += ms;
                        a.2 += 1;
                    }
                    None => acc.push((label.to_string(), ms, 1)),
                }
            }
            let mut s = [0i32; 1];
            self.tok.read(0, &mut s)?;
            token = s[0] as u32;
        }
        let mut out: Vec<(String, f64, usize)> = acc.into_iter().map(|(l, ms, n)| (l, ms / steps as f64, n / steps)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out.push(("total".to_string(), total / steps as f64, 0));
        Ok(out)
    }

    /// A decode step of `b` sequences of `len` tokens each, timed launch by launch, `steps` times: the same table as `profile_steps`, per step.
    pub fn profile_rows(&mut self, b: usize, len: usize, steps: usize) -> Result<Vec<(String, f64, usize)>> {
        let per = (len + steps) / 16 + 2;
        if b * per > self.pool_pages() {
            return err(format!("{b} sequences of {len} tokens need {} blocks, the cache has {}", b * per, self.pool_pages()));
        }
        let pages: Vec<Vec<i32>> = (0..b).map(|k| (0..per).map(|j| (k * per + j) as i32).collect()).collect();
        for (k, pg) in pages.iter().enumerate() {
            let t: Vec<u32> = (0..len).map(|i| 1 + ((k * 7919 + i * 104_729) % 4000) as u32).collect();
            let mut at = 0;
            while at < len {
                let n = self.chunk.min(len - at);
                self.prefill_chunk(pg, at, &t[at..at + n], at + n == len)?;
                at += n;
            }
        }
        let mut acc: Vec<(String, f64, usize)> = Vec::new();
        let mut total = 0.0;
        for st in 0..steps {
            self.dev.sync()?;
            let rows: Vec<Row> = pages.iter().enumerate().map(|(k, pg)| Row { token: 1000 + k as u32, pos: len + st, pages: pg }).collect();
            let start = Ev::new()?;
            *self.marks.borrow_mut() = Some(Vec::new());
            start.record(self.dev.stream)?;
            self.decode_batch(&rows)?;
            let marks = self.marks.borrow_mut().take().unwrap_or_default();
            let mut prev = &start;
            for (label, e) in &marks {
                let ms = e.since(prev)? as f64;
                prev = e;
                total += ms;
                match acc.iter_mut().find(|a| a.0 == *label) {
                    Some(a) => {
                        a.1 += ms;
                        a.2 += 1;
                    }
                    None => acc.push((label.to_string(), ms, 1)),
                }
            }
        }
        let mut out: Vec<(String, f64, usize)> = acc.into_iter().map(|(l, ms, n)| (l, ms / steps as f64, n / steps)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out.push(("total".to_string(), total / steps as f64, 0));
        Ok(out)
    }

    /// One decode step: `token` at the next position; the token that follows it.
    pub fn step(&mut self, token: u32, logprob: bool) -> Result<Sample> {
        if self.n + 1 > self.kv.tokens() {
            return err(format!("the cache holds {} tokens", self.kv.tokens()));
        }
        self.check_window(self.n + 1)?;
        let state = [self.n as i32, (self.n + 1) as i32];
        let slot = [self.n as i64];
        // SAFETY: pinned staging copied before the launches in stream order; live buffers.
        unsafe {
            let p = self.host.ptr as *mut u8;
            std::ptr::copy_nonoverlapping(state.as_ptr() as *const u8, p.add(64), 8);
            std::ptr::copy_nonoverlapping(slot.as_ptr() as *const u8, p.add(72), 8);
            copy_h2d_async(self.st.ptr, p.add(64) as *const _, 16, self.dev.stream)?;
            self.embed(&[token])?;
            match &self.graph {
                Some(g) => g.launch(self.dev.stream)?,
                None => self.forward(1, self.st.ptr, self.st.ptr + 8, self.st.ptr + 4, self.kv.nblocks, self.splits_dec, true, None, None)?,
            }
        }
        self.n += 1;
        self.cached.push(token);
        self.queue_head()?;
        let s = self.read_sample(logprob)?;
        let w = self.head_words();
        if let Err(e) = self.kv.check_head(&w) {
            // (what the cache holds of this sequence is not what was written: nothing of it is kept for a next prompt to begin with)
            self.reset();
            return Err(e);
        }
        Ok(s)
    }

    /// The last step's logits, on the host as f32 (what a sampler draws from).
    pub fn read_logits(&mut self) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut raw = vec![0u16; self.w.cfg.vocab];
        self.logits.read(0, &mut raw)?;
        Ok(raw.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// The tokens a chat can hold: the cache's, within the positions its tables cover.
    pub fn capacity(&self) -> usize {
        self.window.min(self.kv.max_seq())
    }

    /// The prompt through the model for a conversation: what it shares with the tokens already in the cache (the earlier turns) is kept, the rest run (at least the last token, for the
    /// logits). The token that follows it.
    pub fn prefill_session(&mut self, prompt: &[u32], logprob: bool) -> Result<Sample> {
        let shared = self.cached.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1));
        self.n = shared;
        self.cached.truncate(shared);
        self.prefill(&prompt[shared..], logprob)
    }

    /// A reply to `prompt`: its tokens as they are chosen (`each` gets each one and says whether to go on: false stops, as a client that has gone or a stop string does), with how it ended.
    /// Sampling as `o.params` says: the largest logit on the GPU where nothing else is asked for, else the logits' copy to the host and a draw.
    pub fn reply(&mut self, prompt: &[u32], o: &Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, Finish)> {
        if prompt.is_empty() {
            return err("an empty prompt");
        }
        if prompt.len() + 1 > self.capacity() {
            return err(format!("{} tokens, where the context holds {}", prompt.len(), self.capacity()));
        }
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        let mut s = self.prefill_session(prompt, false)?;
        let mut out = Vec::new();
        loop {
            let tok = if greedy { s.token } else { sampler.sample(&self.read_logits()?) };
            if !o.ignore_eos && self.w.cfg.eos.contains(&(tok as i64)) {
                return Ok((out, Finish::Stop));
            }
            out.push(tok);
            if !each(tok) {
                return Ok((out, Finish::Stopped));
            }
            if out.len() >= o.max_new {
                return Ok((out, Finish::Length));
            }
            if self.n + 1 > self.capacity() {
                return Ok((out, Finish::Length));
            }
            s = self.step(tok, false)?;
        }
    }

    /// Greedy text: the prompt, then up to `max_new` tokens (stopping at an end-of-sequence token unless `ignore_eos`).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, ignore_eos: bool, logprob: bool) -> Result<Vec<Sample>> {
        self.generate_with(prompt, max_new, ignore_eos, logprob, &mut |_| {})
    }

    /// `generate`, calling `each` with every token as it is sampled.
    pub fn generate_with(&mut self, prompt: &[u32], max_new: usize, ignore_eos: bool, logprob: bool, each: &mut dyn FnMut(&Sample)) -> Result<Vec<Sample>> {
        self.reset();
        let mut out = Vec::with_capacity(max_new);
        let mut s = self.prefill(prompt, logprob)?;
        each(&s);
        out.push(s);
        while out.len() < max_new {
            if !ignore_eos && self.w.cfg.eos.contains(&(s.token as i64)) {
                break;
            }
            s = self.step(s.token, logprob)?;
            each(&s);
            out.push(s);
        }
        self.kv.check_status()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{embed_plan, read_candidates, HOST_TOPK_TOK, HOST_TOPK_VAL, MAX_ROWS, TOPK};

    #[test]
    fn embedding_groups_once_and_rows_gathered() {
        // tokens 70 and 71 share group 1; 5 and 130 are groups 0 and 2; with two groups a batch: batch 0 holds groups 0 and 1, batch 1 group 2
        let (groups, batches) = embed_plan(&[70, 5, 71, 130, 70], 2);
        assert_eq!(groups, vec![0, 1, 2]);
        assert_eq!(batches[0], vec![(0, 64 + 6), (1, 5), (2, 64 + 7), (4, 64 + 6)]);
        assert_eq!(batches[1], vec![(3, 2)]);
        assert_eq!(embed_plan(&[], 4).1.len(), 0);
    }

    /// A row's candidates are read from where the copies put them: the Qwen3 engine's rows, a step's TOPK of each one after the other from HOST_TOPK_TOK and HOST_TOPK_VAL, and the one row of the lossy engines' `TopGpu`, its tokens and
    /// then its logits from the start.
    #[test]
    fn candidates_are_read_where_the_copies_put_them() {
        let (tok, val) = (HOST_TOPK_TOK / 4, HOST_TOPK_VAL / 4);
        let mut words = vec![0u32; val + MAX_ROWS * TOPK];
        for row in 0..MAX_ROWS {
            for j in 0..TOPK {
                words[tok + row * TOPK + j] = (1000 * row + j) as u32;
                words[val + row * TOPK + j] = (row as f32 - j as f32 / 8.0).to_bits();
            }
        }
        for row in [0, 1, MAX_ROWS - 1] {
            // SAFETY: the buffer holds MAX_ROWS rows of TOPK tokens and logits at the offsets, 4-byte aligned.
            let got = unsafe { read_candidates(words.as_ptr() as *const u8, HOST_TOPK_TOK, HOST_TOPK_VAL, row) };
            assert_eq!(got, (0..TOPK).map(|j| ((1000 * row + j) as u32, row as f32 - j as f32 / 8.0)).collect::<Vec<_>>(), "row {row}");
        }
        let mut one = vec![0u32; 2 * TOPK];
        for j in 0..TOPK {
            one[j] = 7 * j as u32;
            one[TOPK + j] = (-(j as f32)).to_bits();
        }
        // SAFETY: as above, one row.
        let got = unsafe { read_candidates(one.as_ptr() as *const u8, 0, 4 * TOPK, 0) };
        assert_eq!(got, (0..TOPK).map(|j| (7 * j as u32, -(j as f32))).collect::<Vec<_>>());
    }
}
