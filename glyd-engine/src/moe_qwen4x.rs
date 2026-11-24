//! Qwen3.8-Flash-Next's mixture-of-experts layer with its routed experts in the expert cache (`QxMoeCb`, the `MoeBlockQx` model_qwen4x.rs's layers call): the router (softmax over the 512 experts, the top 10 renormalised;
//! a pruned expert never chosen), the rotation of the layer's input, the experts' two products over the cache's places (kernels/qwen4x.cu `qx_cb_*`), the weighted sum and the rotation back, the shared expert times
//! its gate. The experts are glyd-hbr-codes/2 rows at 2,560 x 640 (cb2, cb3, cb4 or ternary, an expert's own format; ~/glyd-scratch/qwen/qwen_pack.py writes them), every layer's file mapped as the cache's host tier
//! (src/offload.rs): the GPU holds `slots` experts of a layer in places, copied in as the router asks for them, and the host computes a miss from its memory (src/cpu_cb.rs) or copies it, as the store plans.

use crate::cb_pack::{CbPack, LayerMap};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result, Stream};
use crate::launch;
use crate::model_qwen4x::{moe_cfg, EngineQx, MoeBlockQx};
use crate::moe::{route_with, select_with, MoeCfg, MoeScratch};
use crate::offload::{Call, ExpertStore, HostMem, Stage};
use crate::qwen4x::ConfigQx;
use crate::weights::{upload_bytes, Layout};
use glyd_gpu::safetensors::Checkpoint;
use std::cell::{Cell, OnceCell};
use std::rc::Rc;
use std::sync::Arc;

/// The width and the neurons the kernels are made for.
const H: usize = 2560;
const I: usize = 640;
/// The most pair-runs (pairs times runs of neurons) the partial sums of a call hold, and the most tokens a pass of the per-pair products takes (the host's pairs' limit: a longer call streams its experts).
const PART_ROWS: usize = 4096;
const PAIR_TOKENS: usize = 1024;
/// Qwen's SwiGLU has no clamp (the host's kernels take a finite one).
const NO_CLAMP: f32 = f32::MAX;
/// Rows a streamed call has at most for the host to take a share of its experts (`stream_experts`).
const HOST_ROWS: usize = 4096;
/// The rows at the end of a prompt whose experts are placed after it (`rewarm`), and how many of a layer's at most: a layer's hint is about 60 MB copied, beside the decode's first steps.
const WARM_ROWS: usize = 1024;
/// How many of a layer's places the prompt's experts take after it: after a long prompt (`LONG_ROWS` rows or more) all of them, copied before the decode's first step (`ExpertStore::drain`), so that its decode
/// starts with places holding what the prompt's end chose, not places a long prompt lent empty and others stale (decode after Strata's 32K prompt 8.1 -> 7.2 ms a token, its first token 0.6 s later); after a
/// shorter one `WARM_TOP`, copied beside the decode.
const WARM_TOP: usize = 32;
const LONG_ROWS: usize = 8192;
/// Tokens a call has at least for the streaming prefill (below, the cache's per-pair products with the host's misses).
pub(crate) const STREAM_MIN: usize = crate::offload::CPU_TOKENS + 1;

/// The kernels of kernels/qwen4x.cu's codebook rows with typed launches.
pub struct KernQc {
    rot_in: crate::kt::qx_cb_rot_in,
    local: crate::kt::glm_cb_local,
    up: crate::kt::qx_cb_up,
    down: crate::kt::qx_cb_down,
    /// the products with the places' addresses read from the GPU's memory (`products_pp`: a graph's launches)
    upi: crate::kt::qx_cb_upi,
    downi: crate::kt::qx_cb_downi,
    downti: crate::kt::qx_cb_downti,
    /// the second product of a call of a few rows on the tensor cores, a place's pairs together, for calls of `tc` to 8 rows (0: none; GLYD_QX_TC=N in a development build)
    downt: crate::kt::qx_cb_downt,
    pub tc: usize,
    sum_rot: crate::kt::qx_cb_sum_rot,
    back: crate::kt::qx_cb_back,
    /// the router and the rotation in one launch (a decode step's front: `qx_router_rot`)
    router_rot: crate::kt::qx_router_rot,
    /// its count of finished blocks (the last resets it) and of its calls (the host word's value), the host word its last block moves and the value the host saw last
    counter: Buf,
    flag: crate::dev::Pinned,
    seen: Cell<u32>,
    /// the host's word for a call in parts whose next graph is queued before the host's sums are made (`QxMoeCb::host_then`): the front's count the sums are for, then whether there are any; qx_cb_back waits for it
    hword: crate::dev::Pinned,
    sh1: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_sh1_1>>; 4],
    sh2: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_sh2_1>>; 4],
    add: crate::kt::qx_add_bf16,
    swiglu: crate::kt::qx_swiglu,
    gate_rows: crate::kt::qx_gate_rows,
    i8bf16: crate::kt::qx_i8_bf16,
    small: crate::kt::small_linear,
}

impl KernQc {
    pub fn load(dev: &Dev) -> Result<KernQc> {
        Ok(KernQc {
            rot_in: dev.typed()?,
            local: dev.typed()?,
            up: dev.typed()?,
            down: dev.typed()?,
            upi: dev.typed()?,
            downi: dev.typed()?,
            downti: dev.typed()?,
            downt: dev.typed()?,
            tc: crate::tune!("GLYD_QX_TC").and_then(|v| v.parse().ok()).unwrap_or(2),
            sum_rot: dev.typed()?,
            back: dev.typed()?,
            router_rot: dev.typed()?,
            counter: Buf::zeroed(8)?,
            flag: {
                let f = crate::dev::Pinned::mapped(64)?;
                // SAFETY: the mapped word is ours, written before any kernel names it.
                unsafe { std::ptr::write_volatile(f.ptr as *mut u32, 0) };
                f
            },
            seen: Cell::new(0),
            hword: {
                let f = crate::dev::Pinned::mapped(64)?;
                // SAFETY: the mapped words are ours, written before any kernel names them.
                unsafe { std::ptr::write_bytes(f.ptr as *mut u8, 0, 64) };
                f
            },
            sh1: [
                dev.by_args::<crate::kt::qx_sh1_1>()?,
                dev.by_args::<crate::kt::qx_sh1_2>()?,
                dev.by_args::<crate::kt::qx_sh1_4>()?,
                dev.by_args::<crate::kt::qx_sh1_8>()?,
            ],
            sh2: [
                dev.by_args::<crate::kt::qx_sh2_1>()?,
                dev.by_args::<crate::kt::qx_sh2_2>()?,
                dev.by_args::<crate::kt::qx_sh2_4>()?,
                dev.by_args::<crate::kt::qx_sh2_8>()?,
            ],
            add: dev.typed()?,
            swiglu: dev.typed()?,
            gate_rows: dev.typed()?,
            i8bf16: dev.typed()?,
            small: dev.typed()?,
        })
    }

    /// `rot_in` and `local` in one launch (the pairs of each token mapped by its first block; k at most 512).
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    /// The router's logits and choices and x' = R x in one launch (n at most 8, the model's width: the front of a decode step); the choices' numbers among the layer's kept experts into gids; the choices, their
    /// weights and x' also into the host's mapped buffers (`host`: ids, weights, x'); where `map` is given (the layer's places, `QxMoeCb`'s), each pair's place into `way`; the host's word moved at the end.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn router_rot(
        &self,
        s: Stream,
        x: u64,
        xs: usize,
        w: u64,
        logits: u64,
        n: usize,
        c: &MoeCfg,
        bias: u64,
        ids: u64,
        wts: u64,
        sg: u64,
        xr: u64,
        xrb: u64,
        local: u64,
        gids: u64,
        experts: usize,
        host: [u64; 3],
        map: u64,
        way: u64,
    ) -> Result<()> {
        if n == 0 || n > 8 || c.experts > 512 || c.router.top_k > 32 || c.router_hidden != H {
            return err(format!("the fused front on {n} rows of {} experts, {} a token, width {}", c.experts, c.router.top_k, c.router_hidden));
        }
        let blocks = c.experts.div_ceil(8) + 5 * n;
        launch!(self.router_rot, [blocks as u32, 1, 1], [512, 1, 1], s; x, xs as i32, w, logits, n as i32, c.experts as i32, H as i32, bias, ids, wts, c.router.top_k as i32, c.router.flags(), c.router.scale, self.counter.ptr, host[1], sg, xr, xrb, local, gids, experts as i32, host[0], host[2], self.flag.ptr as u64, map, way)
    }

    pub unsafe fn rot_in(&self, s: Stream, x: u64, xs: usize, sg: u64, xr: u64, xrb: u64, n: usize) -> Result<()> {
        launch!(self.rot_in, [n as u32, (H / 512) as u32, 1], [512, 1, 1], s; x, xs as i32, sg, xr, xrb)
    }

    /// The pairs' experts as the layer's numbers (-1: one the layer does not hold).
    ///
    /// # Safety
    /// As `rot_in`.
    pub unsafe fn local(&self, s: Stream, ids: u64, local: u64, out: u64, pairs: usize, experts: usize) -> Result<()> {
        launch!(self.local, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; ids, local, out, pairs as i32, experts as i32)
    }

    /// The first products of the pairs that have a place (`way`: each pair's place, -1 for none) into gu [pairs, 2 I].
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn up(&self, s: Stream, places: [u64; 2], strides: [usize; 2], way: u64, xr: u64, gu: u64, k: usize, pairs: usize, mixed: bool) -> Result<()> {
        launch!(self.up, [(2 * I / 32) as u32, pairs as u32, 1], [256, 1, 1], s; places[0], strides[0] as u64, places[1], strides[1] as u64, way, xr, gu, k as i32, mixed as i32)
    }

    /// The activation and the second products, `split` runs of neurons a pair, partial sums into part [pairs, split, H].
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn down(
        &self,
        s: Stream,
        places: [u64; 2],
        strides: [usize; 2],
        way: u64,
        gu: u64,
        part: u64,
        split: usize,
        pairs: usize,
        mixed: bool,
    ) -> Result<()> {
        launch!(self.down, [split as u32, pairs as u32, 1], [256, 1, 1], s; places[0], strides[0] as u64, places[1], strides[1] as u64, way, gu, part, split as i32, NO_CLAMP, mixed as i32)
    }

    /// `up` and `down` (`down_t` where `t`, the split then 1) with the places' two addresses read from `pp` (the GPU's memory), for a graph's launches.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn products_pp(
        &self,
        s: Stream,
        pp: u64,
        strides: [usize; 2],
        way: u64,
        xr: u64,
        gu: u64,
        part: u64,
        k: usize,
        pairs: usize,
        split: usize,
        t: bool,
        mixed: bool,
    ) -> Result<()> {
        launch!(self.upi, [(2 * I / 32) as u32, pairs as u32, 1], [256, 1, 1], s; pp, strides[0] as u64, strides[1] as u64, way, xr, gu, k as i32, mixed as i32)?;
        if t {
            return launch!(self.downti, [(H / 256) as u32, pairs as u32, 1], [256, 1, 1], s; pp, strides[0] as u64, strides[1] as u64, way, gu, part, NO_CLAMP, mixed as i32, pairs as i32);
        }
        launch!(self.downi, [split as u32, pairs as u32, 1], [256, 1, 1], s; pp, strides[0] as u64, strides[1] as u64, way, gu, part, split as i32, NO_CLAMP, mixed as i32)
    }

    /// The activations and the second products of a call of 2 to 8 rows on the tensor cores, a place's pairs together (gu: `up`'s sums), into part [pairs, 1, H]: one run a pair.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn down_t(&self, s: Stream, places: [u64; 2], strides: [usize; 2], way: u64, gu: u64, part: u64, pairs: usize, mixed: bool) -> Result<()> {
        launch!(self.downt, [(H / 256) as u32, pairs as u32, 1], [256, 1, 1], s; places[0], strides[0] as u64, places[1], strides[1] as u64, way, gu, part, NO_CLAMP, mixed as i32, pairs as i32)
    }

    /// y [n, H] bf16 = R^T (extra + the weighted sum of each token's pairs, their runs added): `skip` the pairs left out (the host's: in `extra`), 0 for none.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sum_rot(&self, s: Stream, part: u64, wts: u64, extra: u64, skip: u64, sg: u64, y: u64, k: usize, split: usize, n: usize) -> Result<()> {
        launch!(self.sum_rot, [n as u32, (H / 512) as u32, 1], [512, 1, 1], s; part, wts, extra, skip, sg, y, k as i32, split as i32)
    }
}

/// The runs a pair's neurons are cut into for `pairs` pairs: enough blocks to fill the GPU, the partial sums within `PART_ROWS` (each a divisor of 640 leaving a multiple of 4 neurons a run).
pub(crate) fn runs_for(pairs: usize) -> usize {
    let s = match pairs {
        0..=32 => 16,
        33..=64 => 8,
        65..=128 => 4,
        129..=256 => 2,
        _ => 1,
    };
    let mut s = s.min((PART_ROWS / pairs.max(1)).max(1));
    while !I.is_multiple_of(s) || !(I / s).is_multiple_of(4) {
        s -= 1;
    }
    s
}

/// What every layer's block shares: its scratch (the first call makes it), the experts a token takes now (0: the model's), the bytes of the pack loading may still ask the system to read ahead.
pub struct QxCbShared {
    scratch: OnceCell<QcScratch>,
    pub topk: Cell<usize>,
    pub read_ahead: Cell<u64>,
    /// Whether a prompt chunk of more than `CPU_TOKENS` tokens goes through the streaming prefill (src/cb2_prefill.rs: the experts with pairs streamed through a ring on the GPU, each multiplied by all its tokens on the
    /// tensor cores) instead of the cache's per-pair products.
    pub stream: Cell<bool>,
    /// The layers' mapped files with the ranges of each (offset, bytes) that the read-ahead budget took (the page tables of those are filled in the background after the load: `fill_page_tables`)
    pub maps: std::cell::RefCell<Vec<(Arc<crate::mapped::Mapping>, usize, usize)>>,
    /// Whether the layers' experts are copied into pinned host memory at load (`CbPack::pin_layer`) instead of read from the mapped files.
    pub pin_pack: Cell<bool>,
    /// A long prompt's experts are to be copied to the places before the decode starts (`whole_experts` sets it, `drain` waits for them).
    pub drain: Cell<bool>,
    /// The layers' places in all (bytes): what the GPU holds beside the always-on matrices (the decode's ceiling, src/qwen4x_serve.rs).
    pub places_bytes: Cell<usize>,
}

impl QxCbShared {
    pub fn new() -> Rc<QxCbShared> {
        Rc::new(QxCbShared {
            scratch: OnceCell::new(),
            topk: Cell::new(0),
            read_ahead: Cell::new(u64::MAX),
            stream: Cell::new(false),
            maps: std::cell::RefCell::new(Vec::new()),
            places_bytes: Cell::new(0),
            pin_pack: Cell::new(false),
            drain: Cell::new(false),
        })
    }

    /// The page tables of the layers' files (as far as the read-ahead budget went) filled on a thread of its own: a host expert's first read of a page is then no page fault (a call of one pair 100 -> 46 us). The thread
    /// ends when it is done; the files stay mapped as long as the layers.
    pub fn fill_page_tables(&self) {
        let maps: Vec<(Arc<crate::mapped::Mapping>, usize, usize)> = self.maps.borrow_mut().drain(..).collect();
        if maps.is_empty() {
            return;
        }
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let (mut bytes, mut ok) = (0usize, true);
            for (m, off, len) in maps {
                ok &= m.populate(off, len);
                bytes += len;
            }
            eprintln!(
                "[load] {:.1} GB of the pack mapped in {:.1} s{}",
                bytes as f64 / 1e9,
                t0.elapsed().as_secs_f64(),
                if ok { "" } else { " (the system refused some: their first reads fault)" }
            );
        });
    }
}

/// The byte ranges of `map` (offset, length) that the placement's experts are in, in order, those less than 1 MiB apart joined.
fn used_ranges(p: &crate::offload::Placement, map: &crate::mapped::Mapping) -> Vec<(usize, usize)> {
    let (base, end) = (map.ptr as usize, map.ptr as usize + map.len);
    let mut r: Vec<(usize, usize)> = (0..2)
        .flat_map(|k| p.at[k].iter().zip(&p.len[k]).map(|(&a, &l)| (a as usize, l as usize)))
        .filter(|&(a, l)| a >= base && a + l <= end)
        .map(|(a, l)| (a - base, l))
        .collect();
    r.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (a, l) in r {
        match out.last_mut() {
            Some((o, n)) if a <= *o + *n + (1 << 20) => *n = (*n).max(a + l - *o),
            _ => out.push((a, l)),
        }
    }
    out
}

struct QcScratch {
    kern: KernQc,
    xr: Buf,
    xrb: Buf,
    gids: Buf,
    gu: Buf,
    part: Buf,
    // the shared expert's: its activations [n, I], its gate (fp32 [8]), its output [n, H], a prompt chunk's first product [n, 2 I] and gate logits [n], and its matrices made bf16
    sact: Buf,
    sg: Buf,
    sh: Buf,
    sgu: Buf,
    slog: Buf,
    sw: Buf,
    /// the host's sums of a call in parts whose next graph is queued before they are made (fp32 [n, H], the host's memory, which qx_cb_back reads: `QxMoeCb::host_then`)
    hsum: crate::dev::Pinned,
    /// the streaming prefill's kernels, and a prompt's rows taken layer by layer (`MoeBlockQx::whole`)
    pre: OnceCell<crate::cb2_prefill::Cb2Kern>,
    lm: std::cell::RefCell<Option<Whole>>,
}

/// A prompt's rows (up to `rows`) as the layer-by-layer prefill holds them between its parts: x' (bf16 [rows, H]), the router's weights (fp32 [rows, k]) and choices as the layer's numbers (int64), the experts' sum
/// in the rotated basis (fp32 [rows, H]), the shared expert's output (bf16 [rows, H]). One set for every layer (one layer at a time).
struct Whole {
    rows: usize,
    xrb: Buf,
    wts: Buf,
    gids: Buf,
    y: Buf,
    sh: Buf,
}

impl Whole {
    /// Device bytes of the set for `rows` rows of `k` pairs.
    pub fn bytes(rows: usize, k: usize) -> usize {
        rows * (H * 2 + k * 12 + H * 4 + H * 2)
    }
}

/// Device bytes the layer-by-layer prefill holds for `rows` rows beside the engine's (`Whole`).
pub fn whole_bytes(rows: usize, k: usize) -> usize {
    Whole::bytes(rows, k)
}

/// Device bytes a call of the layer-by-layer prefill makes for `rows` rows beside the engine's: the experts' set, and the streaming prefill's ring (sixteen slots of eight experts of up to 2.6 MB), plan and activations
/// and its host share's rows (fp32 in and out).
pub fn whole_call_bytes(rows: usize, k: usize) -> usize {
    Whole::bytes(rows, k) + crate::cb2_prefill::Cb2Kern::device_bytes(rows * k, 2_600_000) + rows.min(8192) * H * 8
}

/// The addresses of the set's buffers (`Whole`).
#[derive(Clone, Copy)]
struct WholePtrs {
    rows: usize,
    xrb: u64,
    wts: u64,
    gids: u64,
    y: u64,
    sh: u64,
}

/// The shared expert as int8 rows with an fp32 scale each (`max |w| / 127`): gate and up [2 I, H], down [H, I]; its gate [H] bf16.
pub struct QxShared {
    gu_q: Buf,
    gu_s: Buf,
    dn_q: Buf,
    dn_s: Buf,
    gate: Buf,
    inter: usize,
}

impl QxShared {
    fn load(ck: &Checkpoint, p: &str, h: usize, inter: usize) -> Result<QxShared> {
        let read = |name: &str, n: usize| -> Result<Vec<u16>> {
            let mut v = vec![0u16; n];
            ck.read_bf16(name, &mut v).map_err(|e| Error(format!("{name}: {e}")))?;
            Ok(v)
        };
        let mut gu = read(&format!("{p}.shared_expert.gate_proj.weight"), inter * h)?;
        gu.extend(read(&format!("{p}.shared_expert.up_proj.weight"), inter * h)?);
        let dn = read(&format!("{p}.shared_expert.down_proj.weight"), h * inter)?;
        let (gq, gs) = crate::model_qwen4x::quant_rows(&gu, 2 * inter, h, 2 * inter, h);
        let (dq, ds) = crate::model_qwen4x::quant_rows(&dn, h, inter, h, inter);
        let bytes = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
        Ok(QxShared {
            gu_q: upload_bytes(&bytes(&gq))?,
            gu_s: upload_bytes(crate::weights::as_bytes(&gs))?,
            dn_q: upload_bytes(&bytes(&dq))?,
            dn_s: upload_bytes(crate::weights::as_bytes(&ds))?,
            gate: upload_bytes(&ck.read(&format!("{p}.shared_expert_gate.weight")).map_err(|e| Error(format!("{p}.shared_expert_gate.weight: {e}")))?)?,
            inter,
        })
    }

    fn bytes(&self) -> usize {
        self.gu_q.bytes + self.gu_s.bytes + self.dn_q.bytes + self.dn_s.bytes + self.gate.bytes
    }
}

/// A mixture-of-experts layer with its routed experts in the expert cache.
pub struct QxMoeCb {
    cfg: MoeCfg,
    router: Buf,
    /// -1e30 for an expert the pack left out (none where it keeps them all)
    bias: Option<Buf>,
    shared: QxShared,
    signs: Buf,
    local_of: Buf,
    experts: usize,
    /// the places on the GPU (codes, codebooks: `slots` each, at most `full` as loaded: fewer while other sequences take memory, `resize_places`); empty while lent to a long prompt's buffers (`release_places`)
    places: std::cell::RefCell<[Buf; 2]>,
    slots: Cell<usize>,
    full: usize,
    /// the experts the last rows of the prompt just read chose most, least first (`whole_experts`): placed after the prompt (`rewarm`) so that the decode starts with them
    warm: std::cell::RefCell<Vec<u16>>,
    #[allow(dead_code)]
    map: Arc<crate::mapped::Mapping>,
    /// the kept experts as the streaming prefill takes them (addresses in the mapping)
    descs: Vec<crate::cb2_prefill::ExpertDesc>,
    strides: [usize; 2],
    mixed: bool,
    /// whether every kept expert is in a format the tensor-core second products take (two-bit and format q)
    tc_ok: bool,
    /// the places of the kept experts as a decode step's products read them on the GPU (`mapped`; GLYD_QX_MAP=0 in a development build: never): the host's mapped memory, an int32 place each (-1: none), kept by
    /// `ExpertStore::run_parts_mapped` and `map_sync`; whether calls other than those have changed the cache since (the map made again before the next step in parts, `sync_map`), and whether experts copied in are
    /// not in it yet; the places' two addresses for the graphs' launches
    map_on: bool,
    pmap: crate::dev::Pinned,
    pmap_stale: Cell<bool>,
    pmap_behind: Cell<bool>,
    pp: Buf,
    store: Rc<ExpertStore>,
    layer: usize,
    common: Rc<QxCbShared>,
}

impl QxMoeCb {
    /// Layer `l` (`l` = layers: the MTP head's): the router and the shared expert's gate (bf16), the shared expert at `level`, the layer's rotation, and the experts: `pack`'s file mapped as the cache's host tier, with
    /// `slots` places on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        l: usize,
        c: &ConfigQx,
        ck: &Checkpoint,
        level: Layout,
        pack: &CbPack,
        store: &Rc<ExpertStore>,
        slots: usize,
        common: &Rc<QxCbShared>,
    ) -> Result<QxMoeCb> {
        if (c.hidden, c.inter, pack.hidden, pack.inter) != (H, I, H, I) {
            return err(format!(
                "experts of {} neurons over {} (the pack's {} over {}): the kernels take {I} over {H}",
                c.inter, c.hidden, pack.inter, pack.hidden
            ));
        }
        let kept = pack.kept(l).ok_or_else(|| Error(format!("{}: no experts of layer {l}", pack.dir.display())))?.to_vec();
        let p = format!("{}.mlp", c.layer_name(l));
        let tensor = |name: &str| -> Result<Buf> { upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?) };
        let router = tensor(&format!("{p}.gate.weight"))?;
        let mut local = vec![-1i32; c.experts];
        for (i, &e) in kept.iter().enumerate() {
            local[e] = i as i32;
        }
        let bias = if kept.len() < c.experts {
            Some(upload_bytes(&local.iter().flat_map(|&x| if x < 0 { -1e30f32 } else { 0.0 }.to_le_bytes()).collect::<Vec<u8>>())?)
        } else {
            None
        };
        let _ = level; // (the shared expert is held as int8 rows whatever the engine's layout)
        let shared = QxShared::load(ck, &p, c.hidden, c.shared_inter)?;
        let slots = slots.clamp(1, kept.len());
        let lm = pack.map_layer(l)?;
        // (pinned: the experts copied into pinned host memory now, the files' pages given back)
        let lm = if common.pin_pack.get() { pack.pin_layer(l, lm)? } else { lm };
        let (mixed, strides) = (lm.place.is_some(), lm.strides);
        let places = [Buf::new(slots * strides[0])?, Buf::new(slots * strides[1])?];
        common.places_bytes.set(common.places_bytes.get() + places[0].bytes + places[1].bytes);
        let LayerMap { map, place, starts, descs, .. } = lm;
        // (the file read ahead in the background while the rest of the model loads, as far as the budget goes; nothing to read for pinned bytes; with a sidecar only the experts of the file the sidecar leaves it:
        // the rest is never read, and would only push other pages out of the computer's memory)
        let side = place.as_ref().is_some_and(|p| !p.maps.is_empty());
        let ranges = match place.as_ref().filter(|_| side) {
            Some(p) => used_ranges(p, &map),
            None => vec![(0, map.len)],
        };
        for (off, len) in ranges {
            let ahead = if map.is_pinned() { 0 } else { (len as u64).min(common.read_ahead.get()) };
            common.read_ahead.set(common.read_ahead.get().saturating_sub(ahead));
            if ahead > 0 {
                map.will_need(off, ahead as usize);
                common.maps.borrow_mut().push((map.clone(), off, ahead as usize));
            }
        }
        if let Some(p) = place.as_ref() {
            for m in p.maps.iter().skip(1).filter(|m| !m.is_pinned()) {
                let ahead = (m.len as u64).min(common.read_ahead.get());
                common.read_ahead.set(common.read_ahead.get().saturating_sub(ahead));
                m.will_need(0, ahead as usize);
                common.maps.borrow_mut().push((m.clone(), 0, ahead as usize));
            }
        }
        let host = if side { [HostMem::none(), HostMem::none()] } else { [HostMem::mapped(&map, starts.0), HostMem::mapped(&map, starts.1)] };
        store.add_layer_mapped(
            l,
            kept.len(),
            slots,
            host,
            place,
            [strides[0] as u64, strides[1] as u64],
            [places[0].ptr, places[1].ptr],
            Some((NO_CLAMP, H, I)),
        )?;
        let signs = pack.signs(l).ok_or_else(|| Error(format!("no signs of layer {l}")))?;
        let tc_ok = descs.iter().all(|d| [d.up, d.down].iter().all(|f| matches!(f, crate::cpu_cb::Fmt::Cb2 | crate::cpu_cb::Fmt::Tq)));
        let pp = upload_bytes(crate::weights::as_bytes(&[places[0].ptr, places[1].ptr]))?;
        Ok(QxMoeCb {
            cfg: moe_cfg(c),
            router,
            bias,
            shared,
            signs: upload_bytes(&signs.iter().map(|&s| s as u8).collect::<Vec<u8>>())?,
            local_of: upload_bytes(&local.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?,
            experts: c.experts,
            places: std::cell::RefCell::new(places),
            slots: Cell::new(slots),
            full: slots,
            warm: std::cell::RefCell::new(Vec::new()),
            map,
            descs,
            strides,
            mixed,
            tc_ok,
            map_on: crate::tune!("GLYD_QX_MAP").is_none_or(|v| v != "0"),
            pmap: crate::dev::Pinned::mapped(kept.len() * 4)?,
            pmap_stale: Cell::new(true),
            pmap_behind: Cell::new(false),
            pp,
            store: store.clone(),
            layer: l,
            common: common.clone(),
        })
    }

    /// The router's logits and choices of the n rows at `h` into the scratch: a prompt chunk's logits on the tensor cores (fp32 sums, `Kern::gemm_f32`), a few rows' by the router's own kernel (`route_with`).
    ///
    /// # Safety
    /// As `MoeBlockQx::forward`.
    unsafe fn route(&self, e: &EngineQx, c: &MoeCfg, sc: &MoeScratch, h: u64, n: usize) -> Result<()> {
        if n > 8 && n <= sc.max_tokens {
            e.kern.gemm_f32(e.dev.stream, h, self.router.ptr, sc.logits.ptr, n as i32, c.experts as i32, H as i32)?;
            return select_with(c, self.bias.as_ref(), sc, e.dev.stream, n);
        }
        route_with(c, &self.router, self.bias.as_ref(), None, sc, e.dev.stream, h, H, n)
    }

    /// The MoE scratch every layer shares, and this module's beside it, made by the first call.
    fn scratch<'a>(&'a self, e: &'a EngineQx) -> Result<(&'a MoeScratch, &'a QcScratch)> {
        if e.moe_scratch.get().is_none() {
            let _ = e.moe_scratch.set(MoeScratch::routing(&e.dev, self.cfg.experts, self.cfg.router.top_k, H, e.front_rows())?);
        }
        let sc = e.moe_scratch.get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        if self.common.scratch.get().is_none() {
            // (the front's buffers for `front_rows`, the per-pair products' for the engine's chunk)
            let (n, k) = (e.front_rows(), self.cfg.router.top_k);
            let pass = e.chunk().min(PAIR_TOKENS) * k;
            let si = self.shared.inter;
            let made = QcScratch {
                kern: KernQc::load(&e.dev)?,
                xr: Buf::new(n * H * 4)?,
                xrb: Buf::new(n * H * 2)?,
                gids: Buf::new(n * k * 8)?,
                gu: Buf::new(pass * 2 * I * 4)?,
                part: Buf::new(PART_ROWS.max(pass) * H * 4)?,
                sact: Buf::new(n * si * 2)?,
                sg: Buf::new(8 * 4)?,
                sh: Buf::new(n * H * 2)?,
                sgu: Buf::new(n * 2 * si * 2)?,
                slog: Buf::new(n * 2)?,
                sw: Buf::new(2 * si * H * 2)?,
                hsum: crate::dev::Pinned::mapped(n * H * 4)?,
                pre: OnceCell::new(),
                lm: std::cell::RefCell::new(None),
            };
            let _ = self.common.scratch.set(made);
        }
        Ok((sc, self.common.scratch.get().ok_or_else(|| Error("the scratch is not made".into()))?))
    }

    /// `n` rows (at most `PAIR_TOKENS`): the router, x' = R x, the pairs' experts made resident or computed by the host (`ExpertStore::run_early`, the shared expert queued while the host plans), the products of the pairs
    /// with places, the weighted sum with the host's sums and R^T into `out`, the shared expert added.
    ///
    /// # Safety
    /// As `MoeBlockQx::forward`.
    unsafe fn run_rows(&self, e: &EngineQx, h: u64, n: usize, out: u64) -> Result<()> {
        let s = e.dev.stream;
        let mut c = self.cfg;
        if self.common.topk.get() != 0 {
            c.router.top_k = self.common.topk.get().min(self.cfg.router.top_k);
        }
        let (sc, cs) = self.scratch(e)?;
        let (k, pairs) = (c.router.top_k, n * c.router.top_k);
        let stream = n >= STREAM_MIN && self.common.stream.get();
        if n == 0 || n > e.chunk() || (!stream && n > PAIR_TOKENS) {
            return err(format!("the block on {n} rows, where a chunk is {} and a pass of the per-pair products {PAIR_TOKENS}", e.chunk()));
        }
        route_with(&c, &self.router, self.bias.as_ref(), None, sc, s, h, H, n)?;
        crate::tmark!(s, "router");
        cs.kern.rot_in(s, h, H, self.signs.ptr, cs.xr.ptr, cs.xrb.ptr, n)?;
        cs.kern.local(s, sc.ids.ptr, self.local_of.ptr, cs.gids.ptr, pairs, self.experts)?;
        if n >= STREAM_MIN && self.common.stream.get() {
            // a long chunk: every expert with pairs streamed through the GPU once and multiplied by all its tokens, the sums in the rotated basis into xr, then R^T and the shared expert
            self.shared_out(e, cs, h, n, cs.sh.ptr)?;
            crate::tmark!(s, "stream: shared expert");
            self.stream_experts(e, cs, n, cs.xrb.ptr, cs.gids.ptr, sc.wts.ptr, cs.xr.ptr)?;
            crate::tmark!(s, "stream: experts");
            cs.kern.sum_rot(s, 0, 0, cs.xr.ptr, 0, self.signs.ptr, out, 0, 1, n)?;
            self.shared_add(e, cs, out, n, cs.sh.ptr)?;
            crate::tmark!(s, "stream: sum");
            return Ok(());
        }
        let tc = self.tc(cs, n);
        let split = if tc { 1 } else { runs_for(pairs) };
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs };
        let places = self.place_ptrs();
        let mut early = || {
            self.shared_out(e, cs, h, n, cs.sh.ptr)?;
            crate::tmark!(s, "shared expert");
            Ok(())
        };
        self.store.run_early(self.layer, &e.dev, &call, &mut early, &mut |stage, wave, waves, hy| match stage {
            Stage::Products => {
                if !hy.gpu {
                    return Ok(());
                }
                cs.kern.up(s, places, self.strides, sc.way.ptr, cs.xr.ptr, cs.gu.ptr, k, pairs, self.mixed)?;
                if tc {
                    return cs.kern.down_t(s, places, self.strides, sc.way.ptr, cs.gu.ptr, cs.part.ptr, pairs, self.mixed);
                }
                cs.kern.down(s, places, self.strides, sc.way.ptr, cs.gu.ptr, cs.part.ptr, split, pairs, self.mixed)
            }
            Stage::Sum => {
                if wave + 1 == waves {
                    cs.kern.sum_rot(s, cs.part.ptr, sc.wts.ptr, hy.extra, hy.ids, self.signs.ptr, out, k, split, n)?;
                }
                Ok(())
            }
        })?;
        self.shared_add(e, cs, out, n, cs.sh.ptr)
    }

    /// The routed experts' sum (fp32 [n, H], rotated basis, written into `y`) of n rows with x' at `xrb` and their choices (`gids`: the layer's numbers, `wts`), every expert with pairs streamed through the GPU once and
    /// multiplied by all its rows on the tensor cores (`cb2_prefill_experts`; the experts in places read where they are).
    ///
    /// # Safety
    /// As `MoeBlockQx::forward`; the buffers hold n rows.
    #[allow(clippy::too_many_arguments)]
    unsafe fn stream_experts(&self, e: &EngineQx, cs: &QcScratch, n: usize, xrb: u64, gids: u64, wts: u64, y: u64) -> Result<()> {
        let s = e.dev.stream;
        if cs.pre.get().is_none() {
            let _ = cs.pre.set(crate::cb2_prefill::Cb2Kern::load_dims(&e.dev, H, I, NO_CLAMP)?);
        }
        let kern = cs.pre.get().ok_or_else(|| Error("the streaming prefill is not made".into()))?;
        kern.pinned.set(self.common.pin_pack.get());
        let at = self.store.resident_places(self.layer, s)?;
        let descs: Vec<crate::cb2_prefill::ExpertDesc> = self
            .descs
            .iter()
            .zip(&at)
            .map(|(d, &pl)| {
                let mut d = *d;
                if pl >= 0 && d.up != crate::cpu_cb::Fmt::Tern && d.down != crate::cpu_cb::Fmt::Tern {
                    let p = self.place_ptrs();
                    d.resident = Some((p[0] + pl as u64 * self.strides[0] as u64, p[1] + pl as u64 * self.strides[1] as u64));
                }
                d
            })
            .collect();
        // the host's share (experts of few pairs computed by the CPU while the GPU streams the rest) for calls of up to HOST_ROWS rows, where the copies bound the call: balanced by the calls' times from 12 rows on 12
        // threads (a longer call's GPU loop is bound by its products, and its rows' trips to the host and back cost more than the copies it saves); a development build's GLYD_CB2_HOST=R[:T] fixes it (0: none)
        let share = kern.host.get();
        match crate::tune!("GLYD_CB2_HOST") {
            Some(v) => {
                let mut it = v.split(':').map(|x| x.parse::<usize>().unwrap_or(0));
                let (r, t) = (it.next().unwrap_or(0), it.next().unwrap_or(12));
                kern.host.set((r > 0).then_some(crate::cb2_prefill::HostShare { max_rows: r, threads: t.max(1), auto: false }));
            }
            None if n > HOST_ROWS => kern.host.set(None),
            None => kern.host.set(Some(share.filter(|h| h.auto).unwrap_or(crate::cb2_prefill::HostShare { max_rows: 12, threads: 12, auto: true }))),
        }
        crate::cb2_prefill::cb2_prefill_experts(&e.dev, kern, &descs, n, self.cfg.router.top_k, xrb, gids, wts, y, s)?;
        if crate::tune!("GLYD_CB2_HOSTLOG").is_some() && kern.host.get().is_some() {
            let t = kern.took.get();
            eprintln!(
                "host share: layer {} rows {n}: {} experts, {} pairs, host {:.1} ms, GPU loop {:.1} ms",
                self.layer, t.experts, t.pairs, t.host_ms, t.gpu_ms
            );
        }
        Ok(())
    }

    /// Whether a call of n rows takes the tensor-core second products (`KernQc::down_t`): `tc` to 8 rows of a layer whose formats they take.
    fn tc(&self, cs: &QcScratch, n: usize) -> bool {
        self.tc_ok && cs.kern.tc > 0 && (cs.kern.tc..=8).contains(&n)
    }

    /// The runs of neurons the second products of a call of n rows leave a pair (the sum after them adds them): one for the tensor cores', else `runs_for`.
    fn split(&self, cs: &QcScratch, n: usize) -> usize {
        if self.tc(cs, n) {
            1
        } else {
            runs_for(n * self.cfg.router.top_k)
        }
    }

    /// The addresses of the places (codes, codebooks).
    fn place_ptrs(&self) -> [u64; 2] {
        let p = self.places.borrow();
        [p[0].ptr, p[1].ptr]
    }

    /// The places made again or given up: their addresses for the graphs' launches (`pp`), and the map made again before the next step in parts (the cache has been emptied or cut).
    fn places_moved(&self) -> Result<()> {
        self.pp.write(0, &self.place_ptrs())?;
        self.pmap_stale.set(true);
        Ok(())
    }

    /// Whether a decode step's products run in the front's graph by the map of places, the host taking the pairs without a place only (`ExpertStore::run_parts_mapped`): where the store serves the layer in parts in
    /// the mode Cpu (misses on the host, promotions behind the calls); else the host plans each call and queues its products (`run_parts`).
    fn mapped(&self) -> bool {
        self.map_on && self.store.offload().mode == crate::offload::Mode::Cpu && self.store.supports_parts(self.layer)
    }

    /// The map of places as the host writes it.
    ///
    /// # Safety
    /// No kernel that reads it is queued or running (a step's start, or a host part after its front's word).
    #[allow(clippy::mut_from_ref)] // the map is pinned host memory, not `self`'s; R15 (a sound device API) replaces this
    unsafe fn pmap_mut(&self) -> &mut [i32] {
        std::slice::from_raw_parts_mut(self.pmap.ptr as *mut i32, self.descs.len())
    }

    /// The layer-by-layer prefill's set for the prompt being read (`EngineQx::whole_now` rows), made by its first call (and freed after the prompt: `whole_release`).
    fn whole_set(&self, e: &EngineQx, cs: &QcScratch) -> Result<WholePtrs> {
        let (rows, k) = (e.whole_now(), self.cfg.router.top_k);
        if rows == 0 {
            return err("no prompt is being taken layer by layer");
        }
        let mut lm = cs.lm.borrow_mut();
        if lm.as_ref().is_none_or(|w| w.rows < rows) {
            *lm = None;
            *lm = Some(Whole {
                rows,
                xrb: Buf::new(rows * H * 2)?,
                wts: Buf::new(rows * k * 4)?,
                gids: Buf::new(rows * k * 8)?,
                y: Buf::new(rows * H * 4)?,
                sh: Buf::new(rows * H * 2)?,
            });
        }
        let w = lm.as_ref().expect("the set");
        Ok(WholePtrs { rows: w.rows, xrb: w.xrb.ptr, wts: w.wts.ptr, gids: w.gids.ptr, y: w.y.ptr, sh: w.sh.ptr })
    }
}

impl QxMoeCb {
    /// The shared expert on the n rows at `h` into the scratch's `sh` (gated): two fused launches for a few rows, else the int8 rows made bf16 and the engine's dense products.
    ///
    /// # Safety
    /// As `MoeBlockQx::forward`.
    unsafe fn shared_out(&self, e: &EngineQx, cs: &QcScratch, h: u64, n: usize, dst: u64) -> Result<()> {
        let (s, k, sh, si) = (e.dev.stream, &cs.kern, &self.shared, self.shared.inter);
        if n <= 8 {
            // (H a multiple of 512, I of 128: the kernels' loads)
            let (k1, k2) = (crate::model_qwen4x::rows_kernel(&k.sh1, n), crate::model_qwen4x::rows_kernel(&k.sh2, n));
            launch!(k1, [si.div_ceil(8) as u32 + 1, 1, 1], [256, 1, 1], s; h, sh.gu_q.ptr, sh.gu_s.ptr, sh.gate.ptr, cs.sact.ptr, cs.sg.ptr, n as i32, H as i32, si as i32)?;
            return launch!(k2, [(H as u32).div_ceil(32), 1, 1], [256, 1, 1], s; cs.sact.ptr, sh.dn_q.ptr, sh.dn_s.ptr, cs.sg.ptr, dst, n as i32, H as i32, si as i32);
        }
        launch!(k.i8bf16, [((2 * si * H / 4) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sh.gu_q.ptr, sh.gu_s.ptr, cs.sw.ptr, (2 * si) as i32, H as i32, (2 * si) as i32, H as i32)?;
        e.kern.gemm_fill(s, h, cs.sw.ptr, cs.sgu.ptr, n as i32, (2 * si) as i32, H as i32)?;
        launch!(k.swiglu, [((n * si) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; cs.sgu.ptr, cs.sact.ptr, n as i32, si as i32)?;
        launch!(k.i8bf16, [((H * si / 4) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sh.dn_q.ptr, sh.dn_s.ptr, cs.sw.ptr, H as i32, si as i32, H as i32, si as i32)?;
        e.kern.gemm_fill(s, cs.sact.ptr, cs.sw.ptr, dst, n as i32, H as i32, si as i32)?;
        launch!(k.small, [n as u32, 1, 1], [256, 1, 1], s; h, H as i32, sh.gate.ptr, cs.slog.ptr, 1i32, 1i32, H as i32)?;
        launch!(k.gate_rows, [((n * H) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; dst, cs.slog.ptr, n as i32, H as i32)
    }

    /// out += the shared expert's output (`shared_out`'s).
    ///
    /// # Safety
    /// As `MoeBlockQx::forward`.
    unsafe fn shared_add(&self, e: &EngineQx, cs: &QcScratch, out: u64, n: usize, src: u64) -> Result<()> {
        launch!(cs.kern.add, [((n * H) as u32).div_ceil(256), 1, 1], [256, 1, 1], e.dev.stream; out, src, (n * H) as i32)
    }
}

impl MoeBlockQx for QxMoeCb {
    unsafe fn forward(&self, e: &EngineQx, h: u64, n: usize, out: u64) -> Result<()> {
        self.pmap_stale.set(true);
        if n >= STREAM_MIN && self.common.stream.get() {
            return self.run_rows(e, h, n, out);
        }
        let pass = e.chunk().min(PAIR_TOKENS);
        let mut r = 0;
        while r < n {
            let m = pass.min(n - r);
            self.run_rows(e, h + (r * H * 2) as u64, m, out + (r * H * 2) as u64)?;
            r += m;
        }
        Ok(())
    }

    fn segmented(&self) -> bool {
        self.common.topk.get() == 0 && self.store.supports_parts(self.layer)
    }

    unsafe fn prepare(&self, e: &EngineQx) -> Result<()> {
        self.scratch(e)?;
        // (for the biggest call of any kind, so that no later call makes them again: the graphs' copies name their addresses)
        self.store.prepare_parts(self.layer, &e.dev, crate::offload::CPU_TOKENS, e.chunk() * self.cfg.router.top_k)
    }

    unsafe fn front(&self, e: &EngineQx, h: u64, n: usize) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let (c, s) = (&self.cfg, e.dev.stream);
        // (the logits, the choices and x' in one launch; the choices, their weights and x' written to the host's mapped buffers by the kernel itself, no copies: `ExpertStore::parts_host`)
        let [hids, hwts, hz] = self.store.parts_host()?;
        let mapped = self.mapped();
        let map = if mapped { self.pmap.ptr as u64 } else { 0 };
        cs.kern.router_rot(
            s,
            h,
            H,
            self.router.ptr,
            sc.logits.ptr,
            n,
            c,
            self.bias.as_ref().map_or(0, |b| b.ptr),
            sc.ids.ptr,
            sc.wts.ptr,
            self.signs.ptr,
            cs.xr.ptr,
            cs.xrb.ptr,
            self.local_of.ptr,
            cs.gids.ptr,
            self.experts,
            [hids, hwts, hz],
            map,
            sc.way.ptr,
        )?;
        crate::tmark!(s, "router");
        if mapped {
            // (the shared expert and the products of the pairs with a place, no wait for the host: it computes the others meanwhile, `host`)
            self.shared_out(e, cs, h, n, cs.sh.ptr)?;
            let (k, tc) = (c.router.top_k, self.tc(cs, n));
            cs.kern.products_pp(s, self.pp.ptr, self.strides, sc.way.ptr, cs.xr.ptr, cs.gu.ptr, cs.part.ptr, k, n * k, self.split(cs, n), tc, self.mixed)?;
        }
        Ok(())
    }

    unsafe fn host_then(&self, e: &EngineQx, n: usize, next: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        if !self.mapped() {
            self.host(e, n)?;
            return next();
        }
        let (sc, cs) = self.scratch(e)?;
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs: n * self.cfg.router.top_k };
        // (the front's graph has moved the host's word once more: waited for by looking at it)
        let want = cs.kern.seen.get().wrapping_add(1);
        cs.kern.seen.set(want);
        self.store.arm_ready(cs.kern.flag.ptr as *const u32, want);
        // SAFETY: the host's buffer of sums holds `front_rows` rows (the scratch's), and the next graph's back reads it only after the word below; the front that read the map has run when the store has seen its word,
        // and the next is queued by `next`, after the map's changes.
        let sums = std::slice::from_raw_parts_mut(cs.hsum.ptr as *mut f32, cs.hsum.bytes / 4);
        // (the kernel census times each launch alone, waiting for it: the next graph then after the sums)
        let ahead = !crate::dev::kprof_on();
        let r = if ahead {
            self.store.run_parts_mapped(self.layer, &e.dev, &call, self.pmap_mut(), next, sums)
        } else {
            self.store.run_parts_mapped(self.layer, &e.dev, &call, self.pmap_mut(), &mut || Ok(()), sums)
        };
        // the word the next graph's back waits for, whatever came of the call (the GPU is not left waiting): whether there are sums, then the front's count
        let w = cs.kern.hword.ptr as *mut u32;
        std::ptr::write_volatile(w.add(1), matches!(r, Ok((_, true))) as u32);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        std::ptr::write_volatile(w, want);
        if r.is_err() {
            // (the next graph may be queued, its front moving the word once more: the count the word's once the GPU is done)
            let _ = e.dev.sync();
            cs.kern.seen.set(std::ptr::read_volatile(cs.kern.flag.ptr as *const u32));
        }
        if r?.0 {
            self.pmap_behind.set(true);
        }
        if !ahead {
            next()?;
        }
        Ok(())
    }

    unsafe fn host(&self, e: &EngineQx, n: usize) -> Result<()> {
        if self.mapped() {
            return self.host_then(e, n, &mut || Ok(()));
        }
        let (sc, cs) = self.scratch(e)?;
        let (k, s) = (self.cfg.router.top_k, e.dev.stream);
        let pairs = n * k;
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs };
        let places = self.place_ptrs();
        let mut early = || {
            self.shared_out(e, cs, e.h.ptr, n, cs.sh.ptr)?;
            crate::tmark!(s, "shared expert");
            Ok(())
        };
        // (the front's graph has moved the host's word once more: waited for by looking at it)
        let want = cs.kern.seen.get().wrapping_add(1);
        cs.kern.seen.set(want);
        self.store.arm_ready(cs.kern.flag.ptr as *const u32, want);
        self.pmap_stale.set(true);
        let tc = self.tc(cs, n);
        self.store.run_parts(self.layer, &e.dev, &call, &mut early, &mut |hy| {
            if !hy.gpu {
                return Ok(());
            }
            cs.kern.up(s, places, self.strides, sc.way.ptr, cs.xr.ptr, cs.gu.ptr, k, pairs, self.mixed)?;
            if tc {
                return cs.kern.down_t(s, places, self.strides, sc.way.ptr, cs.gu.ptr, cs.part.ptr, pairs, self.mixed);
            }
            cs.kern.down(s, places, self.strides, sc.way.ptr, cs.gu.ptr, cs.part.ptr, runs_for(pairs), pairs, self.mixed)
        })
    }

    unsafe fn back(&self, e: &EngineQx, out: u64, n: usize) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let (k, s) = (self.cfg.router.top_k, e.dev.stream);
        cs.kern.sum_rot(s, cs.part.ptr, sc.wts.ptr, sc.cpu.ptr, sc.way.ptr, self.signs.ptr, out, k, self.split(cs, n), n)?;
        self.shared_add(e, cs, out, n, cs.sh.ptr)
    }

    unsafe fn back_post(&self, e: &EngineQx, n: usize, x: u64, inj: u64) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let k = self.cfg.router.top_k;
        // (a call whose graph is queued before the host's sums are made, `host_then`: they are read from the host's memory once its word says so)
        let (extra, hword, gen) = if self.mapped() { (cs.hsum.ptr as u64, cs.kern.hword.ptr as u64, cs.kern.counter.ptr + 4) } else { (sc.cpu.ptr, 0, 0) };
        launch!(cs.kern.back, [n as u32, (H / 512) as u32, 1], [512, 1, 1], e.dev.stream; cs.part.ptr, sc.wts.ptr, extra, sc.way.ptr, self.signs.ptr, cs.sh.ptr, inj, x, k as i32, self.split(cs, n) as i32, hword, gen)
    }

    fn whole(&self) -> bool {
        self.common.stream.get() && self.common.topk.get() == 0
    }

    fn whole_release(&self, e: &EngineQx) -> Result<()> {
        if let Some(cs) = self.common.scratch.get() {
            *cs.lm.borrow_mut() = None;
            if let Some(k) = cs.pre.get() {
                e.dev.sync()?;
                k.release_buffers();
            }
        }
        Ok(())
    }

    fn release_places(&self) -> Result<usize> {
        let held = self.places.borrow().iter().map(|b| b.bytes).sum::<usize>();
        if held <= 2 {
            return Ok(0);
        }
        // (the copies into them done and the cache emptied first; the store keeps no address of them: none until `restore_places`)
        self.store.reset_places(self.layer, [0, 0])?;
        *self.places.borrow_mut() = [Buf::new(1)?, Buf::new(1)?];
        self.places_moved()?;
        Ok(held)
    }

    fn restore_places(&self) -> Result<()> {
        if self.places.borrow().iter().map(|b| b.bytes).sum::<usize>() > 2 {
            return Ok(());
        }
        let made = [Buf::new(self.slots.get() * self.strides[0])?, Buf::new(self.slots.get() * self.strides[1])?];
        self.store.reset_places(self.layer, [made[0].ptr, made[1].ptr])?;
        *self.places.borrow_mut() = made;
        self.places_moved()
    }

    fn place_slots(&self) -> (usize, usize, usize) {
        (self.slots.get(), self.full, self.strides[0] + self.strides[1])
    }

    fn resize_places(&self, e: &EngineQx, slots: usize) -> Result<()> {
        let (slots, was) = (slots.clamp(1, self.full), self.slots.get());
        if slots == was {
            return Ok(());
        }
        self.slots.set(slots);
        // (places lent to a long prompt are made again in the new number by `restore_places`)
        if self.places.borrow().iter().map(|b| b.bytes).sum::<usize>() <= 2 {
            self.pmap_stale.set(true);
            return self.store.resize_places(self.layer, slots, [0, 0]);
        }
        // the copies into them done (every layer's), then each buffer made again with the first places' bytes moved over (the cache keeps those experts); where the GPU has not the room for both at once, the
        // old one freed first and the layer's cache emptied
        self.store.drain()?;
        let keep = slots.min(was);
        let mut kept = true;
        for k in 0..2 {
            let bytes = slots * self.strides[k];
            let made = if e.dev.mem_info()?.0 >= bytes + (16 << 20) {
                let b = Buf::new(bytes)?;
                // SAFETY: both buffers live, `keep` places of `strides[k]` bytes in each; synchronized before the old one is freed.
                unsafe { crate::dev::copy_d2d_async(b.ptr, self.places.borrow()[k].ptr, keep * self.strides[k], e.dev.stream)? };
                e.dev.sync()?;
                b
            } else {
                self.places.borrow_mut()[k] = Buf::new(1)?;
                kept = false;
                Buf::new(bytes)?
            };
            self.places.borrow_mut()[k] = made;
        }
        let p = self.place_ptrs();
        self.store.resize_places(self.layer, slots, p)?;
        if !kept {
            self.store.reset_places(self.layer, p)?;
        }
        self.places_moved()
    }

    unsafe fn whole_front(&self, e: &EngineQx, h: u64, n: usize, at: usize) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let lm = self.whole_set(e, cs)?;
        let mut c = self.cfg;
        if self.common.topk.get() != 0 {
            c.router.top_k = self.common.topk.get().min(self.cfg.router.top_k);
        }
        let (k, kk, s) = (c.router.top_k, self.cfg.router.top_k, e.dev.stream);
        if n == 0 || n > e.front_rows() || at + n > lm.rows || k != kk {
            return err(format!(
                "rows {at} to {} of a prompt taken layer by layer ({} rows, a part at most {}, {k} of {kk} pairs)",
                at + n,
                lm.rows,
                e.front_rows()
            ));
        }
        self.route(e, &c, sc, h, n)?;
        crate::tmark!(s, "router");
        crate::dev::copy_d2d_async(lm.wts + (at * k * 4) as u64, sc.wts.ptr, n * k * 4, s)?;
        cs.kern.rot_in(s, h, H, self.signs.ptr, cs.xr.ptr, lm.xrb + (at * H * 2) as u64, n)?;
        cs.kern.local(s, sc.ids.ptr, self.local_of.ptr, lm.gids + (at * k * 8) as u64, n * k, self.experts)?;
        self.shared_out(e, cs, h, n, lm.sh + (at * H * 2) as u64)?;
        crate::tmark!(s, "whole: front");
        Ok(())
    }

    unsafe fn whole_experts(&self, e: &EngineQx, rows: usize) -> Result<()> {
        let (_, cs) = self.scratch(e)?;
        let lm = self.whole_set(e, cs)?;
        if rows == 0 || rows > lm.rows {
            return err(format!("{rows} rows of a prompt taken layer by layer, where the set holds {}", lm.rows));
        }
        self.stream_experts(e, cs, rows, lm.xrb, lm.gids, lm.wts, lm.y)?;
        crate::tmark!(e.dev.stream, "whole: experts");
        // the experts of the prompt's last rows, by their pairs (the router's choices are done: the streaming call read them back), for the decode's places after the prompt
        let k = self.cfg.router.top_k;
        let last = rows.min(WARM_ROWS);
        let mut ids = vec![0i64; last * k];
        crate::dev::read_dev(lm.gids + ((rows - last) * k * 8) as u64, &mut ids)?;
        let mut count = vec![0u32; self.descs.len()];
        for &g in &ids {
            if g >= 0 && (g as usize) < count.len() {
                count[g as usize] += 1;
            }
        }
        let mut top: Vec<u16> = (0..count.len() as u16).filter(|&g| count[g as usize] > 0).collect();
        top.sort_by_key(|&g| std::cmp::Reverse(count[g as usize]));
        // (a prompt in parts: long where any part was)
        let long = rows >= LONG_ROWS || self.common.drain.get();
        if long {
            self.common.drain.set(true);
        }
        top.truncate(if long { self.slots.get() } else { WARM_TOP.min(self.slots.get()) });
        top.reverse();
        *self.warm.borrow_mut() = top;
        Ok(())
    }

    fn drain(&self) -> Result<()> {
        if self.common.drain.replace(false) {
            return self.store.drain();
        }
        Ok(())
    }

    fn rewarm(&self) -> Result<()> {
        let warm = std::mem::take(&mut *self.warm.borrow_mut());
        if warm.is_empty() {
            return Ok(());
        }
        self.pmap_stale.set(true);
        self.store.hint(self.layer, &warm)
    }

    fn sync_map(&self) -> Result<()> {
        let (stale, behind) = (self.pmap_stale.get(), self.pmap_behind.get());
        if !(stale || behind) || !self.mapped() {
            return Ok(());
        }
        // SAFETY: a step's start: every front queued before has run (its host part saw its word).
        if self.store.map_sync(self.layer, unsafe { self.pmap_mut() }, stale)? {
            self.pmap_stale.set(false);
            self.pmap_behind.set(false);
        }
        Ok(())
    }

    unsafe fn whole_back(&self, e: &EngineQx, out: u64, n: usize, at: usize) -> Result<()> {
        let (_, cs) = self.scratch(e)?;
        let lm = self.whole_set(e, cs)?;
        let s = e.dev.stream;
        cs.kern.sum_rot(s, 0, 0, lm.y + (at * H * 4) as u64, 0, self.signs.ptr, out, 0, 1, n)?;
        self.shared_add(e, cs, out, n, lm.sh + (at * H * 2) as u64)?;
        crate::tmark!(s, "whole: back");
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.router.bytes
            + self.bias.as_ref().map_or(0, |b| b.bytes)
            + self.shared.bytes()
            + self.signs.bytes
            + self.local_of.bytes
            + self.places.borrow().iter().map(|b| b.bytes).sum::<usize>()
    }
}

/// The factory `EngineQx::new` takes for experts in `pack`: each layer's block a `QxMoeCb` with `slots` places, all sharing `store` and `common`; the shared expert at the engine's layout.
pub fn factory<'a>(
    pack: &'a CbPack,
    slots: usize,
    store: &'a Rc<ExpertStore>,
    common: &'a Rc<QxCbShared>,
) -> impl Fn(usize, &ConfigQx, &Checkpoint, Layout) -> Result<Box<dyn MoeBlockQx>> + 'a {
    move |l, c, ck, layout| {
        crate::status::STATUS.beat();
        Ok(Box::new(QxMoeCb::load(l, c, ck, layout, pack, store, slots, common)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every call's runs divide the neurons into runs of a multiple of 4 (the kernel's groups), and the partial sums fit `PART_ROWS` (or the pairs alone).
    #[test]
    fn the_runs_fit() {
        for pairs in 1..5000 {
            let s = runs_for(pairs);
            assert!(I.is_multiple_of(s) && (I / s).is_multiple_of(4) && (pairs * s <= PART_ROWS || s == 1), "{pairs}: {s}");
        }
        assert_eq!(runs_for(10), 16);
    }
}
