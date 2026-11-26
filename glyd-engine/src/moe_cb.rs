//! The Moe kind's block at the hummingbird level of a codes pack (glyd-hbr-codes/2, src/cb_pack.rs: each routed expert rows of codebook codes in the basis of its
//! layer's rotation, of two, three or four bits, ternary, or format q's trellis rows of two, three or four bits, an expert's own format), made from the layer's kind
//! and its roles' tensors (src/dynamic/experts.rs `block`, `Held::codes`): the router (the kind's softmax over the layer's experts, the top k renormalised; an expert
//! the pack prunes never chosen), the rotation of the layer's input, the experts' two products over the expert cache's places (kernels/qwen4x.cu `qx_cb_*`: the
//! kernels of every format), the weighted sum and the rotation back, the shared expert times its gate. Every layer's file is mapped as the cache's host tier
//! (src/offload.rs): the GPU holds `places` experts of a layer, copied in as the router asks for them (as many places as the layer's experts: every expert on the
//! GPU), and the host computes a miss from its memory (src/cpu_cb.rs) or copies it, as the store plans. The shared expert is held as int8 rows with a scale each, its
//! kernels' form (`SharedRows`).

use crate::cb_pack::{CbPack, LayerMap};
use crate::dev::{Buf, Dev};
use crate::dynamic::experts::Ix;
use crate::ffi::{err, Error, Result, Stream};
use crate::kern_nh::{MoeBlock, NhHost};
use crate::launch;
use crate::moe::{route_with, Act, MoeCfg, MoeScratch, Scoring};
use crate::offload::{Call, ExpertStore, HostMem, Stage};
use crate::runtime::describe::Source;
use crate::runtime::kinds;
use crate::weights::upload_bytes;
use glyd_gpu::safetensors::Checkpoint;
use std::cell::{Cell, OnceCell};
use std::rc::Rc;
use std::sync::Arc;

/// The width and the neurons the kernels are made for: their lanes take 80 codes of a row each (32 lanes) in the first product and 40 outputs a thread
/// in the second, and the rotation is five transforms of 512, so no other width or count of neurons runs (`takes`).
const H: usize = 2560;
const I: usize = 640;

/// Whether the kernels take experts of `inter` neurons over a width of `hidden`, or the refusal that names both and what the kernels take.
pub fn takes(hidden: usize, inter: usize) -> Result<()> {
    if (hidden, inter) == (H, I) {
        return Ok(());
    }
    err(format!(
        "experts of {inter} neurons over a width of {hidden}: the codebook experts' kernels take exactly {I} neurons over a width of {H} (each lane of the first product \
         takes 80 codes of a row, each thread of the second 40 outputs, the rotation five transforms of 512)"
    ))
}
/// The most pair-runs (pairs times runs of neurons) the partial sums of a call hold, and the most tokens a pass of the per-pair products takes (the host's pairs' limit: a longer call streams its experts).
const PART_ROWS: usize = 4096;
const PAIR_TOKENS: usize = 1024;
/// The SwiGLU the block takes has no clamp (`cfg_of`); the host's kernels take a finite one.
const NO_CLAMP: f32 = f32::MAX;
/// Rows a streamed call has at most for the host to take a share of its experts (`stream_experts`).
const HOST_ROWS: usize = 4096;
/// Tokens a call has at least for the streaming prefill (below, the cache's per-pair products with the host's misses).
pub(crate) const STREAM_MIN: usize = crate::offload::CPU_TOKENS + 1;

/// The kernels of kernels/qwen4x.cu's codebook rows with typed launches.
pub struct KernCb {
    rot_in: crate::kt::qx_cb_rot_in,
    local: crate::kt::glm_cb_local,
    up: crate::kt::qx_cb_up,
    down: crate::kt::qx_cb_down,
    /// the second product of a call of a few rows on the tensor cores, a place's pairs together, for calls of `tc` to 8 rows (0: none; GLYD_QX_TC=N in a development build)
    downt: crate::kt::qx_cb_downt,
    pub tc: usize,
    sum_rot: crate::kt::qx_cb_sum_rot,
    sh1: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_sh1_1>>; 4],
    sh2: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_sh2_1>>; 4],
    add: crate::kt::qx_add_bf16,
    swiglu: crate::kt::qx_swiglu,
    gate_rows: crate::kt::qx_gate_rows,
    i8bf16: crate::kt::qx_i8_bf16,
    small: crate::kt::small_linear,
}

impl KernCb {
    pub fn load(dev: &Dev) -> Result<KernCb> {
        Ok(KernCb {
            rot_in: dev.typed()?,
            local: dev.typed()?,
            up: dev.typed()?,
            down: dev.typed()?,
            downt: dev.typed()?,
            tc: crate::tune!("GLYD_QX_TC").and_then(|v| v.parse().ok()).unwrap_or(2),
            sum_rot: dev.typed()?,
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
pub struct CbShared {
    scratch: OnceCell<CbScratch>,
    pub topk: Cell<usize>,
    pub read_ahead: Cell<u64>,
    /// Whether a prompt chunk of more than `CPU_TOKENS` tokens goes through the streaming prefill (src/cb2_prefill.rs: the experts with pairs streamed through a ring on the GPU, each multiplied by all its tokens on the
    /// tensor cores) instead of the cache's per-pair products.
    pub stream: Cell<bool>,
    /// The layers' mapped files with the ranges of each (offset, bytes) that the read-ahead budget took (the page tables of those are filled in the background after the load: `fill_page_tables`)
    pub maps: std::cell::RefCell<Vec<(Arc<crate::mapped::Mapping>, usize, usize)>>,
    /// Whether the layers' experts are copied into pinned host memory at load (`CbPack::pin_layer`) instead of read from the mapped files.
    pub pin_pack: Cell<bool>,
}

impl CbShared {
    pub fn new() -> Rc<CbShared> {
        Rc::new(CbShared {
            scratch: OnceCell::new(),
            topk: Cell::new(0),
            read_ahead: Cell::new(u64::MAX),
            stream: Cell::new(false),
            maps: std::cell::RefCell::new(Vec::new()),
            pin_pack: Cell::new(false),
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

struct CbScratch {
    kern: KernCb,
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
    /// the streaming prefill's kernels
    pre: OnceCell<crate::cb2_prefill::Cb2Kern>,
}

/// The shared expert as int8 rows with an fp32 scale each (`max |w| / 127`): gate and up [2 I, H], down [H, I]; its gate [H] bf16.
pub struct SharedRows {
    gu_q: Buf,
    gu_s: Buf,
    dn_q: Buf,
    dn_s: Buf,
    gate: Buf,
    inter: usize,
}

impl SharedRows {
    /// The shared expert of `gate`, `up` and `down` (bf16 [inter, h], [inter, h], [h, inter]: their names) made int8 rows, its gate `router` ([1, h] bf16) as
    /// released.
    fn load(ck: &Checkpoint, [gate, up, down, router]: [&str; 4], h: usize) -> Result<SharedRows> {
        let (_, t) = ck.get(gate).ok_or_else(|| Error(format!("no tensor {gate}")))?;
        let inter = match t.shape[..] {
            [r, c] if c as usize == h => r as usize,
            _ => return err(format!("{gate}: {:?}, where a shared expert's gate of {h} columns is read", t.shape)),
        };
        let read = |name: &str, n: usize| -> Result<Vec<u16>> {
            let mut v = vec![0u16; n];
            ck.read_bf16(name, &mut v).map_err(|e| Error(format!("{name}: {e}")))?;
            Ok(v)
        };
        let mut gu = read(gate, inter * h)?;
        gu.extend(read(up, inter * h)?);
        let dn = read(down, h * inter)?;
        let (gq, gs) = quant_rows(&gu, 2 * inter, h, 2 * inter, h);
        let (dq, ds) = quant_rows(&dn, h, inter, h, inter);
        let bytes = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
        Ok(SharedRows {
            gu_q: upload_bytes(&bytes(&gq))?,
            gu_s: upload_bytes(crate::weights::as_bytes(&gs))?,
            dn_q: upload_bytes(&bytes(&dq))?,
            dn_s: upload_bytes(crate::weights::as_bytes(&ds))?,
            gate: upload_bytes(&ck.read(router).map_err(|e| Error(format!("{router}: {e}")))?)?,
            inter,
        })
    }

    fn bytes(&self) -> usize {
        self.gu_q.bytes + self.gu_s.bytes + self.dn_q.bytes + self.dn_s.bytes + self.gate.bytes
    }
}

/// A mixture-of-experts layer with its routed experts in the expert cache.
pub struct CbMoe {
    cfg: MoeCfg,
    router: Buf,
    /// -1e30 for an expert the pack left out (none where it keeps them all)
    bias: Option<Buf>,
    shared: SharedRows,
    signs: Buf,
    local_of: Buf,
    experts: usize,
    /// the places on the GPU (codes, codebooks: the store's `slots` each)
    places: std::cell::RefCell<[Buf; 2]>,
    #[allow(dead_code)]
    map: Arc<crate::mapped::Mapping>,
    /// the kept experts as the streaming prefill takes them (addresses in the mapping)
    descs: Vec<crate::cb2_prefill::ExpertDesc>,
    strides: [usize; 2],
    mixed: bool,
    /// whether every kept expert is in a format the tensor-core second products take (two-bit and format q)
    tc_ok: bool,
    store: Rc<ExpertStore>,
    layer: usize,
    common: Rc<CbShared>,
}

/// How a stage holds its blocks' routed experts at the hummingbird level of a codes pack (src/dynamic/experts.rs `Held::codes`): the pack (its sidecars laid
/// over it), the places a layer on the GPU (as many as a layer keeps: every expert there), the expert cache's store they are in and the blocks' shared settings.
#[derive(Clone, Copy)]
pub struct Codes<'a> {
    pub pack: &'a CbPack,
    pub places: usize,
    pub store: &'a Rc<ExpertStore>,
    pub common: &'a Rc<CbShared>,
}

/// What the codes block takes of the kind `m` over a model `hidden` wide, experts of `inter` neurons: the core's shape (src/dynamic/experts.rs `cfg`), or the
/// refusal in words of what its kernels do not run: another width or count of neurons (`takes`), experts in a latent, a router other than a softmax one or one
/// with a choice bias (the mask of the experts a pack prunes is the bias the block's router takes), an activation other than SwiGLU without a clamp, no shared
/// expert or one without its gate.
pub fn cfg_of(m: &kinds::Moe, hidden: usize, inter: usize) -> Result<MoeCfg> {
    takes(hidden, inter)?;
    let c = crate::dynamic::experts::cfg(m, hidden, inter)?;
    if m.latent.is_some() || c.router.scoring != Scoring::Softmax || m.router.bias {
        return err(format!("experts of {:?} routed by {:?}: the codes block takes a softmax router with no choice bias, no latent", m.expert, m.router));
    }
    if c.act != (Act::SwiGlu { limit: None }) || m.shared.is_none() || !m.shared_gate {
        return err(format!(
            "experts of {:?} beside a shared expert of {:?} (gated: {}): the codes block takes SwiGLU without a clamp and a gated shared expert",
            m.expert, m.shared, m.shared_gate
        ));
    }
    Ok(c)
}

impl CbMoe {
    /// The block of the kind `m` whose tensors `roles` names over a model `hidden` wide (`cfg_of`), at `ix` (`ix.layer` the pack's layer, `ix.moe` the expert cache's):
    /// the router and the shared expert's gate as released (bf16), the shared expert as int8 rows (`SharedRows`), the layer's rotation, and the experts of
    /// `codes.pack`, its layer's file mapped as the cache's host tier, with `codes.places` places on the GPU.
    pub fn load(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, ix: Ix, codes: Codes) -> Result<CbMoe> {
        let (pack, store, common, l) = (codes.pack, codes.store, codes.common, ix.layer);
        let cfg = cfg_of(m, hidden, pack.inter).map_err(|e| e.context(&format!("layer {l}'s experts")))?;
        takes(pack.hidden, pack.inter).map_err(|e| e.context("the pack's"))?;
        let one = |r: &str| match roles.iter().find(|(n, _)| n == r).map(|(_, s)| s) {
            Some(Source::Tensor(n)) => Ok(n.as_str()),
            other => err(format!("layer {l}'s experts' role {r} is {other:?}, where one tensor is")),
        };
        let kept = pack.kept(l).ok_or_else(|| Error(format!("{}: no experts of layer {l}", pack.dir.display())))?.to_vec();
        if let Some(e) = kept.iter().find(|&&e| e >= m.experts) {
            return err(format!("{}: layer {l} keeps expert {e}, where the layer has {}", pack.dir.display(), m.experts));
        }
        let router = one("router")?;
        let router = upload_bytes(&ck.read(router).map_err(|e| Error(format!("{router}: {e}")))?)?;
        let mut local = vec![-1i32; m.experts];
        for (i, &e) in kept.iter().enumerate() {
            local[e] = i as i32;
        }
        let bias = if kept.len() < m.experts {
            Some(upload_bytes(&local.iter().flat_map(|&x| if x < 0 { -1e30f32 } else { 0.0 }.to_le_bytes()).collect::<Vec<u8>>())?)
        } else {
            None
        };
        let shared = SharedRows::load(ck, [one("shared_gate")?, one("shared_up")?, one("shared_down")?, one("shared_router")?], hidden)?;
        let slots = codes.places.clamp(1, kept.len());
        let lm = pack.map_layer(l)?;
        // (pinned: the experts copied into pinned host memory now, the files' pages given back)
        let lm = if common.pin_pack.get() { pack.pin_layer(l, lm)? } else { lm };
        let (mixed, strides) = (lm.place.is_some(), lm.strides);
        let places = [Buf::new(slots * strides[0])?, Buf::new(slots * strides[1])?];
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
            ix.moe,
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
        Ok(CbMoe {
            cfg,
            router,
            bias,
            shared,
            signs: upload_bytes(&signs.iter().map(|&s| s as u8).collect::<Vec<u8>>())?,
            local_of: upload_bytes(&local.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?,
            experts: m.experts,
            places: std::cell::RefCell::new(places),
            map,
            descs,
            strides,
            mixed,
            tc_ok,
            store: store.clone(),
            layer: ix.moe,
            common: common.clone(),
        })
    }

    /// The MoE scratch every layer shares, and this module's beside it, made by the first call.
    fn scratch<'a>(&'a self, e: &'a dyn CbHost) -> Result<(&'a MoeScratch, &'a CbScratch)> {
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::routing(e.dev(), self.cfg.experts, self.cfg.router.top_k, H, e.front_rows())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        if self.common.scratch.get().is_none() {
            // (the front's buffers for `front_rows`, the per-pair products' for the engine's chunk)
            let (n, k) = (e.front_rows(), self.cfg.router.top_k);
            let pass = e.chunk().min(PAIR_TOKENS) * k;
            let si = self.shared.inter;
            let made = CbScratch {
                kern: KernCb::load(e.dev())?,
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
                pre: OnceCell::new(),
            };
            let _ = self.common.scratch.set(made);
        }
        Ok((sc, self.common.scratch.get().ok_or_else(|| Error("the scratch is not made".into()))?))
    }

    /// `n` rows (at most `PAIR_TOKENS`): the router, x' = R x, the pairs' experts made resident or computed by the host (`ExpertStore::run_early`, the shared expert queued while the host plans), the products of the pairs
    /// with places, the weighted sum with the host's sums and R^T into `out`, the shared expert added.
    ///
    /// # Safety
    /// As `MoeBlock::forward`.
    unsafe fn run_rows(&self, e: &dyn CbHost, h: u64, n: usize, out: u64) -> Result<()> {
        let s = e.dev().stream;
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
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, rows: 0, tokens: n, pairs };
        let places = self.place_ptrs();
        let mut early = || {
            self.shared_out(e, cs, h, n, cs.sh.ptr)?;
            crate::tmark!(s, "shared expert");
            Ok(())
        };
        self.store.run_early(self.layer, e.dev(), &call, &mut early, &mut |stage, wave, waves, hy| match stage {
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
    /// As `MoeBlock::forward`; the buffers hold n rows.
    #[allow(clippy::too_many_arguments)]
    unsafe fn stream_experts(&self, e: &dyn CbHost, cs: &CbScratch, n: usize, xrb: u64, gids: u64, wts: u64, y: u64) -> Result<()> {
        let s = e.dev().stream;
        if cs.pre.get().is_none() {
            let _ = cs.pre.set(crate::cb2_prefill::Cb2Kern::load_dims(e.dev(), H, I, NO_CLAMP)?);
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
        crate::cb2_prefill::cb2_prefill_experts(e.dev(), kern, &descs, n, self.cfg.router.top_k, xrb, gids, wts, y, s)?;
        if crate::tune!("GLYD_CB2_HOSTLOG").is_some() && kern.host.get().is_some() {
            let t = kern.took.get();
            eprintln!(
                "host share: layer {} rows {n}: {} experts, {} pairs, host {:.1} ms, GPU loop {:.1} ms",
                self.layer, t.experts, t.pairs, t.host_ms, t.gpu_ms
            );
        }
        Ok(())
    }

    /// Whether a call of n rows takes the tensor-core second products (`KernCb::down_t`): `tc` to 8 rows of a layer whose formats they take.
    fn tc(&self, cs: &CbScratch, n: usize) -> bool {
        self.tc_ok && cs.kern.tc > 0 && (cs.kern.tc..=8).contains(&n)
    }

    /// The addresses of the places (codes, codebooks).
    fn place_ptrs(&self) -> [u64; 2] {
        let p = self.places.borrow();
        [p[0].ptr, p[1].ptr]
    }
}

impl CbMoe {
    /// The shared expert on the n rows at `h` into the scratch's `sh` (gated): two fused launches for a few rows, else the int8 rows made bf16 and the engine's dense products.
    ///
    /// # Safety
    /// As `MoeBlock::forward`.
    unsafe fn shared_out(&self, e: &dyn CbHost, cs: &CbScratch, h: u64, n: usize, dst: u64) -> Result<()> {
        let (s, k, sh, si) = (e.dev().stream, &cs.kern, &self.shared, self.shared.inter);
        if n <= 8 {
            // (H a multiple of 512, I of 128: the kernels' loads)
            let (k1, k2) = (rows_kernel(&k.sh1, n), rows_kernel(&k.sh2, n));
            launch!(k1, [si.div_ceil(8) as u32 + 1, 1, 1], [256, 1, 1], s; h, sh.gu_q.ptr, sh.gu_s.ptr, sh.gate.ptr, cs.sact.ptr, cs.sg.ptr, n as i32, H as i32, si as i32)?;
            return launch!(k2, [(H as u32).div_ceil(32), 1, 1], [256, 1, 1], s; cs.sact.ptr, sh.dn_q.ptr, sh.dn_s.ptr, cs.sg.ptr, dst, n as i32, H as i32, si as i32);
        }
        launch!(k.i8bf16, [((2 * si * H / 4) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sh.gu_q.ptr, sh.gu_s.ptr, cs.sw.ptr, (2 * si) as i32, H as i32, (2 * si) as i32, H as i32)?;
        e.gemm(h, cs.sw.ptr, cs.sgu.ptr, n as i32, (2 * si) as i32, H as i32)?;
        launch!(k.swiglu, [((n * si) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; cs.sgu.ptr, cs.sact.ptr, n as i32, si as i32)?;
        launch!(k.i8bf16, [((H * si / 4) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sh.dn_q.ptr, sh.dn_s.ptr, cs.sw.ptr, H as i32, si as i32, H as i32, si as i32)?;
        e.gemm(cs.sact.ptr, cs.sw.ptr, dst, n as i32, H as i32, si as i32)?;
        launch!(k.small, [n as u32, 1, 1], [256, 1, 1], s; h, H as i32, sh.gate.ptr, cs.slog.ptr, 1i32, 1i32, H as i32)?;
        launch!(k.gate_rows, [((n * H) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; dst, cs.slog.ptr, n as i32, H as i32)
    }

    /// out += the shared expert's output (`shared_out`'s).
    ///
    /// # Safety
    /// As `MoeBlock::forward`.
    unsafe fn shared_add(&self, e: &dyn CbHost, cs: &CbScratch, out: u64, n: usize, src: u64) -> Result<()> {
        launch!(cs.kern.add, [((n * H) as u32).div_ceil(256), 1, 1], [256, 1, 1], e.dev().stream; out, src, (n * H) as i32)
    }
}

impl CbMoe {
    /// The block on n rows at `h` into `out`, whichever engine runs it: a long call streamed, else passes of at most `PAIR_TOKENS` rows over the cache's places.
    ///
    /// # Safety
    /// As `MoeBlock::forward`.
    unsafe fn rows(&self, e: &dyn CbHost, h: u64, n: usize, out: u64) -> Result<()> {
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
}

/// What a block's own calls read of the engine that runs them: the executor's (src/dynamic, through `NhHost`).
pub trait CbHost {
    fn dev(&self) -> &Dev;
    fn moe_scratch(&self) -> &OnceCell<MoeScratch>;
    fn chunk(&self) -> usize;
    /// The rows the front's scratch takes at once (the executor's chunk).
    fn front_rows(&self) -> usize;
    /// y [m, n] = x [m, k] w^T in bf16, w a block's own rows on the GPU.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    unsafe fn gemm(&self, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()>;
}

/// The executor as the block's host.
struct OnNh<'a>(&'a dyn NhHost);

impl CbHost for OnNh<'_> {
    fn dev(&self) -> &Dev {
        self.0.dev()
    }
    fn moe_scratch(&self) -> &OnceCell<MoeScratch> {
        self.0.moe_scratch()
    }
    fn chunk(&self) -> usize {
        self.0.chunk()
    }
    fn front_rows(&self) -> usize {
        self.0.chunk()
    }
    unsafe fn gemm(&self, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        self.0.gemm(x, w, y, m, n, k)
    }
}

/// The block on the executor (`Op::Moe` of a block whose experts are not all on the GPU runs the block's own launches over the executor's buffers): the router,
/// the routed experts from the shared expert store (src/offload.rs: places on the GPU, the host's memory behind them), the shared expert. A call reads the
/// router's choices on the host, so no graph captures it; the executor ends each step's statistics in its store.
impl MoeBlock for CbMoe {
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        self.rows(&OnNh(e), h, n, out)
    }
    /// (its router leaves a call's choices in the experts' scratch, `sc.ids`, as a resident block's does)
    fn top_k(&self) -> Option<usize> {
        Some(self.cfg.router.top_k)
    }
    /// (its router, the shared expert, the rotation's signs, the local numbering and the places)
    fn bytes(&self) -> usize {
        self.router.bytes
            + self.bias.as_ref().map_or(0, |b| b.bytes)
            + self.shared.bytes()
            + self.signs.bytes
            + self.local_of.bytes
            + self.places.borrow().iter().map(|b| b.bytes).sum::<usize>()
    }
    fn reads_host(&self) -> bool {
        true
    }
    fn store(&self) -> Option<Rc<ExpertStore>> {
        Some(self.store.clone())
    }
}

/// A matrix's rows as int8 with an fp32 scale each (a row of zeros: scale 1), `rows_pad` rows and `cols_pad` columns (the padding zeros).
pub(crate) fn quant_rows(w: &[u16], rows: usize, cols: usize, rows_pad: usize, cols_pad: usize) -> (Vec<i8>, Vec<f32>) {
    let (mut q, mut sc) = (vec![0i8; rows_pad * cols_pad], vec![1f32; rows_pad]);
    for r in 0..rows {
        let row: Vec<f32> = w[r * cols..(r + 1) * cols].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let m = row.iter().fold(0f32, |a, v| a.max(v.abs()));
        if m > 0.0 {
            sc[r] = m / 127.0;
            for (c, v) in row.iter().enumerate() {
                q[r * cols_pad + c] = (v / sc[r]).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, sc)
}

/// The kernel of a few-rows family (1, 2, 4 or 8 rows) for n rows (1 to 8).
pub(crate) fn rows_kernel<A>(k: &[crate::dev::ByArgs<A>; 4], n: usize) -> &crate::dev::ByArgs<A> {
    &k[match n {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        _ => 3,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codes block takes the kind of the newest Qwen family's experts (a softmax over its experts, the top k renormalised, SwiGLU without a clamp, a gated
    /// shared expert) at its kernels' widths; what its kernels do not run is refused in words.
    #[test]
    fn the_kind_the_codes_block_takes() {
        use crate::runtime::kinds::{Latent, Mlp, Moe, Router, RouterScore};
        let mlp = Mlp { act: kinds::Act::Silu, gated: true, bias: false, clamp: None };
        let router = Router { score: RouterScore::Softmax, logit_scale: 1.0, bias: false, groups: None, top_k: 10, renorm: true, scale: 1.0, hash: false };
        let m = Moe { router, experts: 512, expert: mlp, shared: Some(mlp), shared_gate: true, latent: None };
        let c = cfg_of(&m, 2560, 640).unwrap();
        assert_eq!((c.experts, c.hidden, c.inter, c.router.top_k, c.act), (512, 2560, 640, 10, Act::SwiGlu { limit: None }));
        let refused = |edit: &dyn Fn(&mut Moe), (h, i): (usize, usize), word: &str| {
            let mut x = m.clone();
            edit(&mut x);
            let e = cfg_of(&x, h, i).unwrap_err().msg;
            assert!(e.contains(word), "{e}");
        };
        refused(&|_| {}, (4096, 2048), "exactly 640 neurons over a width of 2560");
        refused(&|x| x.router.score = RouterScore::Sigmoid, (2560, 640), "softmax router");
        refused(&|x| x.router.bias = true, (2560, 640), "choice bias");
        refused(
            &|x| {
                x.expert.clamp = Some(7.0);
                x.shared = Some(x.expert);
            },
            (2560, 640),
            "SwiGLU without a clamp",
        );
        refused(&|x| x.shared_gate = false, (2560, 640), "gated shared expert");
        refused(&|x| x.latent = Some(Latent { dim: 512, norm_in: None, expert_norm: None, norm_out: None }), (2560, 640), "no latent");
    }

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
