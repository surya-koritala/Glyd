//! Kimi-K3 (src/kimi.rs; the math: research/kimi-k3.md) as a description. No residual stream: the partial sum of the current block of layers is X, the completed
//! blocks (the embedding first) are Blk's, and every block's input is their depth mix and norm (`Op::DepthMix`, the layer before's add in it). The delta
//! attention is GLM's operations (q, k, v, the full-rank output gate, the forget gate's first product and beta one product of the layer's input, read where it
//! leaves them: `View`); the latent attention the layer kind's (`Op::Mla`, src/dynamic/mla.rs: q_a, the gate and kv_a one product; the absorbed query with the key part
//! the heads share, a dense attention, the sigmoid output gate); the experts a block on the mixture-of-experts core (MXFP4 as released, packed exactly as DeepSeek-V4's
//! FP4, situ, on the latent, its sum normed before the up projection), the shared experts the core's shared MLP; layer 0's dense MLP with situ. Norms are
//! KimiRMSNorm's (two roundings). The CPU reference (src/kimi_ref.rs) checks it (`kimicheck`).

use super::{DepthMixOp, Dyn, Loaded, MlaIn, MlaOp, ModelDesc, MoeOp, Op, Opts, Slot, Stage, View};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result, Stream};
use crate::glm::GlmKind;
use crate::kern35::Mat;
use crate::kern_nh::{MoeBlock, NhHost};
use crate::kimi::ConfigKimi;
use crate::launch;
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeScratch, MoeWeights, Router, Shared};
use crate::moe_rows::Down;
use crate::offload::{ExpertStore, PinnedPack};
use crate::weights::{as_bytes, pack_words, upload_bytes, upload_packed, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;
use std::rc::Rc;

// The kernels' names: those the .cu sources give them, in every build (kname!).
/// kernels/kimi.cu's kernels, KimiRMSNorm (kernels/mla.cu's `rms2_rows512`) and engine.cu's copy of a fused product's part.
pub struct KernKimi {
    rms2: crate::kt::rms2_rows512,
    rows: crate::kt::parts_rows,
    /// the expert cache's plan and copies on the GPU (src/dynamic/tp_cache.rs's device path)
    pub(crate) dc_plan: crate::kt::dc_plan,
    pub(crate) dc_fill: crate::kt::dc_fill,
}

impl KernKimi {
    pub fn load(dev: &Dev) -> Result<KernKimi> {
        Ok(KernKimi { rms2: dev.typed()?, rows: dev.typed()?, dc_plan: dev.typed()?, dc_fill: dev.typed()? })
    }

    /// y = KimiRMSNorm(x) (rows xs and ys apart, k values).
    ///
    /// # Safety
    /// Live device buffers of those sizes, on stream `s`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rms2(&self, s: Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, k: usize, eps: f32, n: u32) -> Result<()> {
        launch!(self.rms2, [n, 1, 1], [512, 1, 1], s; x, xs as i32, w, y, ys as i32, k as i32, eps)
    }

    /// dst [n, w] (rows ds apart) = n rows of w values of src (rows ss apart).
    ///
    /// # Safety
    /// As `rms2`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows(&self, s: Stream, src: u64, ss: usize, dst: u64, ds: usize, w: usize, n: usize) -> Result<()> {
        launch!(self.rows, [((n * w) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; src, ss as i64, dst, ds as i64, w as i32, n as i32)
    }
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// A tensor's bf16 words (an fp32 one rounded: none of the matrices is).
pub(crate) fn words(ck: &Checkpoint, name: &str) -> Result<(Vec<u16>, Vec<u64>)> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let (dtype, shape) = (t.dtype.clone(), t.shape.clone());
    let b = io(ck.read(name))?;
    let w = match dtype.as_str() {
        "BF16" => b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        d => return err(format!("{name}: {d}, where a bf16 matrix is read")),
    };
    Ok((w, shape))
}

/// An fp32 vector (A_log, dt_bias, the router's bias), from fp32 or widened exactly from bf16.
fn f32s(ck: &Checkpoint, name: &str) -> Result<Buf> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let b = io(ck.read(name))?;
    match t.dtype.as_str() {
        "F32" => upload_bytes(&b),
        "BF16" => upload_bytes(&b.chunks(2).flat_map(|c| ((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect::<Vec<u8>>()),
        d => err(format!("{name}: {d}, where fp32 or bf16 is read")),
    }
}

/// A mixture-of-experts layer: the core's layer (the router over the model's width, the experts on the latent, the shared MLP), the latent's two projections and norm.
struct KimiMoe {
    layer: MoeLayer,
    /// the latent's down projection and the shared experts' gate and up: one product of the layer's input (a row's latent values, then the gate's and
    /// the up's; zeros after, to a multiple of a GPU's slice); the shared expert's own first matrix a shape alone
    dg: Mat,
    up: Mat,
    norm: Buf,
    latent: usize,
    eps: f32,
    k: Rc<KernKimi>,
}

impl KimiMoe {
    /// The fused product of h into Mix, and where the experts read z, where their sum goes and where the shared expert reads its gate and up: one token's
    /// in place (z at Mix's start, the sum into Xc); a prompt's copied out first (z into Xc, the gate and up into the scratch), the sum into Mix.
    unsafe fn down(&self, e: &dyn NhHost, sc: &MoeScratch, h: u64, n: usize) -> Result<(u64, u64, u64)> {
        let (mix, xc, s) = (e.mix().ptr, e.xc().ptr, e.dev().stream);
        let (row, gw) = (self.dg.whole_rows(), self.layer.shared.as_ref().map_or(0, |x| 2 * x.inter));
        if e.mix().bytes < n * row * 2 || e.xc().bytes < n * self.latent * 2 {
            return err(format!("{n} rows of {row} values do not fit the engine's buffers"));
        }
        e.lin(&self.dg, h, n, mix)?;
        let gu = mix + (self.latent * 2) as u64;
        if n == 1 {
            return Ok((mix, xc, gu));
        }
        let k = &self.k;
        k.rows(s, mix, row, xc, self.latent, self.latent, n)?;
        k.rows(s, gu, row, sc.sg.ptr, gw, gw, n)?;
        Ok((xc, mix, sc.sg.ptr))
    }
}

impl MoeBlock for KimiMoe {
    /// The router on h; z, and the shared experts' gate and up, of h (`down`); the experts' weighted sum of z; its norm; up of that into out; the shared
    /// MLP's rest added.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let (c, s) = (&self.layer.cfg, e.dev().stream);
        if e.mix().bytes < n * c.router_hidden * 2 || e.xc().bytes < n * c.router_hidden * 2 {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.router_hidden));
        }
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[&self.layer], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        self.layer.route(sc, s, h, c.router_hidden, n)?;
        e.mark("MoE: router");
        let (z, y, gu) = self.down(e, sc, h, n)?;
        e.mark("MoE: latent down and shared gate, up product");
        self.layer.experts(e.lib(), sc, s, z, n, y)?;
        e.mark("MoE: experts");
        self.k.rms2(s, y, self.latent, self.norm.ptr, z, self.latent, self.latent, self.eps, n as u32)?;
        e.mark("MoE: latent norm");
        e.lin(&self.up, z, n, out)?;
        e.mark("MoE: latent up product");
        self.layer.shared_from(e.lib(), sc, s, gu, n)?;
        self.layer.shared_add(sc, s, n, out)
    }

    fn graphable(&self) -> bool {
        !self.layer.reads_host()
    }

    fn reads_host(&self) -> bool {
        self.layer.reads_host()
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes() + self.dg.bytes() + self.up.bytes() + self.norm.bytes
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        [&self.dg, &self.up].iter().filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None }).collect()
    }
}

/// A mixture-of-experts layer whose experts are not all on the GPU: the core's layer over `slots` places for them (the expert cache's, src/offload.rs: every expert in
/// the computer's memory, the ones the router chooses copied into the places), the latent's projections and norm. What it computes is `KimiMoe`'s to the bit; a
/// decode step is not captured in a graph (the choices are read on the host to place the experts).
struct KimiMoeCached {
    moe: KimiMoe,
    store: Rc<ExpertStore>,
    index: usize,
    /// (a decode step's guess of the experts, `Op::MoeEarly`: none where no copies are made for it)
    guess: Option<Guess>,
    /// (a development build's GLYD_KIMI_STATS: the stage's first cached layer reports its store every 64 one-token calls; their count and the time of the last report)
    report: Option<(std::cell::Cell<u32>, std::cell::Cell<std::time::Instant>)>,
}

/// A decode step's early guess of a cached layer's experts: at most `copies` of the ones not in places copied for it, its choices read back to `ids` (pinned) after
/// `ev`, and whether one is queued and not yet taken up. A development build's GLYD_KIMI_EARLY of several counts ("0,1,2") takes them in turn, the stage's
/// layers together, a turn for each of GLYD_KIMI_STATS's reports (`turn`, shared by the stage's layers), to compare them in one run.
struct Guess {
    copies: Rc<[usize]>,
    turn: Rc<std::cell::Cell<usize>>,
    ids: crate::dev::Pinned,
    ev: crate::dev::Ev,
    pending: std::cell::Cell<bool>,
}

/// Copies a layer makes for its early guess at most: none. On 8x A100 (research/kimi-k3-session3-2026-10-22) the guess's copies were mostly experts the router then
/// did not choose, and took the PCIe time of its misses: 64 tokens in 14.7 s with no guess, 15.1-15.6 s with one or two copies a layer (Kimi-K3's attention output
/// weighs too much in its experts' input for the input without it to tell them). A development build's GLYD_KIMI_EARLY sets it, for a better guess.
const GUESS_COPIES: usize = 0;

impl Guess {
    fn copies(&self) -> usize {
        self.copies[self.turn.get() % self.copies.len()]
    }
}

impl MoeBlock for KimiMoeCached {
    fn early_wanted(&self) -> bool {
        self.guess.as_ref().is_some_and(|g| g.copies() > 0)
    }

    /// The router on `h`, its choices read back for `forward` to place before the layer's call (`ExpertStore::guess`).
    unsafe fn early(&self, e: &dyn NhHost, h: u64) -> Result<()> {
        let Some(g) = &self.guess else { return Ok(()) };
        let m = &self.moe;
        let (c, s) = (&m.layer.cfg, e.dev().stream);
        let Some(sc) = e.moe_scratch().get() else { return Ok(()) };
        m.layer.route(sc, s, h, c.router_hidden, 1)?;
        crate::dev::copy_d2h_async(g.ids.ptr, sc.ids.ptr, c.router.top_k * 8, s)?;
        g.ev.record(s)?;
        g.pending.set(true);
        Ok(())
    }

    /// `KimiMoe::forward`'s launches with the experts' products over the places the router's choices were put in; the shared MLP queued before them.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let m = &self.moe;
        let (c, s) = (&m.layer.cfg, e.dev().stream);
        if e.mix().bytes < n * c.router_hidden * 2 || e.xc().bytes < n * c.router_hidden * 2 {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.router_hidden));
        }
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[&m.layer], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        // (the early guess's copies queued: its choices were read back before the attention ran)
        if let Some(g) = self.guess.as_ref().filter(|g| g.pending.replace(false) && n == 1) {
            g.ev.wait_host()?;
            let ids: Vec<u16> = std::slice::from_raw_parts(g.ids.ptr as *const i64, c.router.top_k).iter().map(|&x| x as u16).collect();
            self.store.guess(self.index, &ids, g.copies())?;
        }
        let (z, y, gu) = m.down(e, sc, h, n)?;
        m.layer.route(sc, s, h, c.router_hidden, n)?;
        // (the shared experts queued while the host reads the choices and plans, beside the copies, instead of after them; a decode's call planned on the GPU where the store does: src/devplan.rs)
        m.layer.via_store(
            &self.store,
            self.index,
            crate::offload::At::of(e.dev()),
            sc,
            z,
            n,
            y,
            &mut || m.layer.shared_from(e.lib(), sc, s, gu, n),
            &mut |waves| m.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), waves),
        )?;
        m.k.rms2(s, y, m.latent, m.norm.ptr, z, m.latent, m.latent, m.eps, n as u32)?;
        e.lin(&m.up, z, n, out)?;
        if let Some((c, at)) = self.report.as_ref().filter(|_| n == 1) {
            c.set(c.get() + 1);
            if c.get() % 64 == 0 {
                let early = self.guess.as_ref().map_or(0, Guess::copies);
                eprintln!(
                    "layers from {}: 64 steps in {:.1} ms, {early} early copies a layer; {}",
                    self.index,
                    at.get().elapsed().as_secs_f64() * 1e3,
                    self.store.report()?
                );
                at.set(std::time::Instant::now());
                self.store.reset_stats()?;
                if let Some(g) = &self.guess {
                    g.turn.set(g.turn.get() + 1);
                }
            }
        }
        m.layer.shared_add(sc, s, n, out)
    }

    /// A decode step's calls planned on the GPU (src/devplan.rs) need no host: the step is one graph.
    fn graphable(&self) -> bool {
        self.store.device_on(self.index)
    }

    fn reads_host(&self) -> bool {
        !self.store.device_on(self.index)
    }

    fn bytes(&self) -> usize {
        self.moe.bytes()
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        self.moe.pack_shapes()
    }

    fn store(&self) -> Option<Rc<ExpertStore>> {
        Some(self.store.clone())
    }
}

/// A mixture-of-experts layer of the expert cache across a group of GPUs (src/dynamic/tp_cache.rs): `KimiMoeCached`'s, the layer's stacks this GPU's slices
/// of the places, the plan the first GPU's.
struct KimiMoeTp {
    moe: KimiMoe,
    rank: Rc<super::tp_cache::Rank>,
    index: usize,
    places: super::tp_cache::Places,
    report: Option<(std::cell::Cell<u32>, std::cell::Cell<std::time::Instant>)>,
    /// The device path (`Places::dev`'s table): a decode step's plan and copies on the GPU, its fills and their count; None: every call planned by the
    /// first GPU's thread.
    dev: Option<(crate::dev::Buf, crate::dev::Buf)>,
}

/// The cache's decode steps planned on the GPU (kernels/kimi.cu's dc_plan and dc_fill: no host in a step, so the step is a graph): a development
/// build's GLYD_KIMI_DEVICE=1.
pub fn device_path() -> bool {
    crate::tune!("GLYD_KIMI_DEVICE").as_deref() == Some("1")
}

impl MoeBlock for KimiMoeTp {
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        use super::tp_cache::{Call as TpCall, Rank, Stage as TpStage};
        let m = &self.moe;
        let (c, s) = (&m.layer.cfg, e.dev().stream);
        if e.mix().bytes < n * c.router_hidden * 2 || e.xc().bytes < n * c.router_hidden * 2 {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.router_hidden));
        }
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[&m.layer], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        let (z, y, gu) = m.down(e, sc, h, n)?;
        m.layer.route(sc, s, h, c.router_hidden, n)?;
        // (a decode step on the device path: the places looked up and the misses copied in by the GPU, the products over them; nothing waits for the host)
        if let (1, Some((fills, nfill)), Some(t)) = (n, &self.dev, self.places.dev.as_ref()) {
            let pairs = c.router.top_k;
            let (po, eo, la, rf, me) = t.at();
            let k = &m.k;
            crate::launch!(k.dc_plan, [1, 1, 1], [256, 1, 1], s; sc.ids.ptr, pairs as i32, po, eo, la, rf, me, t.experts as i32, t.slots as i32, sc.way.ptr, fills.ptr, nfill.ptr)?;
            crate::launch!(k.dc_fill, [16, pairs as u32, 1], [256, 1, 1], s; fills.ptr, nfill.ptr, t.reg.ptr)?;
            m.layer.shared_from(e.lib(), sc, s, gu, n)?;
            m.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), 1)?;
            m.layer.experts_sum(sc, s, n, y, 0, 0)?;
            m.k.rms2(s, y, m.latent, m.norm.ptr, z, m.latent, m.latent, m.eps, n as u32)?;
            e.lin(&m.up, z, n, out)?;
            return m.layer.shared_add(sc, s, n, out);
        }
        let call = TpCall { ids: sc.ids.ptr, wts: sc.wts.ptr, z, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs: n * c.router.top_k };
        self.rank.run(self.index, &self.places, &call, s, &mut || m.layer.shared_from(e.lib(), sc, s, gu, n), &mut |stage| match stage {
            TpStage::Products(_, waves, true) => m.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), waves),
            TpStage::Products(..) => Rank::zero(y, n * c.hidden * 2, s),
            TpStage::Sum(extra) => m.layer.experts_sum(sc, s, n, y, 0, extra),
        })?;
        m.k.rms2(s, y, m.latent, m.norm.ptr, z, m.latent, m.latent, m.eps, n as u32)?;
        e.lin(&m.up, z, n, out)?;
        if let Some((cnt, at)) = self.report.as_ref().filter(|_| n == 1) {
            cnt.set(cnt.get() + 1);
            if cnt.get() % 64 == 0 {
                eprintln!("64 steps in {:.1} ms; {}", at.get().elapsed().as_secs_f64() * 1e3, self.rank.report());
                at.set(std::time::Instant::now());
            }
        }
        m.layer.shared_add(sc, s, n, out)
    }

    fn graphable(&self) -> bool {
        self.dev.is_some()
    }

    fn bytes(&self) -> usize {
        self.moe.bytes()
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        self.moe.pack_shapes()
    }
}

/// A layer's experts packed straight into pinned memory, the expert cache's host tier, from the checkpoint (`f` the layer's name): each expert's three
/// matrices read with each thread's own buffers, gate and up packed into the two halves of its first matrix's place and down into its second's, checked
/// there (glyd_gpu::pack's `_into`s); in the layout with packed scales where every expert's scales fit it, else as released. Whether packed, and the
/// two matrices' strides (codes, scales) and buffers. No copy of the layer is made: a load of Kimi-K3 joined each layer's 15 GB, copied it into pinned
/// memory and freed it, and spent about half its time in the TLB shootdowns of that.
fn pack_into_host(
    ck: &Checkpoint,
    f: &str,
    experts: usize,
    inter: usize,
    latent: usize,
    threads: usize,
    portable: bool,
) -> Result<(bool, [(u64, u64, crate::dev::Pinned, crate::dev::Pinned); 2])> {
    use glyd_gpu::pack::{fp4_bytes, pack_fp4_into, pack_fp4r_into};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    for packed in [true, false] {
        let ((gd, gs), (dd, ds)) = (fp4_bytes(inter, latent, packed), fp4_bytes(latent, inter, packed));
        let (ud, us) = (2 * gd, 2 * gs);
        // (portable: pinned for every GPU's copies, where every GPU of a group copies its slices from it)
        let pin = |b: usize| if portable { crate::dev::Pinned::portable(b) } else { crate::dev::Pinned::new(b) };
        let bufs = [pin(experts * ud)?, pin(experts * us)?, pin(experts * dd)?, pin(experts * ds)?];
        let ptrs: Vec<usize> = bufs.iter().map(|b| b.ptr as usize).collect();
        let (next, bad, failed) = (AtomicUsize::new(0), AtomicBool::new(false), std::sync::Mutex::new(None::<Error>));
        let pk = if packed { pack_fp4r_into } else { pack_fp4_into };
        std::thread::scope(|sc| {
            for _ in 0..threads.min(experts).max(1) {
                sc.spawn(|| {
                    let mut raw: [Vec<u8>; 6] = Default::default();
                    loop {
                        let x = next.fetch_add(1, Ordering::Relaxed);
                        if x >= experts || bad.load(Ordering::Relaxed) || failed.lock().map_or(true, |g| g.is_some()) {
                            break;
                        }
                        let read = |w: &str, part: &str, buf: &mut Vec<u8>| -> Result<()> {
                            let n = format!("{f}.experts.{x}.{w}.{part}");
                            let (file, t) = ck.get(&n).ok_or_else(|| Error(format!("no tensor {n}")))?;
                            buf.resize(t.bytes as usize, 0);
                            io(file.read_into_uncached(t, buf))
                        };
                        let r = (|| -> Result<bool> {
                            for (k, (w, part)) in [
                                ("w1", "weight_packed"),
                                ("w1", "weight_scale"),
                                ("w3", "weight_packed"),
                                ("w3", "weight_scale"),
                                ("w2", "weight_packed"),
                                ("w2", "weight_scale"),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                read(w, part, &mut raw[k])?;
                            }
                            // SAFETY: expert x's place in each buffer (x times its stride, its stride long) is written by this item alone, and the buffers live for the scope.
                            let place = |b: usize, stride: usize| unsafe { std::slice::from_raw_parts_mut((ptrs[b] as *mut u8).add(x * stride), stride) };
                            let (ud_, us_, dd_, ds_) = (place(0, ud), place(1, us), place(2, dd), place(3, ds));
                            let (g_d, u_d) = ud_.split_at_mut(gd);
                            let (g_s, u_s) = us_.split_at_mut(gs);
                            Ok(pk(&raw[0], &raw[1], inter, latent, g_d, g_s)
                                && pk(&raw[2], &raw[3], inter, latent, u_d, u_s)
                                && pk(&raw[4], &raw[5], latent, inter, dd_, ds_))
                        })();
                        match r {
                            Ok(true) => {}
                            Ok(false) => bad.store(true, Ordering::Relaxed),
                            Err(e) => {
                                if let Ok(mut g) = failed.lock() {
                                    g.get_or_insert(e);
                                }
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().ok().flatten() {
            return Err(e);
        }
        if !bad.into_inner() {
            let [a, b, c, d] = bufs;
            return Ok((packed, [(ud as u64, us as u64, a, b), (dd as u64, ds as u64, c, d)]));
        }
    }
    err(format!("{f}: experts the FP4 layouts do not hold exactly"))
}

/// The GPU memory one of the experts takes (a place of the expert cache): its first and second matrices in the larger of the two FP4 layouts a layer may be packed
/// in (packed scales where its scales allow, else as released: about 4.5% more).
pub fn expert_bytes(c: &ConfigKimi) -> Result<u64> {
    let e = &c.experts;
    let one = |tag: u32| -> Result<u64> {
        let ((ud, us), (dd, ds)) = (crate::moe::expert_strides(tag, 2 * e.inter, e.latent)?, crate::moe::expert_strides(tag, e.latent, e.inter)?);
        Ok(ud + us + dd + ds)
    };
    Ok(one(glyd_gpu::pack::FP4R_TAG)?.max(one(glyd_gpu::pack::FP4_TAG)?))
}

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_stage(dir, o, Stage::whole(ConfigKimi::load(dir)?.layers))
}

/// Stage `st` of the model (its layers; the embedding where it is the first, the final mix, norm and head where the last), every expert on the GPU.
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    load_with_places(dir, o, st, None)
}

/// Stage `st` with `places` of each mixture-of-experts layer's experts on the GPU (None, or as many as it has: all; fewer: the expert cache, its host tier pinned).
pub fn load_with_places(dir: &Path, o: &Opts, st: Stage, places: Option<usize>) -> Result<Dyn> {
    load_with_caching(dir, o, st, places, None)
}

/// `load_with_places`, a hummingbird directory's experts in the expert cache where `caching` is given (the stage's places and memory, src/hbg_store.rs; `places` is the penguin level's cache's and goes with no hummingbird
/// directory).
pub fn load_with_caching(dir: &Path, o: &Opts, st: Stage, places: Option<usize>, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let c = ConfigKimi::load(dir)?;
    let (kd, md) = (&c.kda, &c.mla);
    if kd.dim != crate::kern_glm::GM_D || c.hidden % 64 != 0 || c.hidden > 8192 || md.kv_rank > 512 || (md.kv_rank + md.rope) % 64 != 0 {
        return err(format!("delta heads of {}, a width of {}, latents of {} + {}: this engine runs heads of 128, widths of 64s up to 8,192 and latents up to 512 (with the shared part, in 64s)", kd.dim, c.hidden, md.kv_rank, md.rope));
    }
    let dev = Dev::new(o.device)?;
    let ck = io(Checkpoint::open(dir))?;
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's, all on the GPU)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let hb_store = caching.map(crate::hbg_store::store_for);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let layout = o.layout.unwrap_or(Layout::Tiered);
    let (h, p) = (c.hidden, c.prefix);
    let shared = Rc::new(KernKimi::load(&dev)?);
    let store = ExpertStore::new();
    // (the host's misses on a crew of 64 threads at most, each kept to a core: a decode step's call of one to three of Kimi-K3's 17 MB experts took 4.2 ms on all
    // 240 of an 8x A100 box's cores, 0.6 ms on 64 (fp4bench); a development build's GLYD_KIMI_HOST: the mode, copy, cpu, split or auto (the default))
    // (the experts in the computer's memory: each expert's bytes dropped from the page cache once they are read and packed, so that each layer's 15 GB of
    // pinned memory comes from what the layers before freed; the download's cache had made each layer wait on reclaim and compaction, most of a 45-minute
    // load. Not the whole checkpoint dropped first: a box whose disk reads 0.5 GB/s (Lambda's 8x A100, 2026-10-22) then read all of it again from disk)
    // (every layer across a group of GPUs: this GPU's slice of a big matrix's rows, of each expert's: src/dynamic/tp.rs)
    let tp = super::tp::comm();
    let crew = crate::cpu_experts::default_threads().min(64);
    let guess_copies: Rc<[usize]> = crate::tune!("GLYD_KIMI_EARLY")
        .and_then(|v| v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<Vec<usize>>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![GUESS_COPIES])
        .into();
    let (guessing, guess_turn) = (guess_copies.iter().any(|&k| k > 0), Rc::new(std::cell::Cell::new(0usize)));
    // (else the caching's mode, where one is given: tiercheck's --mode)
    let mode = crate::tune!("GLYD_KIMI_HOST")
        .as_deref()
        .and_then(crate::offload::Mode::parse)
        .or(caching.map(|c| c.offload.mode))
        .unwrap_or(crate::offload::Mode::Auto);
    let device = caching.is_some_and(|c| c.offload.device);
    store.set_offload(crate::offload::Offload { mode, threads: crew, pin: true, device, ..Default::default() });
    // (the expert cache across a group of GPUs: its host tier and plans shared, this GPU's copies and pinned buffers: src/dynamic/tp_cache.rs)
    let tp_rank = match (&tp, places.filter(|&p| p < c.experts.experts)) {
        (Some(t), Some(_)) => {
            let cache = t.shared(super::tp_cache::TpCache::default)?;
            let slice = expert_bytes(&c)? / t.n as u64;
            Some(Rc::new(super::tp_cache::Rank::new(cache, t.rank, t.n, o.chunk.max(1), c.experts.top_k, c.experts.latent, store.offload(), slice)?))
        }
        _ => None,
    };
    // (this GPU's place and the group's size, for the threads that pack the experts)
    let tpr = tp.as_ref().map(|t| (t.rank, t.n));
    if let Some((_, n)) = tpr.filter(|&(_, n)| (2 * c.experts.inter) % (64 * n) != 0 || c.experts.latent % (64 * n) != 0) {
        return err(format!("experts of {} x {} across {n} GPUs: each GPU's rows in 64s are wanted", 2 * c.experts.inter, c.experts.latent));
    }
    // a matrix packed lossless (rows not a multiple of 64: held dense, as it is); several stacked as one where `names` has more; sliced where `slice` and
    // the group of GPUs takes it
    // (the weights of `names` stacked, this GPU's rows where `slice`: the weights, the whole rows, this GPU's rows, the columns, whether sliced)
    let weights_as = |names: &[String], slice: bool| -> Result<(Vec<u16>, usize, usize, usize, bool)> {
        let (mut parts, mut rows, mut cols) = (Vec::new(), 0usize, 0usize);
        for n in names {
            let name = format!("{n}.weight");
            let (f, t) = ck.get(&name).ok_or_else(|| Error(format!("no tensor {name}")))?;
            if t.shape.len() != 2 || t.dtype != "BF16" || (cols != 0 && t.shape[1] as usize != cols) {
                return err(format!("{name}: {} {:?}, not a bf16 matrix of {cols} columns", t.dtype, t.shape));
            }
            parts.push((f, t, rows));
            rows += t.shape[0] as usize;
            cols = t.shape[1] as usize;
        }
        // (a fused product's rows, several tensors', padded with zero rows to a multiple of 512: 64 a GPU in groups of up to 8, one layout for every
        // group; its parts are read at their own columns, the padding never)
        if names.len() > 1 {
            rows = fused(rows);
        }
        let sliced = tp.as_ref().filter(|t| slice && rows % 64 == 0 && cols % 16 == 0 && t.slices(rows, cols));
        let (r0, rn) = sliced.map_or((0, rows), |t| t.rows(rows));
        // (rows r0..r0 + rn of the matrices stacked, each read as those bytes alone: a GPU of a group reads its slice, not the whole)
        let mut w = Vec::with_capacity(rn * cols);
        for (f, t, at) in parts {
            let (a, b) = (r0.max(at), (r0 + rn).min(at + t.shape[0] as usize));
            if a >= b {
                continue;
            }
            let sub = glyd_gpu::safetensors::TensorInfo { offset: t.offset + ((a - at) * cols * 2) as u64, bytes: ((b - a) * cols * 2) as u64, ..t.clone() };
            let bytes = io(f.read(&sub))?;
            w.extend(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
        w.resize(rn * cols, 0);
        Ok((w, rows, rn, cols, sliced.is_some()))
    };
    let pack_as = |names: &[String], slice: bool| -> Result<Mat> {
        let (w, rows, rn, cols, sliced) = weights_as(names, slice)?;
        if rows % 64 != 0 || cols % 16 != 0 {
            return Ok(Mat::Dense(upload_bytes(as_bytes(&w))?, rows, cols));
        }
        let mut pk = upload_packed(pack_words(&w, rn, cols, layout, &names[0])?)?;
        if sliced {
            pk.tp_rows = rows;
        }
        Ok(Mat::Packed(pk))
    };
    // (a matrix of a layer whose matrices are in the computer's memory, src/dense_ram.rs: packed here, its bytes kept for the slot; never sliced)
    let host = |names: &[String]| -> Result<crate::weights::Packed> {
        let (w, rows, _, cols, _) = weights_as(names, false)?;
        if rows % 64 != 0 || cols % 16 != 0 {
            return err(format!("{}: {rows} x {cols}, not a packed matrix, in the computer's memory", names[0]));
        }
        pack_words(&w, rows, cols, layout, &names[0])
    };
    let pack = |names: &[String]| pack_as(names, true);
    let one = |n: String| pack(&[n]);
    let small = |n: &str| -> Result<Buf> { upload_bytes(&io(ck.read(n))?) };
    let mut l_ = Loaded::default();
    // (the stage's last layers' matrices in the computer's memory where the cache says so, src/dense_ram.rs: the delta attention's three; the latent attention's blocks keep theirs)
    let mut ram = crate::dense_ram::RamBuilder::new(crate::dense_ram::Placer::last(st.layers.0, st.layers.1, caching.map_or(0, |x| x.dense)));
    if tp.is_some() && caching.is_some_and(|x| x.dense > 0) {
        return err("Kimi-K3's dense layers in the computer's memory: a stage on one GPU, not every layer across GPUs");
    }
    let (mut states, mut layers, mut rings) = (Vec::new(), Vec::new(), Vec::new());
    let mla = latent(&c);
    let put = |v: &mut Vec<Buf>, b: Buf| {
        v.push(b);
        v.len() - 1
    };
    let put_m = |v: &mut Vec<Mat>, m: Mat| {
        v.push(m);
        v.len() - 1
    };
    let (qd, nh) = (kd.heads * kd.dim, kd.heads);
    let w32 = dtypes(&ck, &c)?;
    let early = guessing && places.is_some_and(|p| p < c.experts.experts);
    // (Mix holds the delta attention's fused product, Proj a prompt's forget-gate values copied out of it)
    let (mut mix_w, mut proj_w) = (3 * qd, kd.dim);
    let e = &c.experts;
    for l in 0..c.layers {
        // (the blocks completed before layer l: the embedding at 0, then one each res_block layers)
        let before = l.div_ceil(c.res_block);
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        crate::status::STATUS.beat();
        let q = format!("{p}layers.{l}");
        let a = format!("{q}.self_attn");
        let (in_n, post_n) =
            (put(&mut l_.small, small(&format!("{q}.input_layernorm.weight"))?), put(&mut l_.small, small(&format!("{q}.post_attention_layernorm.weight"))?));
        let res = |sm: &mut Vec<Buf>, n: &str| -> Result<(usize, usize)> {
            Ok((put(sm, small(&format!("{q}.{n}_res_norm.weight"))?), put(sm, small(&format!("{q}.{n}_res_proj.weight"))?)))
        };
        let (an, ap) = res(&mut l_.small, "self_attention")?;
        let (mn, mp) = res(&mut l_.small, "mlp")?;
        // (the layer before's output added here where that layer is this stage's: its add is this mix's; a block's first layer makes X a block)
        let mut ops = vec![Op::DepthMix(DepthMixOp {
            add: l > st.layers.0,
            nb: before,
            w_n: an,
            w_p: ap,
            push: (l % c.res_block == 0).then_some(before),
            w: in_n,
            eps: c.eps,
        })];
        if c.moe[l] && early {
            ops.push(Op::MoeEarly {
                block: l_.moe.len(),
                mix: DepthMixOp { add: false, nb: before + usize::from(l % c.res_block == 0), w_n: mn, w_p: mp, push: None, w: post_n, eps: c.eps },
            });
        }
        match c.kinds[l] {
            GlmKind::Kda => {
                let m0 = l_.mats.len();
                // (q, k, v, the output gate, the forget gate's first product and beta: one product of the layer's input, three fewer a layer; the gates read
                // where it leaves them, a prompt's copied out first: Op::Parts)
                let fa =
                    ck.get(&format!("{a}.f_a_proj.weight")).map(|(_, t)| t.shape[0] as usize).ok_or_else(|| Error(format!("no tensor {a}.f_a_proj.weight")))?;
                if fa != kd.dim {
                    return err(format!("{a}.f_a_proj.weight: a forget gate of rank {fa}, where the config's delta heads are {}", kd.dim));
                }
                let names = ["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"].map(|x| format!("{a}.{x}"));
                if ram.in_ram(l) {
                    for n in [names.to_vec(), vec![format!("{a}.o_proj")], vec![format!("{a}.f_b_proj")]] {
                        ram.push(l, &mut l_.mats, host(&n)?)?;
                    }
                } else {
                    l_.mats.push(pack(&names)?);
                    l_.mats.push(one(format!("{a}.o_proj"))?);
                    l_.mats.push(one(format!("{a}.f_b_proj"))?);
                }
                let row = l_.mats[m0].whole_rows();
                (mix_w, proj_w) = (mix_w.max(row), proj_w.max(fa));
                // (small: the three convolutions' weights as one, A_log, dt_bias (fp32), the gated norm's weight; the convolutions' and the norm's bf16 or fp32 as
                // released (Kimi-K3's are fp32: their values, not bf16 ones), read by the kernels as they are)
                let mut conv = Vec::new();
                let (cw32, nw32) = w32[l];
                for x in ["q", "k", "v"] {
                    conv.extend(io(ck.read(&format!("{a}.{x}_conv1d.weight")))?);
                }
                // (the kernels read A_log for each head, dt_bias for each channel, each convolution's taps for each channel and the gated norm for each head
                // channel: a tensor shorter than that would be read past its end on the GPU. A longer one is read in its first values, as the release's A_log
                // of 128 for 96 heads is)
                let elems = |n: &str| ck.get(n).map(|(_, t)| t.shape.iter().product::<u64>() as usize).ok_or_else(|| Error(format!("no tensor {n}")));
                let wanted = [(format!("{a}.A_log"), nh), (format!("{a}.dt_bias"), qd), (format!("{a}.o_norm.weight"), kd.dim)]
                    .into_iter()
                    .chain(["q", "k", "v"].map(|x| (format!("{a}.{x}_conv1d.weight"), qd * kd.conv_kernel)));
                for (n, want) in wanted {
                    if elems(&n)? < want {
                        return err(format!("{n}: {} values, where the kernels read {want}", elems(&n)?));
                    }
                }
                let s0 = l_.small.len();
                l_.small.extend([
                    upload_bytes(&conv)?,
                    f32s(&ck, &format!("{a}.A_log"))?,
                    f32s(&ck, &format!("{a}.dt_bias"))?,
                    small(&format!("{a}.o_norm.weight"))?,
                ]);
                let (cv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * 3 * qd * 2);
                states.push(nh * kd.dim * kd.dim * 4);
                ops.extend([
                    Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "delta attention: q, k, v and gates product" },
                    Op::KdaConv { w: s0, state: cv, c: 3 * qd, row, w32: cw32 },
                    Op::Parts { row, parts: [(3 * qd, qd, Slot::Iq), (4 * qd, fa, Slot::Proj), (4 * qd + fa, nh, Slot::Ba)] },
                    Op::LinearAt { w: m0 + 2, x: View::at(Slot::Proj, 4 * qd), y: Slot::Qo, name: "delta attention: forget gate product" },
                    Op::KdaDelta { a_log: s0 + 1, dt_bias: s0 + 2, state: ssm, heads: nh, bound: kd.bound, ba: View::at(Slot::Ba, 4 * qd + fa) },
                    Op::KdaOut { w: s0 + 3, heads: nh, w32: nw32, iq: View::at(Slot::Iq, 3 * qd) },
                    Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "delta attention: o product" },
                ]);
            }
            GlmKind::Mla => {
                // (q_a, the gate and kv_a one product, q_b, o; the two norms and kv_b_proj as the bf16 it is: the absorbed form multiplies by its rows)
                let (wkv, _) = words(&ck, &format!("{a}.kv_b_proj.weight"))?;
                let (m0, s0, ring) = (l_.mats.len(), l_.small.len(), rings.len());
                l_.mats.extend([
                    pack(&[format!("{a}.q_a_proj"), format!("{a}.g_proj"), format!("{a}.kv_a_proj_with_mqa")])?,
                    one(format!("{a}.q_b_proj"))?,
                    one(format!("{a}.o_proj"))?,
                ]);
                l_.small.extend([small(&format!("{a}.q_a_layernorm.weight"))?, small(&format!("{a}.kv_a_layernorm.weight"))?, upload_bytes(as_bytes(&wkv))?]);
                let op = mla.placed(m0, s0, ring);
                rings.extend(op.rings());
                ops.push(Op::Mla(op));
            }
        }
        ops.push(Op::DepthMix(DepthMixOp { add: true, nb: before + usize::from(l % c.res_block == 0), w_n: mn, w_p: mp, push: None, w: post_n, eps: c.eps }));
        if c.moe[l] {
            let f = format!("{q}.block_sparse_moe");
            let cfg = MoeCfg {
                experts: e.experts,
                router_hidden: h,
                hidden: e.latent,
                inter: e.inter,
                router: Router { renorm: e.renorm, ..Router::sigmoid(e.top_k, e.scale) },
                act: Act::Situ { beta: c.situ.0, linear: c.situ.1 },
            };
            // the experts as released, packed exactly in the smallest FP4 layout that holds them all (DeepSeek-V4's: packed scales, else as released): gate and up stacked as the first matrix;
            // all on the GPU, or places for `p` of them and every expert in the computer's memory, packed there straight from the checkpoint (pack_into_host)
            let slots = places.filter(|&p| p < e.experts);
            let mut tp_places = None;
            let hbx = crate::hbg_store::hb_layer(
                hb.as_ref(),
                l,
                tpr.is_some(),
                caching.zip(hb_store.as_ref()),
                cfg.act,
                crate::cpu_hbg::Finish::Rotated,
                dev.ordinal,
                e.top_k,
            )?;
            if hbx.is_some() && slots.is_some() {
                return err(format!(
                    "{f}: a hummingbird directory's experts are in the hummingbird expert cache (src/hbg_store.rs), not the penguin level's places"
                ));
            }
            let (up, down) = if hbx.is_some() {
                (ExpertStack::shape_only(e.experts, 2 * e.inter, e.latent), ExpertStack::shape_only(e.experts, e.latent, e.inter))
            } else {
                match slots {
                    Some(p) if tp_rank.is_some() => {
                        let (rank, n) = tpr.ok_or_else(|| Error("a group of GPUs without its place".into()))?;
                        let cache = &tp_rank.as_ref().expect("a group's cache").cache_ref();
                        // (the layers' host tiers packed by the GPUs' threads in turn, each the layers of its number, and shared)
                        let host = if l % n == rank {
                            let (packed, packs) = pack_into_host(&ck, &f, e.experts, e.inter, e.latent, threads, true)?;
                            let [(ud, us, a, b), (dd, ds, c2, d2)] = packs;
                            use super::tp_cache::HostBuf;
                            let h = std::sync::Arc::new(super::tp_cache::HostLayer {
                                mats: [[HostBuf(a), HostBuf(b)], [HostBuf(c2), HostBuf(d2)]],
                                stride: [[ud, us], [dd, ds]],
                                packed,
                                inter: e.inter,
                                latent: e.latent,
                                act: c.situ,
                            });
                            cache.put_host(l, h.clone())?;
                            h
                        } else {
                            cache.host(l)?
                        };
                        let tag = if host.packed { glyd_gpu::pack::FP4R_TAG } else { glyd_gpu::pack::FP4_TAG };
                        let mut up = ExpertStack::pooled(tag, [tag << 24, 0, 0], e.experts, p, 2 * e.inter / n, e.latent)?;
                        let mut down = ExpertStack::pooled(tag, [tag << 24, 0, 0], e.experts, p, e.latent / n, e.inter)?;
                        let slice = [[up.data_stride, up.scale_stride], [down.data_stride, down.scale_stride]];
                        if (0..2).any(|m| (0..2).any(|k| slice[m][k] * n as u64 != host.stride[m][k])) {
                            return err(format!("{f}: experts of {:?} bytes in the computer's memory, where {n} GPUs' places hold {slice:?}", host.stride));
                        }
                        (up.tp_rows, down.tp_rows) = (2 * e.inter, e.latent);
                        let mut pl = super::tp_cache::Places::new([up.buffers(), down.buffers()], slice, e.experts, p)?;
                        // (the device path places a step's experts at once: a layer of fewer places than a token's choices is the host's, in waves)
                        if device_path() && p >= e.top_k {
                            pl.attach_device(&host, rank)?;
                        }
                        tp_places = Some(pl);
                        (up, down)
                    }
                    None => {
                        let fp4 = |x: usize, packed_scales: bool| -> Result<Option<(glyd_gpu::pack::Tiered, glyd_gpu::pack::Tiered)>> {
                            let raw = |w: &str| -> Result<(Vec<u8>, Vec<u8>)> {
                                let b = format!("{f}.experts.{x}.{w}");
                                Ok((io(ck.read(&format!("{b}.weight_packed")))?, io(ck.read(&format!("{b}.weight_scale")))?))
                            };
                            let pk = if packed_scales { glyd_gpu::pack::pack_fp4r_checked } else { glyd_gpu::pack::pack_fp4_checked };
                            let ((mut uq, mut ue), (gq, ge), (dq, de)) = (raw("w1")?, raw("w3")?, raw("w2")?);
                            uq.extend(gq);
                            ue.extend(ge);
                            // (a group of GPUs: this GPU's rows of each, codes half a byte a value and a scale a run of 32)
                            let rows_of = |q: &[u8], sc: &[u8], rows: usize, cols: usize| -> (Vec<u8>, Vec<u8>, usize) {
                                match tpr {
                                    Some((rank, n)) => {
                                        let (r0, rn) = (rank * rows / n, rows / n);
                                        (q[r0 * cols / 2..(r0 + rn) * cols / 2].to_vec(), sc[r0 * cols / 32..(r0 + rn) * cols / 32].to_vec(), rn)
                                    }
                                    None => (q.to_vec(), sc.to_vec(), rows),
                                }
                            };
                            let (uq, ue, ur) = rows_of(&uq, &ue, 2 * e.inter, e.latent);
                            let (dq, de, dr) = rows_of(&dq, &de, e.latent, e.inter);
                            Ok(pk(&uq, &ue, ur, e.latent).zip(pk(&dq, &de, dr, e.inter)))
                        };
                        let (mut ups, mut downs) = (Vec::with_capacity(e.experts), Vec::with_capacity(e.experts));
                        for packed_scales in [true, false] {
                            (ups.clear(), downs.clear());
                            let mut all = true;
                            crate::weights::ordered(e.experts, threads, 2 * threads, &|x| fp4(x, packed_scales), |_, pr| {
                                match pr {
                                    Some((u, d)) => (ups.push(u), downs.push(d)),
                                    None => (all = false, ()),
                                };
                                Ok(())
                            })?;
                            if all {
                                break;
                            }
                        }
                        if ups.len() != e.experts {
                            return err(format!("{f}: experts the FP4 layouts do not hold exactly"));
                        }
                        let (ju, jd) = (Joined::of(&ups.iter().collect::<Vec<_>>())?, Joined::of(&downs.iter().collect::<Vec<_>>())?);
                        drop((ups, downs));
                        let (mut su, mut sd) = (ExpertStack::upload(&ju)?, ExpertStack::upload(&jd)?);
                        if let Some((_, n)) = tpr {
                            (su.tp_rows, sd.tp_rows) = (su.rows * n, sd.rows * n);
                        }
                        (su, sd)
                    }
                    Some(p) => {
                        let (packed, packs) = pack_into_host(&ck, &f, e.experts, e.inter, e.latent, threads, false)?;
                        let tag = if packed { glyd_gpu::pack::FP4R_TAG } else { glyd_gpu::pack::FP4_TAG };
                        let up = ExpertStack::pooled(tag, [tag << 24, 0, 0], e.experts, p, 2 * e.inter, e.latent)?;
                        let down = ExpertStack::pooled(tag, [tag << 24, 0, 0], e.experts, p, e.latent, e.inter)?;
                        let [(ud, us, a, b), (dd, ds, c2, d2)] = packs;
                        if [ud, us, dd, ds] != [up.data_stride, up.scale_stride, down.data_stride, down.scale_stride] {
                            return err(format!(
                                "{f}: experts of {ud} + {us} and {dd} + {ds} bytes, where the places hold {} + {} and {} + {}",
                                up.data_stride, up.scale_stride, down.data_stride, down.scale_stride
                            ));
                        }
                        let host = vec![
                            PinnedPack { data: a, scales: b, stride: [ud, us], dev: up.buffers() },
                            PinnedPack { data: c2, scales: d2, stride: [dd, ds], dev: down.buffers() },
                        ];
                        // (the host computes the misses it is better at from the host tier's bytes as they are: src/cpu_fp4.rs)
                        let half = crate::offload::HostHalf::Fp4 {
                            inter: e.inter,
                            latent: e.latent,
                            packed,
                            act: crate::cpu_fp4::Act::Situ { beta: c.situ.0, linear: c.situ.1 },
                        };
                        store.add_layer_pinned(l, e.experts, p, host, Some(half))?;
                        (up, down)
                    }
                }
            };
            // (the latent's down projection and the shared experts' gate and up, one product of the layer's input: one fewer a layer, and one gather
            // fewer across GPUs; the shared expert's first matrix a shape there, never multiplied by)
            let dg = pack(&[format!("{f}.routed_expert_down_proj"), format!("{f}.shared_experts.gate_proj"), format!("{f}.shared_experts.up_proj")])?;
            mix_w = mix_w.max(dg.whole_rows());
            let Mat::Packed(sd) = one(format!("{f}.shared_experts.down_proj"))? else { return err(format!("{f}.shared_experts.down_proj: not packed")) };
            let si = e.shared * e.inter;
            let w = MoeWeights {
                router: small(&format!("{f}.gate.weight"))?,
                bias: Some(f32s(&ck, &format!("{f}.gate.e_score_correction_bias"))?),
                escale: None,
                up,
                down: Down::Block(down),
            };
            let sh = Shared { inter: si, up: PackBuf::shape_only(2 * si, h)?, down: sd, gate: None };
            let layer = match hbx {
                Some(x) => x.into_layer(cfg, w.router, w.bias, None, Some(sh))?,
                None => MoeLayer::new(cfg, w, Some(sh))?,
            };
            let b = KimiMoe {
                layer,
                dg,
                up: one(format!("{f}.routed_expert_up_proj"))?,
                norm: small(&format!("{f}.routed_expert_norm.weight"))?,
                latent: e.latent,
                eps: c.eps,
                k: shared.clone(),
            };
            ops.push(Op::Moe(MoeOp::of(l_.moe.len())));
            match slots {
                None => l_.moe.push(Box::new(b)),
                Some(_) if tp_places.is_some() => {
                    let report = (crate::tune!("GLYD_KIMI_STATS").is_some() && tpr.is_some_and(|r| r.0 == 0) && !l_.moe.iter().any(|b| b.resident().is_none()))
                        .then(|| (std::cell::Cell::new(0), std::cell::Cell::new(std::time::Instant::now())));
                    let dev = if tp_places.as_ref().is_some_and(|pl| pl.dev.is_some()) {
                        Some((crate::dev::Buf::new(e.top_k * 8)?, crate::dev::Buf::zeroed(4)?))
                    } else {
                        None
                    };
                    l_.moe.push(Box::new(KimiMoeTp {
                        moe: b,
                        rank: tp_rank.clone().expect("a group's cache"),
                        index: l,
                        places: tp_places.take().expect("the places"),
                        report,
                        dev,
                    }))
                }
                Some(_) => {
                    let report = (crate::tune!("GLYD_KIMI_STATS").is_some() && !l_.moe.iter().any(|b| b.resident().is_none()))
                        .then(|| (std::cell::Cell::new(0), std::cell::Cell::new(std::time::Instant::now())));
                    let guess = guessing
                        .then(|| -> Result<Guess> {
                            Ok(Guess {
                                copies: guess_copies.clone(),
                                turn: guess_turn.clone(),
                                ids: crate::dev::Pinned::new(e.top_k * 8)?,
                                ev: crate::dev::Ev::new()?,
                                pending: Default::default(),
                            })
                        })
                        .transpose()?;
                    l_.moe.push(Box::new(KimiMoeCached { moe: b, store: store.clone(), index: l, guess, report }))
                }
            }
        } else {
            let gu = put_m(&mut l_.mats, pack(&[format!("{q}.mlp.gate_proj"), format!("{q}.mlp.up_proj")])?);
            let down = put_m(&mut l_.mats, one(format!("{q}.mlp.down_proj"))?);
            ops.extend([
                Op::Linear { w: gu, x: Slot::H, y: Slot::Gu, name: "MLP: gate and up product" },
                Op::GatedAct { inter: c.inter, act: Act::Situ { beta: c.situ.0, linear: c.situ.1 } },
                Op::Linear { w: down, x: Slot::M, y: Slot::O, name: "MLP: down product" },
            ]);
        }
        // (added by the next layer's mix, or the head's, where that is this stage's; X whole at a stage's end)
        if l + 1 == st.layers.1 && !st.last {
            ops.push(Op::Add { x: Slot::O });
        }
        layers.push(ops);
    }
    if st.last {
        // the final depth mix of every block and the last partial sum, the norm into H (the head reads it)
        let (on, opj, norm) = (
            put(&mut l_.small, small(&format!("{p}output_attn_res_norm.weight"))?),
            put(&mut l_.small, small(&format!("{p}output_attn_res_proj.weight"))?),
            put(&mut l_.small, small(&format!("{p}norm.weight"))?),
        );
        let all = c.layers.div_ceil(c.res_block);
        layers.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::DepthMix(DepthMixOp {
            add: st.layers.1 > st.layers.0,
            nb: all,
            w_n: on,
            w_p: opj,
            push: None,
            w: norm,
            eps: c.eps,
        }));
    }
    // (a stage without the embedding or the head keeps their places, empty, so that their indices hold)
    let empty = || -> Result<Mat> { Ok(Mat::Dense(Buf::new(2)?, 0, 0)) };
    let em = l_.mats.len();
    // (the embedding whole on every GPU: a token's row is read from it)
    l_.mats.push(if st.first { pack_as(&[format!("{p}embed_tokens")], false)? } else { empty()? });
    let lm = l_.mats.len();
    l_.mats.push(if st.last { one("language_model.lm_head".to_string())? } else { empty()? });
    let blocks = c.layers.div_ceil(c.res_block);
    let mut widths = [0usize; super::SLOTS];
    let gu = 2 * c.inter;
    for (s, wd) in [
        (Slot::X, h),
        (Slot::H, h),
        (Slot::O, h),
        (Slot::Blk, blocks * h),
        (Slot::Mix, mix_w.max(h)),
        (Slot::Xc, (3 * qd).max(h)),
        (Slot::Proj, proj_w),
        (Slot::Qo, qd),
        (Slot::Iq, qd),
        (Slot::Ba, nh.max(8)),
        (Slot::Core, qd),
        (Slot::Att, qd),
        (Slot::Gu, gu),
        (Slot::M, c.inter),
    ] {
        widths[s as usize] = wd;
    }
    let d = ModelDesc {
        family: "Kimi-K3",
        hidden: h,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
        rings,
        // (the embedding into X, the partial sum: KimiPush makes it the first block)
        first: Op::Nop,
        layers,
        states,
        final_norm: None,
        decoded: false,
        skinny: false,
        embed: em,
        lm,
        softcap: 0.0,
        eos: c.eos.clone(),
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    };
    l_.ram = ram.finish(&mut l_.mats)?;
    // (the loader's own operations, as it indexed what it loaded, held to the layer kinds' description of the config: one model, or it is not loaded)
    let d = super::lower::agree(lowered(dir, &c, &st, &w32, early)?, &d, &dir.join("config.json").display().to_string())?;
    Dyn::from_parts(dev, d, l_, Vec::new(), layout, st, o)
}

/// Kimi-K3's latent attention as the layer kind's operation (`MlaOp`, its matrices, small tensors and ring at 0: `MlaOp::placed` puts a layer's): q_a, the
/// output gate and kv_a one product, the q latent's norm KimiRMSNorm (rounded twice, its epsilon 1e-6), every position attended to.
pub(crate) fn latent(c: &ConfigKimi) -> MlaOp {
    let m = &c.mla;
    MlaOp {
        heads: m.heads,
        nope: m.nope,
        rope: m.rope,
        v: m.v,
        kv_rank: m.kv_rank,
        q_rank: m.q_rank,
        input: MlaIn::Fused { w: 0 },
        q_b: 0,
        o: 0,
        q_norm: 0,
        kv_norm: 0,
        kv_b: 0,
        q_round: crate::runtime::kinds::Rounding::Twice,
        gate: true,
        scale: ((m.nope + m.rope) as f32).powf(-0.5),
        eps: 1e-6,
        ring: 0,
        indexer: None,
    }
}

/// A fused product's rows (several tensors' stacked): padded with zero rows to a multiple of 512, 64 a GPU in groups of up to 8.
pub(crate) fn fused(rows: usize) -> usize {
    rows.div_ceil(512) * 512
}

/// Each layer's (convolutions fp32, gated norm fp32) as the checkpoint holds them (F32 or BF16; any other type refused), a delta layer's three convolutions alike;
/// (false, false) for a latent attention layer.
/// The kernels read them as they are (Kimi-K3 releases them fp32: values no bf16 holds).
pub(crate) fn dtypes(ck: &Checkpoint, c: &ConfigKimi) -> Result<Vec<(bool, bool)>> {
    // (F32 or BF16 alone: the kernels read 4 or 2 bytes a value by these flags, and the loader's checks count values, so a narrower type would be read past its end)
    let dtype = |n: &str| -> Result<bool> {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        match t.dtype.as_str() {
            "F32" => Ok(true),
            "BF16" => Ok(false),
            d => err(format!("{n}: {d}, where F32 or BF16 is read")),
        }
    };
    (0..c.layers)
        .map(|l| {
            if c.kinds[l] != GlmKind::Kda {
                return Ok((false, false));
            }
            let a = format!("{}layers.{l}.self_attn", c.prefix);
            let cw32 = dtype(&format!("{a}.q_conv1d.weight"))?;
            for x in ["k", "v"] {
                let n = format!("{a}.{x}_conv1d.weight");
                if dtype(&n)? != cw32 {
                    return err(format!("{n}: not the queries' convolution's dtype ({})", if cw32 { "F32" } else { "bf16" }));
                }
            }
            Ok((cw32, dtype(&format!("{a}.o_norm.weight"))?))
        })
        .collect()
}

/// Kimi-K3's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::kimi`), held to `describe`. `w32` the delta
/// layers' dtypes (`dtypes`), `early` whether the cached experts are guessed and copied ahead (the `MoeEarly` step).
pub fn lowered(dir: &Path, c: &ConfigKimi, st: &Stage, w32: &[(bool, bool)], early: bool) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let lowered = super::lower::kimi(&kinds, c.eos.clone(), c.max_pos, st, w32, early)?;
    super::lower::agree(lowered, &describe(c, st, w32, early)?, &label)
}

/// The executor's description of Kimi-K3 for stage `st`, as the loader indexes what it loads. Each layer's small tensors: its input and post-attention norms, the
/// two depth mixes' norms and projections, then a delta layer's convolutions, A_log, dt_bias and gated norm; the last stage's output mix's norm and projection and
/// the final norm after the last layer's. Its matrices: a delta layer's three (q, k, v, the output gate, the forget gate's first product and beta as one; o; the
/// forget gate's second product), a dense MLP's gate and up and its down; then the embedding and the head. A latent attention layer's and a mixture of experts' are
/// their blocks'.
pub fn describe(c: &ConfigKimi, st: &Stage, w32: &[(bool, bool)], early: bool) -> Result<ModelDesc> {
    let (kd, e) = (&c.kda, &c.experts);
    let (h, qd, nh) = (c.hidden, kd.heads * kd.dim, kd.heads);
    let starts: Vec<bool> = (0..c.layers).map(|l| l % c.res_block == 0).collect();
    let moe_rows = e.latent + 2 * e.shared * e.inter;
    let kinds: Vec<KimiLayer> =
        (0..c.layers).map(|l| KimiLayer { kda: c.kinds[l] == GlmKind::Kda, moe: c.moe[l], w32: w32.get(l).copied().unwrap_or_default() }).collect();
    Ok(kimi_desc(
        &KimiDims { h, qd, nh, kd: kd.dim, bound: kd.bound, inter: c.inter, situ: c.situ, eps: c.eps, moe_rows, mla: Some(latent(c)) },
        &kinds,
        &starts,
        st,
        early,
        c.vocab,
        c.max_pos,
        c.eos.clone(),
    ))
}

/// What a Kimi-K3 layer is to the executor: delta attention or latent attention, a mixture of experts or a dense MLP, its delta layer's dtypes.
pub(crate) struct KimiLayer {
    pub kda: bool,
    pub moe: bool,
    pub w32: (bool, bool),
}

/// Kimi-K3's widths and constants as the executor's operations take them: the width, the delta heads' channels, heads and size, the forget gate's bound, the dense
/// MLP's width, SiTU's two betas, the norms' epsilon, the rows of a mixture of experts' first product (the latent's down projection and the shared experts' gate and up),
/// and the latent attention (`latent`, unplaced).
pub(crate) struct KimiDims {
    pub h: usize,
    pub qd: usize,
    pub nh: usize,
    pub kd: usize,
    pub bound: f32,
    pub inter: usize,
    pub situ: (f32, f32),
    pub eps: f32,
    pub moe_rows: usize,
    pub mla: Option<MlaOp>,
}

/// Kimi-K3's executor description: its layers `ls`, the depth mix's groups starting where `starts` says, for stage `st`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kimi_desc(k: &KimiDims, ls: &[KimiLayer], starts: &[bool], st: &Stage, early: bool, vocab: usize, max_pos: usize, eos: Vec<i64>) -> ModelDesc {
    let (h, qd, nh, eps) = (k.h, k.qd, k.nh, k.eps);
    // (the delta attention's fused product: q, k, v, the output gate, the forget gate's first product and beta, padded)
    let (fa, row) = (k.kd, fused(4 * qd + k.kd + nh));
    let (mut mix_w, mut proj_w) = (3 * qd, k.kd);
    let (mut n_small, mut n_mats, mut n_moe, mut states, mut layers, mut rings) = (0, 0, 0, Vec::new(), Vec::new(), Vec::new());
    for (l, ly) in ls.iter().enumerate() {
        // (the depth mix's groups before this layer: a group starts at its first layer)
        let before = starts[..l].iter().filter(|&&s| s).count();
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let s = n_small;
        let (in_n, post_n, an, ap, mn, mp) = (s, s + 1, s + 2, s + 3, s + 4, s + 5);
        n_small += 6;
        let nb = before + usize::from(starts[l]);
        let mut ops = vec![Op::DepthMix(DepthMixOp { add: l > st.layers.0, nb: before, w_n: an, w_p: ap, push: starts[l].then_some(before), w: in_n, eps })];
        if ly.moe && early {
            ops.push(Op::MoeEarly { block: n_moe, mix: DepthMixOp { add: false, nb, w_n: mn, w_p: mp, push: None, w: post_n, eps } });
        }
        if ly.kda {
            let m0 = n_mats;
            n_mats += 3;
            (mix_w, proj_w) = (mix_w.max(row), proj_w.max(fa));
            let s0 = n_small;
            n_small += 4;
            let (cv, ssm) = (states.len(), states.len() + 1);
            states.push(3 * 3 * qd * 2);
            states.push(nh * k.kd * k.kd * 4);
            ops.extend([
                Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "delta attention: q, k, v and gates product" },
                Op::KdaConv { w: s0, state: cv, c: 3 * qd, row, w32: ly.w32.0 },
                Op::Parts { row, parts: [(3 * qd, qd, Slot::Iq), (4 * qd, fa, Slot::Proj), (4 * qd + fa, nh, Slot::Ba)] },
                Op::LinearAt { w: m0 + 2, x: View::at(Slot::Proj, 4 * qd), y: Slot::Qo, name: "delta attention: forget gate product" },
                Op::KdaDelta { a_log: s0 + 1, dt_bias: s0 + 2, state: ssm, heads: nh, bound: k.bound, ba: View::at(Slot::Ba, 4 * qd + fa) },
                Op::KdaOut { w: s0 + 3, heads: nh, w32: ly.w32.1, iq: View::at(Slot::Iq, 3 * qd) },
                Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "delta attention: o product" },
            ]);
        } else {
            let op = k.mla.as_ref().expect("a latent attention layer's kind").placed(n_mats, n_small, rings.len());
            (n_mats, n_small) = (n_mats + op.mat_count(), n_small + op.small_count());
            rings.extend(op.rings());
            ops.push(Op::Mla(op));
        }
        ops.push(Op::DepthMix(DepthMixOp { add: true, nb, w_n: mn, w_p: mp, push: None, w: post_n, eps }));
        if ly.moe {
            mix_w = mix_w.max(fused(k.moe_rows));
            ops.push(Op::Moe(MoeOp::of(n_moe)));
            n_moe += 1;
        } else {
            let (gu, down) = (n_mats, n_mats + 1);
            n_mats += 2;
            ops.extend([
                Op::Linear { w: gu, x: Slot::H, y: Slot::Gu, name: "MLP: gate and up product" },
                Op::GatedAct { inter: k.inter, act: Act::Situ { beta: k.situ.0, linear: k.situ.1 } },
                Op::Linear { w: down, x: Slot::M, y: Slot::O, name: "MLP: down product" },
            ]);
        }
        if l + 1 == st.layers.1 && !st.last {
            ops.push(Op::Add { x: Slot::O });
        }
        layers.push(ops);
    }
    if st.last {
        let all = starts.iter().filter(|&&s| s).count();
        if let Some(last) = layers.last_mut() {
            last.push(Op::DepthMix(DepthMixOp { add: st.layers.1 > st.layers.0, nb: all, w_n: n_small, w_p: n_small + 1, push: None, w: n_small + 2, eps }));
        }
    }
    let blocks = starts.iter().filter(|&&s| s).count();
    let mut widths = [0usize; super::SLOTS];
    for (s, wd) in [
        (Slot::X, h),
        (Slot::H, h),
        (Slot::O, h),
        (Slot::Blk, blocks * h),
        (Slot::Mix, mix_w.max(h)),
        (Slot::Xc, (3 * qd).max(h)),
        (Slot::Proj, proj_w),
        (Slot::Qo, qd),
        (Slot::Iq, qd),
        (Slot::Ba, nh.max(8)),
        (Slot::Core, qd),
        (Slot::Att, qd),
        (Slot::Gu, 2 * k.inter),
        (Slot::M, k.inter),
    ] {
        widths[s as usize] = wd;
    }
    ModelDesc {
        family: "Kimi-K3",
        hidden: h,
        vocab,
        eps,
        max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: Op::Nop,
        layers,
        states,
        final_norm: None,
        decoded: false,
        skinny: false,
        embed: n_mats,
        lm: n_mats + 1,
        softcap: 0.0,
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    }
}
