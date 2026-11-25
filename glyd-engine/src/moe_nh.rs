//! Nemotron 3 Super's latent mixture-of-experts layer on the GPU (`NemotronHMoE`): `NhMoe`, the `MoeBlock` that kern_nh.rs's layers call, built on the mixture-of-experts core (src/moe.rs, kernels/moe.cu).
//! The router reads the layer's normed input at the model's width; the experts work on a latent a quarter as wide (`fc1_latent_proj` down to it before them, `fc2_latent_proj` back up after), relu squared with no
//! gate; the shared expert reads the model's width and its output is added to the experts' once they are back at it. The arithmetic and its roundings are `tools/nemotron_ref.py`'s `moe_mixer` (checked against
//! transformers' own NemotronHMoE there); `NhMoe::forward` is its launches in order.
//!
//! The layer's tensors are read here, not by the executor's loader: the router and its fp32 score-correction bias as the checkpoint has them, and the latent projections, the shared expert and the experts as packs in
//! a block layout (`level`: Int6, Int5 or Int4; at Int4 the experts alone are four-bit codes and the dense matrices six-bit ones, `pack_layout`), packed as they load from the bf16 checkpoint or read from a glyd-v7
//! directory where it holds them saved, each in the layout its entry names (a bias saved there is bf16: the saver holds no fp32). The experts' second matrices (down) are in the row layout of their bits where the
//! directory has them so or the checkpoint is packed here (`pack_layout`): the layer then runs the sparse product (src/moe_rows.rs); a directory saved with them in the block layout runs the library's dense
//! one. The rest of the model is at the same level where the directory is a glyd-v7 one, else in the engine's layout.

use crate::cpu_experts::rows_of_experts;
use crate::ffi::{err, Error, Result, Words};
use crate::kern35::Mat;
use crate::kern_nh::{MoeBlock, NhHost};
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeScratch, MoeWeights, Router, Shared};
use crate::moe_rows::{Down, RowStack, RowsJoined};
use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
use crate::offload::{Call, CpuRows, ExpertStore, HostHalf, PackIn, Stage};
use crate::weights::{ordered, pack_item_map, read_pack, upload_bytes, upload_packed, v7_manifest, Item, Layout, Packed};
use glyd_gpu::json::Value;
use glyd_gpu::pack::{rows_bits, Tiered};
use glyd_gpu::safetensors::Checkpoint;
use std::rc::Rc;

/// The core's layer for a config's mixture-of-experts layers (they are alike): the router over the model's width (sigmoid scores in fp32, the experts picked by their scores plus the bias, the chosen scores
/// divided by their sum where the config says so, times `routed_scaling_factor`), experts of the latent width with relu squared and no gate.
pub fn moe_cfg(c: &ConfigNemotronH) -> MoeCfg {
    let m = &c.moe;
    MoeCfg {
        experts: m.experts,
        router_hidden: c.hidden,
        hidden: m.width(c.hidden),
        inter: m.inter,
        router: Router { renorm: m.norm_topk, ..Router::sigmoid(m.top_k, m.scale) },
        act: Act::Relu2,
    }
}

/// A router bias as the kernels read it, fp32 bytes: from an fp32 tensor as it is, from a bf16 one (a glyd-v7 directory's) widened exactly.
pub(crate) fn bias_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    match dtype {
        "F32" => Ok(bytes.to_vec()),
        "BF16" => Ok(bytes.chunks_exact(2).flat_map(|b| ((u16::from_le_bytes([b[0], b[1]]) as u32) << 16).to_le_bytes()).collect()),
        other => err(format!("the router's score-correction bias is {other}, where F32 or BF16 is wanted")),
    }
}

/// A layer's second matrices on the host, joined in the layout they are held in.
enum DownHost {
    Block(Joined),
    Rows(RowsJoined),
}

impl DownHost {
    /// The second matrices of the packs of the experts, one each: in the row layout (all of them) or a block layout.
    fn of(packs: &[Tiered]) -> Result<DownHost> {
        let all: Vec<&Tiered> = packs.iter().collect();
        if packs.first().is_some_and(|p| rows_bits(p.tiers[0] >> 24).is_some()) {
            RowsJoined::of(&all).map(DownHost::Rows)
        } else {
            Joined::of(&all).map(DownHost::Block)
        }
    }

    fn upload(&self) -> Result<Down> {
        Ok(match self {
            DownHost::Block(j) => Down::Block(ExpertStack::upload(j)?),
            DownHost::Rows(j) => Down::Rows(RowStack::upload(j)?),
        })
    }

    /// Room for `slots` of the layer's experts' second matrices, in the layout they are held in (the expert cache's places).
    fn pooled(&self, slots: usize) -> Result<Down> {
        Ok(match self {
            DownHost::Block(j) => Down::Block(ExpertStack::pooled(j.tag, j.tiers, j.experts, slots, j.rows, j.cols)?),
            DownHost::Rows(j) => Down::Rows(RowStack::pooled(j.tag, j.experts, slots, j.rows, j.cols)?),
        })
    }

    /// The bytes of every expert's codes and of its scales, end to end.
    fn bytes(&self) -> [&[u8]; 2] {
        match self {
            DownHost::Block(j) => [&j.data, &j.scales],
            DownHost::Rows(j) => [&j.data, &j.scales],
        }
    }
}

/// A layer's experts, the first matrices and the second ones of all of them joined as the kernels read them: each expert's two packs read (or packed) on as many threads as the machine has, in order. The second
/// matrices are in the layout `pack_layout` gives them (the row layout, the transpose a neuron a row) or the directory's entries name. ponytail: the host holds the layer's packs twice at the join (the real
/// model's 2.4 GB a layer, 4.8 GB at the peak); stack by stack into the device buffers if a host with less memory runs it.
fn read_experts(ck: &Checkpoint, v7: Option<&(Layout, Value)>, c: &ConfigNemotronH, l: usize, level: Layout) -> Result<(Joined, DownHost)> {
    let packs = c.expert_packs(l);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let (mut up, mut down): (Vec<Tiered>, Vec<Tiered>) = (Vec::new(), Vec::new());
    // (the packs come in the order up, down, up, down, ...: `ConfigNemotronH::expert_packs`)
    ordered(packs.len(), threads, 2 * threads, &|k| read_pack(ck, v7, &packs[k], pack_layout(level, &packs[k].key)), |k, p| {
        let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: an expert that is not in a block or row layout", packs[k].key)) };
        (if k % 2 == 0 { &mut up } else { &mut down }).push(Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers });
        Ok(())
    })?;
    Ok((Joined::of(&up.iter().collect::<Vec<_>>())?, DownHost::of(&down)?))
}

/// A layer's tensors on the host, as they are copied to the GPU: the router (bf16 bytes) and its bias (fp32 bytes), the latent projections and the shared expert's two matrices as packs, the experts joined.
struct Host {
    router: Vec<u8>,
    bias: Vec<u8>,
    /// The latent projections (none where the experts read the model's width).
    fc: Option<(Packed, Packed)>,
    s_up: Packed,
    s_down: Packed,
    experts: Experts,
}

/// A layer's experts on the host: joined packs of a block layout, or a lossless level's stacked matrices (up [experts inter, width], down [experts width, inter]) each one pack.
enum Experts {
    Block(Joined, DownHost),
    Lossless(Packed, Packed),
}

/// A lossless level's experts of layer `l`: each of the two matrices of all the experts stacked and packed as one (the packer's check: it decodes to the weights bit for bit).
fn read_experts_lossless(ck: &Checkpoint, c: &ConfigNemotronH, l: usize, level: Layout) -> Result<Experts> {
    let packs = c.expert_packs(l);
    let stack = |k: usize| -> Result<Packed> {
        let of: Vec<&crate::qwen35::Pack35> = packs.iter().skip(k).step_by(2).collect();
        let it = Item { members: of.iter().map(|p| p.parts[0].0.clone()).collect(), rows: of.iter().map(|p| p.rows()).sum(), cols: of[0].cols };
        pack_item_map(ck, &it, level, &|_| {})
    };
    Ok(Experts::Lossless(stack(0)?, stack(1)?))
}

/// Layer `l`'s tensors from `ck` at `level` (Int6, Int5 or Int4). A glyd-v7 directory is read where `ck` is one of them and at `level`: what it holds saved is read as it is, the rest packed from bf16 (in
/// `pack_layout` of the level).
fn read_layer(l: usize, c: &ConfigNemotronH, ck: &Checkpoint, level: Layout) -> Result<Host> {
    let lossless = matches!(level, Layout::Tiered | Layout::Twelve);
    if !lossless && !matches!(level, Layout::Int6 | Layout::Int5 | Layout::Int4) {
        return err(format!("the experts are held at the penguin, kestrel, swift and wren levels: {} is asked", level.name()));
    }
    if c.kinds.get(l) != Some(&NhKind::Moe) {
        return err(format!("layer {l} is not a mixture-of-experts layer"));
    }
    let v7 = v7_manifest(&ck.dir)?;
    if let Some((got, _)) = &v7 {
        if *got != level {
            return err(format!("{}: packed in the {} layout, where {} is asked", ck.dir.display(), got.name(), level.name()));
        }
    }
    let p = format!("{}layers.{l}.mixer", c.prefix);
    let tensor = |name: &str| -> Result<(String, Vec<u8>)> {
        let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        Ok((t.dtype.clone(), ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?))
    };
    let (dtype, router) = tensor(&format!("{p}.gate.weight"))?;
    if dtype != "BF16" {
        return err(format!("{p}.gate.weight is {dtype}, where the router is read as BF16"));
    }
    let bias = tensor(&format!("{p}.gate.e_score_correction_bias")).and_then(|(d, b)| bias_f32(&d, &b))?;
    // (the layer's dense matrices, in the order `layer_packs` lists them: the latent projections where it has them, the shared expert)
    let mut dense: Vec<Packed> = c.layer_packs(l).iter().map(|pk| read_pack(ck, v7.as_ref(), pk, pack_layout(level, &pk.key))).collect::<Result<_>>()?;
    let want = if c.moe.latent > 0 { 4 } else { 2 };
    if dense.len() != want {
        return err(format!("layer {l}: a mixture-of-experts layer has {want} matrices besides its experts"));
    }
    let (s_down, s_up) = (dense.pop().expect("two"), dense.pop().expect("two"));
    let fc = if want == 4 { Some((dense.remove(0), dense.remove(0))) } else { None };
    let experts = if lossless {
        if v7.is_some() {
            return err(format!("{}: a saved directory's experts at a lossless level are not read yet (pack from the Hub's model)", ck.dir.display()));
        }
        read_experts_lossless(ck, c, l, level)?
    } else {
        let (up, down) = read_experts(ck, v7.as_ref(), c, l, level)?;
        Experts::Block(up, down)
    };
    Ok(Host { router, bias, fc, s_up, s_down, experts })
}

/// A latent mixture-of-experts layer on the GPU: the core's layer (router, experts, shared expert) and the two latent projections.
pub struct NhMoe {
    layer: MoeLayer,
    fc: Option<(Mat, Mat)>,
}

/// The latent projections on the GPU.
fn upload_fc(fc: Option<(Packed, Packed)>) -> Result<Option<(Mat, Mat)>> {
    fc.map(|(a, b)| Ok((Mat::Packed(upload_packed(a)?), Mat::Packed(upload_packed(b)?)))).transpose()
}

fn fc_bytes(fc: &Option<(Mat, Mat)>) -> usize {
    fc.as_ref().map_or(0, |(a, b)| a.bytes() + b.bytes())
}

fn fc_shapes(fc: &Option<(Mat, Mat)>) -> Vec<(i64, i64, bool)> {
    fc.iter()
        .flat_map(|(a, b)| [a, b])
        .filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None })
        .collect()
}

impl NhMoe {
    /// Layer `l` of the model `c` from the checkpoint `ck`, its matrices at `level` (Int6, Int5 or Int4), copied to the GPU. A layer whose shapes the products do not take is refused (`MoeLayer::new`).
    pub fn load(l: usize, c: &ConfigNemotronH, ck: &Checkpoint, level: Layout) -> Result<NhMoe> {
        let h = read_layer(l, c, ck, level)?;
        let (up, down) = match h.experts {
            Experts::Block(up, down) => (ExpertStack::upload(&up)?, down.upload()?),
            Experts::Lossless(up, down) => {
                (ExpertStack::lossless(upload_packed(up)?, c.moe.experts)?, Down::Block(ExpertStack::lossless(upload_packed(down)?, c.moe.experts)?))
            }
        };
        let w = MoeWeights { router: upload_bytes(&h.router)?, bias: Some(upload_bytes(&h.bias)?), escale: None, up, down };
        let shared = Shared { inter: c.moe.shared_inter, up: upload_packed(h.s_up)?, down: upload_packed(h.s_down)?, gate: None };
        Ok(NhMoe { layer: MoeLayer::new(moe_cfg(c), w, Some(shared))?, fc: upload_fc(h.fc)? })
    }
}

/// The factory the executor's Nemotron-H loader takes: each mixture-of-experts layer is an `NhMoe` at `level` (Int6, Int5 or Int4), whatever the layout of the engine's other matrices.
pub fn factory(level: Layout) -> impl Fn(usize, &ConfigNemotronH, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>> {
    let common = Rc::new(crate::moe_hb::HbShared::default());
    move |l, c, ck, _| {
        if level == Layout::Hummingbird {
            return Ok(Box::new(crate::moe_hb::NhMoeHb::load(l, c, ck, None, &common)?));
        }
        Ok(Box::new(NhMoe::load(l, c, ck, level)?))
    }
}

/// The scratch of the engine's mixture-of-experts layers, which the first layer that runs makes (all the model's layers have this one's shape; the engine's warm-up step runs one before the decode graph is captured)
/// for the rows of a prompt chunk.
fn scratch<'a>(e: &'a dyn NhHost, layer: &MoeLayer) -> Result<&'a MoeScratch> {
    if e.moe_scratch().get().is_none() {
        let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[layer], e.chunk())?);
    }
    e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))
}

impl MoeBlock for NhMoe {
    /// The router on `h` (the model's width), `fc1_latent_proj` of `h` into the engine's `mix` buffer, the experts' weighted sum of it into `xc` (both are free in a layer that is not Mamba-2: the latent is the
    /// narrower), `fc2_latent_proj` of that into `out`, then the shared expert on `h` added to `out`. All launches on the engine's stream, none reading a result: a decode step captures into the graph.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let (c, s) = (&self.layer.cfg, e.dev().stream);
        let need = n * c.hidden * 2;
        if e.mix().bytes < need || e.xc().bytes < need {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.hidden));
        }
        let sc = scratch(e, &self.layer)?;
        let Some((fc1, fc2)) = &self.fc else { return self.layer.forward(e.lib(), sc, s, h, n, out) };
        self.layer.route(sc, s, h, c.router_hidden, n)?;
        e.lin(fc1, h, n, e.mix().ptr)?;
        self.layer.experts(e.lib(), sc, s, e.mix().ptr, n, e.xc().ptr)?;
        e.lin(fc2, e.xc().ptr, n, out)?;
        self.layer.add_shared(e.lib(), sc, s, h, n, out)
    }

    fn graphable(&self) -> bool {
        true
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes() + fc_bytes(&self.fc)
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        fc_shapes(&self.fc)
    }

    fn resident(&self) -> Option<(&MoeLayer, Option<(&Mat, &Mat)>)> {
        Some((&self.layer, self.fc.as_ref().map(|(a, b)| (a, b))))
    }
}

/// A latent mixture-of-experts layer whose experts are not all in the GPU's memory: the core's layer over places for `slots` of its experts (`ExpertStack::pooled`, and `RowStack::pooled` for second matrices in the row
/// layout: a place holds an expert's first matrix and its second one in the layout they are held in) and the two latent projections, the experts of the layer all in the expert cache's host tier (src/offload.rs),
/// which copies the ones the router chooses into the places, or (second matrices in the row layout, a mode but `Copy`) has the host compute the pairs of the ones that are not there. What it computes is `NhMoe`'s bit
/// for bit where every miss is copied; the host's sums differ from the GPU's by the order of an fp32 sum and by the weights it does not round to bf16 (`cpu_experts`). A decode step with it is not captured in a CUDA
/// graph: the experts the router chose are read on the host to place them.
pub struct NhMoeCached {
    layer: MoeLayer,
    fc: Option<(Mat, Mat)>,
    store: Rc<ExpertStore>,
    index: usize,
}

impl NhMoeCached {
    /// Layer `l` of the model `c` from the checkpoint `ck` at `level`: its experts into `store` (pinned host memory), places for `slots` of them on the GPU, the rest of its matrices copied to the GPU as `NhMoe::load` does.
    pub fn load(l: usize, c: &ConfigNemotronH, ck: &Checkpoint, level: Layout, store: &Rc<ExpertStore>, slots: usize) -> Result<NhMoeCached> {
        let h = read_layer(l, c, ck, level)?;
        let Experts::Block(h_up, h_down) = &h.experts else {
            return err("the expert cache's places are in block layouts: a lossless level's experts are all on the GPU");
        };
        let cfg = moe_cfg(c);
        let index = c.kinds[..l].iter().filter(|&&k| k == NhKind::Moe).count();
        let up = ExpertStack::pooled(h_up.tag, h_up.tiers, h_up.experts, slots, h_up.rows, h_up.cols)?;
        let down = h_down.pooled(slots)?;
        // (each matrix of the layer's experts: every expert's codes and scales end to end, and the places made for them)
        let (dev, stride) = down.places();
        let packs = [
            PackIn { bytes: [&h_up.data, &h_up.scales], stride: [up.data_stride, up.scale_stride], dev: up.buffers() },
            PackIn { bytes: h_down.bytes(), stride, dev },
        ];
        // the host's half where the store will use it and the second matrices are in the row layout (the host reads rows alone): the first matrices placed in rows too
        let cpu = match (h_down, store.wants_cpu()) {
            (DownHost::Rows(j), true) => {
                let (data, scales) = rows_of_experts(h_up.tag, h_up.experts, h_up.rows, h_up.cols, &h_up.data, &h_up.scales, 0)?;
                Some(HostHalf::Rows(CpuRows {
                    bits: rows_bits(j.tag).ok_or_else(|| Error(format!("a second matrix of tag {}: not a row layout", j.tag)))?,
                    inter: h_up.rows,
                    hidden: h_up.cols,
                    data,
                    scales,
                }))
            }
            _ => None,
        };
        store.add_layer(index, cfg.experts, slots, &packs, cpu)?;
        let w = MoeWeights { router: upload_bytes(&h.router)?, bias: Some(upload_bytes(&h.bias)?), escale: None, up, down };
        let shared = Shared { inter: c.moe.shared_inter, up: upload_packed(h.s_up)?, down: upload_packed(h.s_down)?, gate: None };
        Ok(NhMoeCached { layer: MoeLayer::new(cfg, w, Some(shared))?, fc: upload_fc(h.fc)?, store: store.clone(), index })
    }
}

/// The factory the executor's Nemotron-H loader takes for a model whose experts do not all fit in the GPU's memory: each mixture-of-experts layer is an `NhMoeCached` with places for `slots` experts, all of them sharing `store` (whose
/// counts and hints the caller keeps); a layer of no more experts than places is an `NhMoe`.
pub fn factory_cached(
    level: Layout,
    slots: usize,
    store: &Rc<ExpertStore>,
) -> impl Fn(usize, &ConfigNemotronH, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>> {
    let (store, common) = (store.clone(), Rc::new(crate::moe_hb::HbShared::default()));
    move |l, c, ck, _| {
        if level == Layout::Hummingbird {
            // (`slots` is the places a layer has at the mean of its experts' widths: the layer splits them between the two widths, and holds all its experts where there are as many places as experts)
            return Ok(Box::new(crate::moe_hb::NhMoeHb::load(l, c, ck, (slots < c.moe.experts).then_some((slots, &store)), &common)?));
        }
        if slots >= c.moe.experts {
            Ok(Box::new(NhMoe::load(l, c, ck, level)?))
        } else {
            Ok(Box::new(NhMoeCached::load(l, c, ck, level, &store, slots)?))
        }
    }
}

/// The block layout's tag a level holds the experts in.
fn level_tag(level: Layout) -> Result<u32> {
    match level {
        Layout::Int6 => Ok(glyd_gpu::pack::INT6_TAG),
        Layout::Int5 => Ok(glyd_gpu::pack::INT5_TAG),
        Layout::Int4 => Ok(glyd_gpu::pack::INT4_TAG),
        other => err(format!("the experts are held in the block layouts: {} is asked", other.name())),
    }
}

/// The GPU memory one expert takes at `level` (and so a place of the expert cache): the codes and the scales of its two matrices. The second matrix in its row layout takes the same bytes as in the block layout
/// (`moe_rows::row_strides` and `moe::expert_strides` give the same for the transpose: a test holds them equal), so a cache is the same size whichever layout the experts are held in.
pub fn expert_bytes(c: &ConfigNemotronH, level: Layout) -> Result<u64> {
    if level == Layout::Hummingbird {
        // (the mean of the recipe hb40 over a layer's experts, 40% at three bits and the rest at two: the directory's own bytes are what `lossy_plan` sizes the weights by)
        return crate::moe_hb::expert_bytes_mean(c);
    }
    let (tag, m) = (level_tag(level)?, &c.moe);
    let ((ud, us), (dd, ds)) = (crate::moe::expert_strides(tag, m.inter, m.width(c.hidden))?, crate::moe::expert_strides(tag, m.width(c.hidden), m.inter)?);
    Ok(ud + us + dd + ds)
}

/// The expert cache a budget buys: the planner's (src/plan.rs `sizing`).
pub use crate::plan::sizing::cache_plan;

impl MoeBlock for NhMoeCached {
    /// `NhMoe::forward`'s launches, with the experts' products over the places the router's choices were put in (`ExpertStore::run`: the choices, their weights and z are read on the host, the missing experts copied or
    /// computed there, the pairs' places sent to `MoeScratch::way` and the host's sums to `MoeScratch::cpu`); `fc1_latent_proj` first, so that it is not between the router and the host's reading of its choices. The
    /// shared expert's MLP, which needs none of the experts, is queued before them, to run while the host computes its pairs.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let (c, s) = (&self.layer.cfg, e.dev().stream);
        let need = n * c.hidden * 2;
        if e.mix().bytes < need || e.xc().bytes < need {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.hidden));
        }
        let sc = scratch(e, &self.layer)?;
        // (the experts' input and sum: the latent's, in mix and xc; the model's width, h and out, where there is no latent)
        let (z, y) = match &self.fc {
            Some((fc1, _)) => {
                e.lin(fc1, h, n, e.mix().ptr)?;
                (e.mix().ptr, e.xc().ptr)
            }
            None => (h, out),
        };
        self.layer.route(sc, s, h, c.router_hidden, n)?;
        let call = Call { ids: sc.ids.ptr, wts: sc.wts.ptr, z, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs: n * c.router.top_k };
        self.store.run(self.index, e.dev(), &call, &mut |stage, wave, waves, hy| match stage {
            Stage::Products => {
                if wave == 0 {
                    self.layer.shared_mlp(e.lib(), sc, s, h, n)?;
                }
                if hy.gpu {
                    self.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), waves)?;
                }
                Ok(())
            }
            Stage::Sum => self.layer.experts_sum(sc, s, n, y, hy.ids, hy.extra),
        })?;
        if let Some((_, fc2)) = &self.fc {
            e.lin(fc2, e.xc().ptr, n, out)?;
        }
        self.layer.shared_add(sc, s, n, out)
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes() + fc_bytes(&self.fc)
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        fc_shapes(&self.fc)
    }
}

#[cfg(test)]
mod tests {
    impl Host {
        /// A block level's experts (the tests' levels).
        fn block(&self) -> (&Joined, &DownHost) {
            match &self.experts {
                Experts::Block(u, d) => (u, d),
                Experts::Lossless(..) => panic!("a lossless level's experts"),
            }
        }
    }

    use super::*;
    use crate::kern_nh::{NH_N, NH_P};
    use crate::moe::expert_strides;
    use crate::nemotron::tests::{bf16_bytes, checkpoint};
    use crate::pack::nemotron_plan;
    use crate::qwen35::Pack35;
    use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6, pack_rows, transpose, unpack_int4, unpack_int5, unpack_int6, INT4_TAG, INT5_TAG, INT6_TAG};
    use glyd_gpu::save::{self, Source};

    /// A model of one mixture-of-experts layer (8 experts of 3 a token, a latent of 64, a model width of 128), small enough to pack in a test and with every matrix in whole blocks of the block layouts.
    const SMALL: &str = r#"{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "hidden_size": 128, "num_hidden_layers": 1, "hybrid_override_pattern": "E", "vocab_size": 64,
        "layer_norm_epsilon": 1e-5, "max_position_embeddings": 256, "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 1, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 128, "n_routed_experts": 8, "num_experts_per_tok": 3, "moe_intermediate_size": 128, "moe_shared_expert_intermediate_size": 64, "moe_latent_size": 64,
        "routed_scaling_factor": 5.0, "norm_topk_prob": true, "n_group": 1, "topk_group": 1, "n_shared_experts": 1, "mlp_hidden_act": "relu2", "tie_word_embeddings": false, "eos_token_id": 2}"#;

    fn small() -> ConfigNemotronH {
        ConfigNemotronH::parse(SMALL, "small", None).unwrap()
    }

    /// The layout's tag in a pack's first word.
    fn tag_of(p: &Packed) -> u32 {
        match p.words {
            Words::Tiered { tiers, .. } => tiers[0] >> 24,
            Words::Twelve { .. } => 0,
        }
    }

    /// What reading a layer says: its error, or that it loaded.
    fn says(r: Result<Host>) -> String {
        match r {
            Ok(_) => "loaded".to_string(),
            Err(e) => e.msg,
        }
    }

    /// A model's numbers for the core's layer are the config's: the pinned checkpoint's (22 of 512, scale 5, a latent of 1,024) are the layer moe.rs holds for Nemotron 3 Super, and a config that does not
    /// renormalize says so.
    #[test]
    fn the_layer_of_the_super_config() {
        let c = crate::nemotron::tests::parse();
        let cfg = moe_cfg(&c);
        assert_eq!(cfg, MoeCfg { experts: 512, router_hidden: 4096, hidden: 1024, inter: 2688, router: Router::sigmoid(22, 5.0), act: Act::Relu2 });
        cfg.check().unwrap();
        assert!(!moe_cfg(&ConfigNemotronH { moe: crate::nemotron::MoeDims { norm_topk: false, ..c.moe.clone() }, ..c }).router.renorm);
    }

    /// An expert is 3,784,704 bytes at swift, 4,472,832 at kestrel and 3,096,576 at wren (the codes and scales of its two matrices), and a budget buys places a layer of 512 by them: at swift 14.95, 22.99 and 34 GiB are
    /// 106, 163 and 241; fewer than the choices of a step (22) are none.
    #[test]
    fn an_expert_is_a_few_megabytes_and_a_budget_buys_places() {
        let c = crate::nemotron::tests::parse();
        assert_eq!(
            (expert_bytes(&c, Layout::Int5).unwrap(), expert_bytes(&c, Layout::Int6).unwrap(), expert_bytes(&c, Layout::Int4).unwrap()),
            (3_784_704, 4_472_832, 3_096_576)
        );
        assert!(expert_bytes(&c, Layout::Twelve).is_err() && expert_bytes(&c, Layout::Int3).is_err());
        let gib = |g: f64| (g * (1u64 << 30) as f64) as u64;
        for (cap, places) in [(14.95, 106), (22.99, 163), (34.0, 241)] {
            let plan = cache_plan(&c, Layout::Int5, gib(cap), 22).unwrap().unwrap();
            assert_eq!((plan.slots, plan.layers, plan.experts), (places, 40, 512), "{cap} GiB");
        }
        assert!(cache_plan(&c, Layout::Int5, gib(3.0), 22).unwrap().is_none(), "21 experts a layer are fewer than a step's choices");
    }

    /// An expert's second matrix in a row layout takes the bytes it takes in the block layout, at the real shape and at the three levels: 2,688 neurons of 512, 640 or 768 bytes of codes and 64 of scales, against the
    /// block layout's 1,024 rows of 2,688: the expert cache's places and its plan are the same size for either.
    #[test]
    fn a_second_matrix_in_a_row_layout_takes_the_bytes_of_the_block_layout() {
        use crate::moe_rows::row_strides;
        use glyd_gpu::pack::{INT4_ROW_TAG, INT5_ROW_TAG, INT6_ROW_TAG};
        let c = crate::nemotron::tests::parse();
        for (block, row, per_neuron) in [(INT6_TAG, INT6_ROW_TAG, 768u64), (INT5_TAG, INT5_ROW_TAG, 640), (INT4_TAG, INT4_ROW_TAG, 512)] {
            let (rows, blocks) = (row_strides(row, c.moe.inter, c.moe.latent).unwrap(), expert_strides(block, c.moe.latent, c.moe.inter).unwrap());
            assert_eq!(rows, blocks, "tag {block}");
            assert_eq!(rows, (c.moe.inter as u64 * per_neuron, c.moe.inter as u64 * 64), "tag {block}");
        }
    }

    /// The layer's matrices come in the order the loader takes them, and the experts' in the order it joins them.
    #[test]
    fn the_layers_matrices_in_the_order_they_are_read() {
        let c = crate::nemotron::tests::parse();
        let keys: Vec<(String, usize, usize)> =
            c.layer_packs(1).into_iter().map(|p| (p.key.rsplit("mixer.").next().unwrap().to_string(), p.rows(), p.cols)).collect();
        assert_eq!(
            keys,
            [
                ("fc1_latent_proj".to_string(), 1024, 4096),
                ("fc2_latent_proj".to_string(), 4096, 1024),
                ("shared_experts.up_proj".to_string(), 5376, 4096),
                ("shared_experts.down_proj".to_string(), 4096, 5376)
            ]
        );
        let experts = c.expert_packs(1);
        assert_eq!(experts.len(), 1024);
        assert!(
            experts[0].key.ends_with("experts.0.up_proj")
                && experts[1].key.ends_with("experts.0.down_proj")
                && experts[1023].key.ends_with("experts.511.down_proj")
        );
        assert_eq!((experts[0].rows(), experts[0].cols, experts[1].rows(), experts[1].cols), (2688, 1024, 1024, 2688));
    }

    /// The tiny model `tools/nemotron_ref.py moetoy` builds (its config is the fixture the script's selftest holds it to) is a model this engine runs: the numbers of its kernels, and every matrix of its
    /// mixture-of-experts layers in whole blocks of both block layouts.
    #[test]
    fn the_random_tiny_model_is_one_the_engine_takes() {
        let c = ConfigNemotronH::parse(include_str!("../tests/fixtures/nemotron_h_moetoy_config.json"), "moetoy", None).unwrap();
        assert_eq!(c.kinds, [NhKind::Mamba, NhKind::Moe, NhKind::Attention, NhKind::Moe, NhKind::Mamba, NhKind::Moe]);
        assert!(
            c.m_head_dim == NH_P
                && c.m_state == NH_N
                && c.conv_kernel == 4
                && c.head_dim == 128
                && c.inner() / c.m_groups <= 1024
                && c.heads / c.kv_heads <= 16
                && c.hidden <= 8192
        );
        assert_eq!(
            (c.moe.experts, c.moe.top_k, c.moe.latent, c.moe.inter, c.moe.shared_inter, c.moe.scale, c.moe.norm_topk),
            (16, 4, 128, 128, 256, 5.0, true)
        );
        moe_cfg(&c).check().unwrap();
        for l in 0..c.layers {
            assert!(c.layer_packs(l).iter().all(Pack35::packable), "layer {l}'s matrices are packed");
        }
        for p in c.layer_packs(1).into_iter().chain(c.expert_packs(1)) {
            for tag in [INT6_TAG, INT5_TAG] {
                assert!(expert_strides(tag, p.rows(), p.cols).is_ok(), "{} is {} x {}", p.key, p.rows(), p.cols);
            }
        }
        assert_eq!((c.vocab, c.hidden), (2048, 256));
    }

    /// What `moetoy` relies on of the packers (all three block layouts of a mixture-of-experts layer): weights that are integers of at most the layout's largest code times a power of two, with the largest code in every
    /// run of 32, come back exactly (the fit of the scale lands on the power of two), so the engine's packs hold the tiny model's experts as the script's reference has them.
    #[test]
    fn weights_on_the_grid_are_held_exactly() {
        let (rows, cols) = (128usize, 96usize);
        let mut x = 7u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
        type Unpack = fn(&Tiered, &mut [u16]) -> bool;
        for (kmax, pack, unpack, power) in [
            (31i32, pack_int6 as Pack, unpack_int6 as Unpack, -8),
            (15, pack_int5 as Pack, unpack_int5 as Unpack, -7),
            (7, pack_int4 as Pack, unpack_int4 as Unpack, -6),
        ] {
            let w: Vec<u16> = (0..rows * cols)
                .map(|i| {
                    let k = if i % 32 == 0 {
                        if next() & 1 == 0 {
                            kmax
                        } else {
                            -kmax
                        }
                    } else {
                        (next() % (2 * kmax as u64 + 1)) as i32 - kmax
                    };
                    let v = k as f32 * 2f32.powi(power);
                    (v.to_bits() >> 16) as u16 // (a code of at most 5 bits times a power of two: bf16 holds it)
                })
                .collect();
            let p = pack(&w, rows, cols).unwrap();
            let mut back = vec![0u16; rows * cols];
            assert!(unpack(&p, &mut back));
            assert!(back == w, "kmax {kmax}: the grid's weights did not come back");
        }
    }

    #[test]
    fn a_bias_is_read_as_fp32_from_fp32_or_bf16() {
        let w = [0x3f80u16, 0xbf00, 0x3dcd];
        let widened: Vec<u8> = w.iter().flat_map(|&x| ((x as u32) << 16).to_le_bytes()).collect();
        assert_eq!(bias_f32("F32", &widened).unwrap(), widened);
        assert_eq!(bias_f32("BF16", &bf16_bytes(&w)).unwrap(), widened);
        assert!(bias_f32("F16", &[0; 4]).unwrap_err().msg.contains("F32 or BF16"));
    }

    /// A down host's layout tag, its experts, rows and columns, and its bytes.
    fn down_parts(d: &DownHost) -> (u32, usize, usize, usize, &[u8], &[u8]) {
        match d {
            DownHost::Block(j) => (j.tag, j.experts, j.rows, j.cols, &j.data, &j.scales),
            DownHost::Rows(j) => (j.tag, j.experts, j.rows, j.cols, &j.data, &j.scales),
        }
    }

    /// A layer is read from the bf16 checkpoint as the packers make it (each expert's first matrix packed alone, joined: the stacked pack of the packer; each second matrix, in the row layout, as its transpose packed
    /// alone, joined), at the three levels (at wren the experts four-bit and the dense matrices six-bit); the same layer saved as a glyd-v7 directory (`nemotron_plan`, the saver's own packs) is read to the same bytes,
    /// its bias (saved as bf16) widened; a directory saved with the second matrices in the block layout (the plan without their row layouts: a directory of before) is read in it, as the stacked pack of the packer;
    /// and what is asked that the directory cannot give is refused.
    #[test]
    fn a_layer_is_read_from_bf16_and_from_a_saved_directory_alike() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-moe-nh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = small();
        let src = dir.join("src");
        let w = checkpoint(&src, SMALL, &c);
        let ck = Checkpoint::open(&src).unwrap();
        let p = "backbone.layers.0.mixer";
        let (e, lat, inter, hid, sh) = (8usize, 64usize, 128usize, 128usize, 64usize);
        let stacked = |name: &dyn Fn(usize) -> String| -> Vec<u16> { (0..e).flat_map(|i| w[&name(i)].clone()).collect() };
        let (up_w, down_w) = (stacked(&|i| format!("{p}.experts.{i}.up_proj.weight")), stacked(&|i| format!("{p}.experts.{i}.down_proj.weight")));
        type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
        for (level, pack, dense_pack, tags) in [
            (Layout::Int6, pack_int6 as Pack, pack_int6 as Pack, (INT6_TAG, INT6_TAG)),
            (Layout::Int5, pack_int5 as Pack, pack_int5 as Pack, (INT5_TAG, INT5_TAG)),
            (Layout::Int4, pack_int4 as Pack, pack_int6 as Pack, (INT4_TAG, INT6_TAG)),
        ] {
            let h = read_layer(0, &c, &ck, level).unwrap();
            // the experts: the stacked pack of the packer, the codes and the scales; the second matrices as the row layout of each one's transpose
            let bits = level.row_of().unwrap().row_bits().unwrap();
            let up = pack(&up_w, e * inter, lat).unwrap();
            let rows_packs: Vec<Tiered> =
                (0..e).map(|i| pack_rows(bits, &transpose(&down_w[i * lat * inter..(i + 1) * lat * inter], lat, inter), inter, lat).unwrap()).collect();
            let want_down = RowsJoined::of(&rows_packs.iter().collect::<Vec<_>>()).unwrap();
            assert_eq!((h.block().0.tag, tag_of(&h.fc.as_ref().unwrap().0), tag_of(&h.s_up)), (tags.0, tags.1, tags.1), "{level:?}: the layouts");
            assert!(
                h.block().0.data == up.data && h.block().0.scales == up.blocks && (h.block().0.experts, h.block().0.rows, h.block().0.cols) == (e, inter, lat),
                "{level:?}: the first matrices"
            );
            assert!(matches!(h.block().1, DownHost::Rows(_)), "{level:?}: the second matrices are in the row layout");
            assert_eq!(down_parts(h.block().1), (want_down.tag, e, inter, lat, &want_down.data[..], &want_down.scales[..]), "{level:?}: the second matrices");
            // the dense matrices and the router
            for (got, name, rows, cols) in [
                (&h.fc.as_ref().unwrap().0, "fc1_latent_proj", lat, hid),
                (&h.fc.as_ref().unwrap().1, "fc2_latent_proj", hid, lat),
                (&h.s_up, "shared_experts.up_proj", sh, hid),
                (&h.s_down, "shared_experts.down_proj", hid, sh),
            ] {
                let want = dense_pack(&w[&format!("{p}.{name}.weight")], rows, cols).unwrap();
                assert!(got.data == want.data && got.b1 == want.blocks && (got.rows, got.cols) == (rows, cols), "{level:?}: {name}");
            }
            assert_eq!(h.router, bf16_bytes(&w[&format!("{p}.gate.weight")]));
            assert_eq!(h.bias, w[&format!("{p}.gate.e_score_correction_bias")].iter().flat_map(|&x| ((x as u32) << 16).to_le_bytes()).collect::<Vec<u8>>());

            // saved as a glyd-v7 directory, read back: the same bytes (the bias, rounded to bf16 by the saver, is a bf16 value already here)
            let out = dir.join(level.name());
            let have = &|n: &str| ck.get(n).is_some_and(|(_, t)| t.dtype == "F32");
            save::save_planned(&Source::find(src.to_str().unwrap()).unwrap(), &out, 2, 5 << 30, level, &nemotron_plan(&c, level, have)).unwrap();
            let saved = Checkpoint::open(&out).unwrap();
            assert!(saved.get(&format!("{p}.experts.7.down_proj.glyd_data")).is_some() && saved.get(&format!("{p}.experts.7.down_proj.weight")).is_none());
            let s = read_layer(0, &c, &saved, level).unwrap();
            assert!(
                s.block().0.data == h.block().0.data && s.block().0.scales == h.block().0.scales && down_parts(s.block().1) == down_parts(h.block().1),
                "{level:?}: the experts of the saved directory"
            );
            for (a, b) in [
                (&s.fc.as_ref().unwrap().0, &h.fc.as_ref().unwrap().0),
                (&s.fc.as_ref().unwrap().1, &h.fc.as_ref().unwrap().1),
                (&s.s_up, &h.s_up),
                (&s.s_down, &h.s_down),
            ] {
                assert!(a.data == b.data && a.b1 == b.b1 && (a.rows, a.cols) == (b.rows, b.cols), "{level:?}: a dense matrix of the saved directory");
            }
            assert_eq!((&s.router, &s.bias), (&h.router, &h.bias));
            assert_eq!((s.block().0.tag, tag_of(&s.fc.as_ref().unwrap().0)), (tags.0, tags.1), "{level:?}: the saved directory's layouts");
            // a directory saved before the row layouts (the second matrices in the level's block layout): read in it, as the stacked pack of the packer
            let plan = nemotron_plan(&c, level, have);
            let old_plan = plan.replace(&format!(", \"layout\": \"{}\"", level.row_of().unwrap().name()), "");
            assert!(plan.contains(&format!("\"layout\": \"{}\"", level.row_of().unwrap().name())) && old_plan != plan);
            let before = dir.join(format!("{}-before", level.name()));
            save::save_planned(&Source::find(src.to_str().unwrap()).unwrap(), &before, 2, 5 << 30, level, &old_plan).unwrap();
            let b = read_layer(0, &c, &Checkpoint::open(&before).unwrap(), level).unwrap();
            let down = pack(&down_w, e * lat, inter).unwrap();
            assert!(matches!(b.block().1, DownHost::Block(_)), "{level:?}: the second matrices of a directory of before are in the block layout");
            assert_eq!(
                down_parts(b.block().1),
                (tags.0, e, lat, inter, &down.data[..], &down.blocks[..]),
                "{level:?}: the second matrices of a directory of before"
            );
            assert!(b.block().0.data == h.block().0.data, "{level:?}: and the first");
            // a directory in the other level, and the levels there are none of
            let other = if level == Layout::Int6 { Layout::Int5 } else { Layout::Int6 };
            assert!(says(read_layer(0, &c, &saved, other)).contains("packed in"));
        }
        // (the lossless levels: the experts' stacked matrices packed whole)
        for level in [Layout::Tiered, Layout::Twelve] {
            assert_eq!(says(read_layer(0, &c, &ck, level)), "loaded", "{level:?}");
        }
        assert!(says(read_layer(0, &c, &ck, Layout::Int3)).contains("penguin, kestrel, swift and wren levels"));
        let mamba = ConfigNemotronH { kinds: vec![NhKind::Mamba], ..c.clone() };
        assert!(says(read_layer(0, &mamba, &ck, Layout::Int6)).contains("not a mixture-of-experts layer"));
        assert!(says(read_layer(1, &c, &ck, Layout::Int6)).contains("not a mixture-of-experts layer"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
