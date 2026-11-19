//! GLM-5.3-Flash's mixture-of-experts layer on the GPU (`Glm5NextTextMoE`): `GlmMoe`, the `MoeBlock` that model_glm.rs's layers call, built on the mixture-of-experts core (src/moe.rs, kernels/moe.cu) as it stands: the layer of
//! the core's comments, with the router, the experts and the shared expert all reading the layer's normed input at the model's width (`MoeLayer::forward`). The router scores the 288 experts in fp32 (sigmoid scores; the 8
//! picked by their scores plus the fp32 `e_score_correction_bias`; the chosen scores divided by their sum and times `routed_scaling_factor`, 2.5), each expert is a SwiGLU of 2,048 with the gate at most 10 and the up within
//! 10 of zero, and the shared expert, the same on the same input, is added to their weighted sum. The arithmetic and its roundings are `tools/glm_ref.py`'s `moe_mixer` (the structure checked against transformers'
//! Glm5NextTextMoE there); `GlmMoe::forward` is the core's launches in order.
//!
//! The layer's tensors are read here, not by `WeightsGlm::load`: the router and its fp32 bias as the checkpoint has them, the shared expert's two matrices and the experts as packs in a block layout (`level`: Int6, Int5 or
//! Int4; the experts have no other, `ExpertStack`), packed as they load from the checkpoint's FP8 (as the bf16 it dequantises to) or bf16 or read from a glyd-v7 directory where it holds them saved. The rest of the model is at the
//! same level where the directory is a glyd-v7 one, else in the engine's layout.

use crate::dev::Dev;
use crate::f8::{F8Experts, F8Mat};
use crate::ffi::{err, Error, Lib, Result, Words};
use crate::glm::ConfigGlm;
use crate::model_glm::{f32_bytes, EngineGlm, MoeBlock};
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeScratch, MoeWeights, Router, Shared};
use crate::moe_rows::Down;
use crate::weights::{ordered, pack_item_map, read_pack, upload_bytes, upload_packed, v7_manifest, Item, Layout, PackBuf, Packed};
use glyd_gpu::json::Value;
use glyd_gpu::pack::Tiered;
use glyd_gpu::safetensors::Checkpoint;

/// The core's layer for a config's mixture-of-experts layers (they are alike): the router and the experts over the model's width (sigmoid scores in fp32, the experts picked by their scores plus the bias, the chosen
/// scores divided by their sum where the config says so, times `routed_scaling_factor`), SwiGLU with the config's clamp.
pub fn moe_cfg(c: &ConfigGlm) -> MoeCfg {
    let m = &c.experts;
    MoeCfg { experts: m.experts, router_hidden: c.hidden, hidden: c.hidden, inter: m.inter, router: Router { renorm: m.norm_topk, ..Router::sigmoid(m.top_k, m.scale) }, act: Act::SwiGlu { limit: Some(m.limit) } }
}

/// A layer's experts, the first matrices (gate and up as one) and the second ones (down) of all of them joined as the library's grouped products read them: each expert's two packs read (or packed) on as many threads as
/// the machine has, in order. ponytail: the host holds the layer's packs twice at the join (the real model's 5.9 GB a layer at Int6, 11.8 GB at the peak); stack by stack into the device buffers if a host with less memory
/// runs it.
fn read_experts(ck: &Checkpoint, v7: Option<&(Layout, Value)>, c: &ConfigGlm, l: usize, level: Layout) -> Result<(Joined, Joined)> {
    let packs = c.expert_packs(l);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let (mut up, mut down): (Vec<Tiered>, Vec<Tiered>) = (Vec::new(), Vec::new());
    // (the packs come in the order gate and up, down, gate and up, down, ...: `ConfigGlm::expert_packs`)
    ordered(packs.len(), threads, 2 * threads, &|k| read_pack(ck, v7, &packs[k], level), |k, p| {
        let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: an expert that is not in a block layout", packs[k].key)) };
        (if k % 2 == 0 { &mut up } else { &mut down }).push(Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers });
        Ok(())
    })?;
    Ok((Joined::of(&up.iter().collect::<Vec<_>>())?, Joined::of(&down.iter().collect::<Vec<_>>())?))
}

/// A layer's tensors on the host, as they are copied to the GPU: the router (bf16 bytes) and its bias (fp32 bytes), the shared expert's two matrices as packs, the experts (joined block packs, or a
/// lossless level's two stacked matrices).
struct Host {
    router: Vec<u8>,
    bias: Vec<u8>,
    s_up: Packed,
    s_down: Packed,
    experts: Experts,
}

enum Experts {
    Block(Joined, Joined),
    Lossless(Packed, Packed),
    /// read on the GPU in the FP8 format (`read_experts_f8`)
    F8,
}

#[cfg(test)]
impl Host {
    /// A block level's experts (a test's: the lossless level's are packs).
    fn block(&self) -> (&Joined, &Joined) {
        match &self.experts {
            Experts::Block(u, d) => (u, d),
            Experts::Lossless(..) | Experts::F8 => panic!("the lossless level's experts"),
        }
    }
}

/// A lossless level's experts of layer `l`: each of the two matrices of all the experts stacked and packed as one (the packer's check: it decodes to the weights bit for bit; an FP8 checkpoint's as the bf16 it
/// dequantises to).
fn read_experts_lossless(ck: &Checkpoint, c: &ConfigGlm, l: usize, level: Layout) -> Result<Experts> {
    let packs = c.expert_packs(l);
    let stack = |k: usize| -> Result<Packed> {
        let of: Vec<&crate::qwen35::Pack35> = packs.iter().skip(k).step_by(2).collect();
        let it = Item { members: of.iter().flat_map(|p| p.parts.iter().map(|(n, _)| n.clone())).collect(), rows: of.iter().map(|p| p.rows()).sum(), cols: of[0].cols };
        pack_item_map(ck, &it, level, &|_| {})
    };
    Ok(Experts::Lossless(stack(0)?, stack(1)?))
}

/// Whether layer `l`'s experts are FP8 as `read_experts_f8` reads them: E4M3 with 128 x 128 block scales, the intermediate size and the width multiples of 128.
fn f8_experts(ck: &Checkpoint, c: &ConfigGlm, l: usize) -> bool {
    let name = format!("{}layers.{l}.mlp.experts.0.gate_proj.weight", c.prefix);
    c.fp8_block == Some([128, 128]) && c.experts.inter.is_multiple_of(128) && c.hidden.is_multiple_of(128) && ck.get(&name).is_some_and(|(_, t)| t.dtype == "F8_E4M3")
}

/// Whether layer `l`'s shared expert is FP8 (`read_shared_f8` reads it so).
fn f8_shared(ck: &Checkpoint, c: &ConfigGlm, l: usize) -> bool {
    let name = format!("{}layers.{l}.mlp.shared_experts.gate_proj.weight", c.prefix);
    c.experts.shared_inter.is_multiple_of(128) && ck.get(&name).is_some_and(|(_, t)| t.dtype == "F8_E4M3")
}

/// This GPU's rows of an FP8 matrix (an expert's) of `rows`, where it is a GPU of a group (src/dynamic/tp.rs): its slice in whole blocks of 128 rows (the
/// scales'); all of them where it is a GPU on its own.
fn tp_rows(rows: usize) -> Result<(usize, usize)> {
    match crate::dynamic::tp::slicer() {
        None => Ok((0, rows)),
        Some(s) if rows % (128 * s.n) == 0 => Ok(s.rows(rows)),
        Some(s) => err(format!("an FP8 matrix of {rows} rows across {} GPUs: each GPU's rows whole blocks of 128 are wanted", s.n)),
    }
}

/// A lossless level's experts of layer `l` in the FP8 format, as the checkpoint has them: each expert's gate and up (its first matrix, [2 inter, hidden]) stacked, and its down, each stack packed as one on the GPU
/// (the bytes uploaded, packed, freed; where the build is a development one, unpacked and compared byte for byte).
fn read_experts_f8(ck: &Checkpoint, c: &ConfigGlm, l: usize, dev: &Dev, lib: &Lib) -> Result<F8Experts> {
    let (e, inter, hid) = (c.experts.experts, c.experts.inter, c.hidden);
    let p = format!("{}layers.{l}.mlp.experts", c.prefix);
    let stack = |names: &[&str], rows: usize, cols: usize| -> Result<F8Mat> {
        let (mut q, mut sc) = (Vec::with_capacity(e * names.len() * rows * cols), Vec::with_capacity(e * names.len() * rows / 128 * cols / 128));
        for x in 0..e {
            let (mut eq, mut es) = (Vec::new(), Vec::new());
            for n in names {
                let w = format!("{p}.{x}.{n}.weight");
                let (_, t) = ck.get(&w).ok_or_else(|| Error(format!("no tensor {w}")))?;
                if t.dtype != "F8_E4M3" || t.shape != [rows as u64, cols as u64] {
                    return err(format!("{w} is {} {:?}, where F8_E4M3 [{rows}, {cols}] is read", t.dtype, t.shape));
                }
                eq.extend(ck.read(&w).map_err(|er| Error(format!("{w}: {er}")))?);
                es.extend(f32_bytes(ck, &format!("{w}_scale_inv"))?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
            }
            // (a GPU of a group: its slice of the expert's rows, whole blocks of 128 for the scales, src/dynamic/tp.rs)
            let (r0, rn) = tp_rows(names.len() * rows)?;
            let gc = cols.div_ceil(128);
            q.extend_from_slice(&eq[r0 * cols..(r0 + rn) * cols]);
            sc.extend_from_slice(&es[r0 / 128 * gc..(r0 + rn) / 128 * gc]);
        }
        let all = names.len() * rows;
        let mut m = F8Mat::pack(dev, lib, &q, &sc, e * tp_rows(all)?.1, cols, cfg!(feature = "dev"))?;
        if tp_rows(all)?.1 < all {
            m.tp_rows = all;
        }
        Ok(m)
    };
    Ok(F8Experts { up: stack(&["gate_proj", "up_proj"], inter, hid)?, down: stack(&["down_proj"], hid, inter)?, experts: e })
}

/// Layer `l`'s shared expert in the FP8 format as the checkpoint has it: its gate and up stacked (its first matrix), its down.
fn read_shared_f8(ck: &Checkpoint, c: &ConfigGlm, l: usize, dev: &Dev, lib: &Lib) -> Result<(F8Mat, F8Mat)> {
    let (si, hid) = (c.experts.shared_inter, c.hidden);
    let p = format!("{}layers.{l}.mlp.shared_experts", c.prefix);
    let stack = |names: &[&str], rows: usize, cols: usize| -> Result<F8Mat> {
        let (mut q, mut sc) = (Vec::new(), Vec::new());
        for n in names {
            let w = format!("{p}.{n}.weight");
            let (_, t) = ck.get(&w).ok_or_else(|| Error(format!("no tensor {w}")))?;
            if t.dtype != "F8_E4M3" || t.shape != [rows as u64, cols as u64] {
                return err(format!("{w} is {} {:?}, where F8_E4M3 [{rows}, {cols}] is read", t.dtype, t.shape));
            }
            q.extend(ck.read(&w).map_err(|er| Error(format!("{w}: {er}")))?);
            sc.extend(f32_bytes(ck, &format!("{w}_scale_inv"))?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        }
        // (a GPU of a group: its slice of the rows, src/dynamic/tp.rs)
        let all = names.len() * rows;
        let (r0, rn) = tp_rows(all)?;
        let gc = cols.div_ceil(128);
        let mut m = F8Mat::pack(dev, lib, &q[r0 * cols..(r0 + rn) * cols], &sc[r0 / 128 * gc..(r0 + rn) / 128 * gc], rn, cols, cfg!(feature = "dev"))?;
        if rn < all {
            m.tp_rows = all;
        }
        Ok(m)
    };
    Ok((stack(&["gate_proj", "up_proj"], si, hid)?, stack(&["down_proj"], hid, si)?))
}

/// Layer `l`'s tensors from `ck` at `level` (Int6, Int5 or Int4). A glyd-v7 directory is read where `ck` is one of them and at `level`: what it holds saved is read as it is, the rest packed (from FP8 or bf16).
fn read_layer(l: usize, c: &ConfigGlm, ck: &Checkpoint, level: Layout) -> Result<Host> {
    read_layer_with(l, c, ck, level, false)
}

/// `read_layer`, the experts left to `read_experts_f8` where `f8`.
fn read_layer_with(l: usize, c: &ConfigGlm, ck: &Checkpoint, level: Layout, f8: bool) -> Result<Host> {
    let lossless = matches!(level, Layout::Tiered | Layout::Twelve);
    if !lossless && !matches!(level, Layout::Int6 | Layout::Int5 | Layout::Int4) {
        return err(format!("the experts are held at the penguin, kestrel, swift and wren levels: {level:?} is asked"));
    }
    if !c.moe_of(l) {
        return err(format!("layer {l} is not a mixture-of-experts layer"));
    }
    let v7 = v7_manifest(&ck.dir)?;
    if let Some((got, _)) = &v7 {
        if *got != level {
            return err(format!("{}: packed in the {got:?} layout, where {level:?} is asked", ck.dir.display()));
        }
    }
    let p = format!("{}layers.{l}.mlp", c.prefix);
    let name = format!("{p}.gate.weight");
    let (_, info) = ck.get(&name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    if info.dtype != "BF16" {
        return err(format!("{name} is {}, where the router is read as BF16", info.dtype));
    }
    let router = ck.read(&name).map_err(|e| Error(format!("{name}: {e}")))?;
    let bias = f32_bytes(ck, &format!("{p}.gate.e_score_correction_bias"))?;
    // (the layer's matrices besides the experts: the shared expert's gate and up as one, its down)
    let packs = c.layer_packs(l);
    let [s_up, s_down] = &packs[packs.len() - 2..] else { return err(format!("layer {l}: a mixture-of-experts layer has two matrices besides its experts and its attention's")) };
    // (with the FP8 experts, the shared expert's FP8 matrices are read on the GPU too: their shapes alone here)
    let shape = |p: &crate::qwen35::Pack35| Packed { data: Vec::new(), b1: Vec::new(), b2: Vec::new(), words: Words::Tiered { blocks: 0, block_base: 0, tiers: [0; 3] }, rows: p.rows(), cols: p.cols };
    let (s_up, s_down) = if f8 && f8_shared(ck, c, l) { (shape(s_up), shape(s_down)) } else { (read_pack(ck, v7.as_ref(), s_up, level)?, read_pack(ck, v7.as_ref(), s_down, level)?) };
    let experts = if f8 {
        Experts::F8
    } else if lossless {
        read_experts_lossless(ck, c, l, level)?
    } else {
        let (up, down) = read_experts(ck, v7.as_ref(), c, l, level)?;
        Experts::Block(up, down)
    };
    Ok(Host { router, bias, s_up, s_down, experts })
}

/// A mixture-of-experts layer on the GPU: the core's layer (router, experts, shared expert).
pub struct GlmMoe {
    layer: MoeLayer,
}

impl GlmMoe {
    /// Layer `l` of the model `c` from the checkpoint `ck`, its matrices at `level` (Int6, Int5 or Int4), copied to the GPU. A layer whose shapes the products do not take is refused (`MoeLayer::new`).
    /// `f8`: at the lossless level, FP8 experts are held in the FP8 format as released (on that device, by that library), not as the bf16 they dequantise to.
    pub fn load(l: usize, c: &ConfigGlm, ck: &Checkpoint, level: Layout, f8: Option<(&Dev, &Lib)>) -> Result<GlmMoe> {
        let f8 = f8.filter(|_| matches!(level, Layout::Tiered | Layout::Twelve) && f8_experts(ck, c, l));
        let h = read_layer_with(l, c, ck, level, f8.is_some())?;
        let (router, bias) = (upload_bytes(&h.router)?, Some(upload_bytes(&h.bias)?));
        let (up, down) = match h.experts {
            Experts::Block(u, d) => (ExpertStack::upload(&u)?, ExpertStack::upload(&d)?),
            Experts::Lossless(u, d) => (ExpertStack::lossless(upload_packed(u)?, c.experts.experts)?, ExpertStack::lossless(upload_packed(d)?, c.experts.experts)?),
            Experts::F8 => {
                let (dev, lib) = f8.expect("FP8 experts are read with a device");
                let experts = read_experts_f8(ck, c, l, dev, lib)?;
                if !f8_shared(ck, c, l) {
                    let shared = Shared { inter: c.experts.shared_inter, up: upload_packed(h.s_up)?, down: upload_packed(h.s_down)?, gate: None };
                    return Ok(GlmMoe { layer: MoeLayer::new_f8(moe_cfg(c), router, bias, experts, Some(shared))? });
                }
                let shared = Shared { inter: c.experts.shared_inter, up: PackBuf::shape_only(h.s_up.rows, h.s_up.cols)?, down: PackBuf::shape_only(h.s_down.rows, h.s_down.cols)?, gate: None };
                let (su, sd) = read_shared_f8(ck, c, l, dev, lib)?;
                return Ok(GlmMoe { layer: MoeLayer::new_f8(moe_cfg(c), router, bias, experts, Some(shared))?.with_shared8(su, sd)? });
            }
        };
        let shared = Shared { inter: c.experts.shared_inter, up: upload_packed(h.s_up)?, down: upload_packed(h.s_down)?, gate: None };
        let w = MoeWeights { router, bias, escale: None, up, down: Down::Block(down) };
        Ok(GlmMoe { layer: MoeLayer::new(moe_cfg(c), w, Some(shared))? })
    }
}

/// The factory `EngineGlm::new` takes: each mixture-of-experts layer is a `GlmMoe` at `level` (Int6, Int5 or Int4), whatever the layout of the engine's other matrices.
pub fn factory(level: Layout) -> impl Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>> {
    move |l, c, ck, _| Ok(Box::new(GlmMoe::load(l, c, ck, level, None)?))
}

/// `factory` whose lossless level holds FP8 experts in the FP8 format (`GlmMoe::load`'s `f8`; GLYD_GLM_F8=0: as the bf16 they dequantise to).
pub fn factory_f8<'a>(level: Layout, dev: &'a Dev, lib: &'a Lib) -> impl Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>> + 'a {
    let on = crate::tune!("GLYD_GLM_F8").as_deref() != Some("0");
    move |l, c, ck, _| Ok(Box::new(GlmMoe::load(l, c, ck, level, on.then_some((dev, lib)))?))
}

/// The scratch of the engine's mixture-of-experts layers, which the first layer that runs makes (all the model's layers have this one's shape; the engine's warm-up step runs one before the decode graph is captured) for
/// the rows of a prompt chunk.
fn scratch<'a>(e: &'a EngineGlm, layer: &MoeLayer) -> Result<&'a MoeScratch> {
    if e.moe_scratch.get().is_none() {
        let _ = e.moe_scratch.set(MoeScratch::new(&e.dev, &e.lib, &[layer], e.chunk())?);
    }
    e.moe_scratch.get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))
}

impl MoeBlock for GlmMoe {
    /// The router on `h`, the experts' weighted sum of it into `out`, then the shared expert on `h` added to `out`: `MoeLayer::forward`. All launches on the engine's stream, none reading a result: a decode step captures
    /// into the graph.
    unsafe fn forward(&self, e: &EngineGlm, h: u64, n: usize, out: u64) -> Result<()> {
        let sc = scratch(e, &self.layer)?;
        self.layer.forward(&e.lib, sc, e.dev.stream, h, n, out)
    }

    fn graphable(&self) -> bool {
        true
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes()
    }

    fn layer(&self) -> Option<&MoeLayer> {
        Some(&self.layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::GlmKind;
    use crate::moe::expert_strides;
    use glyd_gpu::pack::{INT4_TAG, INT5_TAG, INT6_TAG};

    /// What reading a layer says: its error, or that it loaded.
    fn says(r: Result<Host>) -> String {
        match r {
            Ok(_) => "loaded".to_string(),
            Err(e) => e.0,
        }
    }

    /// The layer of the pinned config (8 of 288, scale 2.5, an expert of 2,048 clamped at 10) is the layer moe.rs holds for GLM-5.3-Flash, and a config that does not renormalize says so.
    #[test]
    fn the_layer_of_the_flash_config() {
        let c = crate::glm::tests::parse();
        let cfg = moe_cfg(&c);
        assert_eq!(cfg, MoeCfg { experts: 288, router_hidden: 4096, hidden: 4096, inter: 2048, router: Router::sigmoid(8, 2.5), act: Act::SwiGlu { limit: Some(10.0) } });
        cfg.check().unwrap();
        assert_eq!((cfg.up_rows(), cfg.fused()), (4096, 0));
        assert!(!moe_cfg(&ConfigGlm { experts: crate::glm::GlmMoeDims { norm_topk: false, ..c.experts.clone() }, ..c }).router.renorm);
    }

    /// The layer's matrices come in the order the loader takes them: the shared expert's last two, the experts' after, gate and up as one then down, expert by expert.
    #[test]
    fn the_layers_matrices_in_the_order_they_are_read() {
        let c = crate::glm::tests::parse();
        let keys: Vec<(String, usize, usize)> = c.layer_packs(4).into_iter().skip(7).map(|p| (p.key.rsplit("layers.4.").next().unwrap().to_string(), p.rows(), p.cols)).collect();
        assert_eq!(keys, [("mlp.shared_experts.gate_proj".to_string(), 4096, 4096), ("mlp.shared_experts.down_proj".to_string(), 4096, 2048)]);
        let experts = c.expert_packs(4);
        assert_eq!(experts.len(), 576);
        assert!(experts[0].key.ends_with("experts.0.gate_proj") && experts[1].key.ends_with("experts.0.down_proj") && experts[575].key.ends_with("experts.287.down_proj"));
        assert_eq!((experts[0].rows(), experts[0].cols, experts[1].rows(), experts[1].cols), (4096, 4096, 4096, 2048));
        assert!(c.expert_packs(0).is_empty() && c.kinds[3] == GlmKind::Mla && c.moe[3]);
    }

    /// The tiny model's mixture-of-experts layers are ones the block takes: every matrix of them in whole blocks of the three block layouts.
    #[test]
    fn the_tiny_models_layers_are_ones_the_block_takes() {
        let text = crate::glm::tests::tiny_text();
        let c = ConfigGlm::parse(text, "tiny", None).unwrap();
        let cfg = moe_cfg(&c);
        cfg.check().unwrap();
        for l in (0..c.layers).filter(|&l| c.moe[l]) {
            for p in c.layer_packs(l).into_iter().rev().take(2).chain(c.expert_packs(l)) {
                for tag in [INT6_TAG, INT5_TAG, INT4_TAG] {
                    assert!(expert_strides(tag, p.rows(), p.cols).is_ok(), "{} is {} x {}", p.key, p.rows(), p.cols);
                }
            }
        }
    }

    /// A layer is read from the FP8 checkpoint as the packers make it from the bf16 the matrices dequantise to (the same checkpoint with them dequantised first), at the three levels; the same layer saved as a glyd-v7
    /// directory (`pack`'s own packs) is read to the same bytes, its bias fp32; and what is asked that the checkpoint cannot give is refused.
    #[test]
    fn a_layer_is_read_from_fp8_bf16_and_a_saved_directory_alike() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-moe-glm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let text = crate::glm::tests::tiny_text();
        let c = ConfigGlm::parse(text, "tiny", None).unwrap();
        crate::glm::tests::checkpoint(&dir, text, &c);
        let (fp8, bf16) = (Checkpoint::open(&dir.join("fp8")).unwrap(), Checkpoint::open(&dir.join("bf16")).unwrap());
        let l = 1;
        assert!(fp8.get("model.language_model.layers.1.mlp.experts.3.down_proj.weight_scale_inv").is_some());
        for level in [Layout::Int6, Layout::Int5, Layout::Int4] {
            let (a, b) = (read_layer(l, &c, &fp8, level).unwrap(), read_layer(l, &c, &bf16, level).unwrap());
            let same = |x: &Packed, y: &Packed| x.data == y.data && x.b1 == y.b1 && (x.rows, x.cols) == (y.rows, y.cols);
            assert!(same(&a.s_up, &b.s_up) && same(&a.s_down, &b.s_down), "{level:?}: the shared expert");
            assert!(a.block().0.data == b.block().0.data && a.block().0.scales == b.block().0.scales && a.block().1.data == b.block().1.data && a.block().1.scales == b.block().1.scales, "{level:?}: the experts");
            assert_eq!((a.block().0.experts, a.block().0.rows, a.block().0.cols, a.block().1.rows, a.block().1.cols), (4, 384, 256, 256, 192));
            assert_eq!((a.s_up.rows, a.s_up.cols, a.s_down.rows, a.s_down.cols), (384, 256, 256, 192));
            assert!(a.router == b.router && a.bias == b.bias && a.router.len() == 4 * 256 * 2 && a.bias.len() == 4 * 4);
            // the bias is fp32 in the checkpoint (not rounded to bf16), and the router bf16
            assert_eq!(a.bias, fp8.read("model.language_model.layers.1.mlp.gate.e_score_correction_bias").unwrap());
            // saved at this level, read back
            let out = dir.join(level.name());
            crate::pack::run_with(dir.join("fp8").to_str().unwrap(), &out, 2, level, true).unwrap();
            let saved = Checkpoint::open(&out).unwrap();
            assert!(saved.get("model.language_model.layers.1.mlp.experts.3.down_proj.glyd_data").is_some() && saved.get("model.language_model.layers.1.mlp.experts.3.down_proj.weight").is_none());
            let s = read_layer(l, &c, &saved, level).unwrap();
            assert!(same(&s.s_up, &a.s_up) && same(&s.s_down, &a.s_down), "{level:?}: the shared expert of the saved directory");
            assert!(s.block().0.data == a.block().0.data && s.block().0.scales == a.block().0.scales && s.block().1.data == a.block().1.data && s.block().1.scales == a.block().1.scales, "{level:?}: the experts of the saved directory");
            assert!(s.router == a.router && s.bias == a.bias);
            // a directory in the other level
            let other = if level == Layout::Int6 { Layout::Int5 } else { Layout::Int6 };
            assert!(says(read_layer(l, &c, &saved, other)).contains("packed in"));
        }
        // (the lossless level: the experts stacked and packed as one)
        assert_eq!(says(read_layer(l, &c, &fp8, Layout::Tiered)), "loaded");
        assert!(says(read_layer(l, &c, &fp8, Layout::Int3)).contains("kestrel, swift and wren levels"));
        assert!(says(read_layer(0, &c, &fp8, Layout::Int6)).contains("not a mixture-of-experts layer"));
        assert!(says(read_layer(9, &c, &fp8, Layout::Int6)).contains("not a mixture-of-experts layer"));
        // (the packs read: the shared expert's two and the four experts' two each)
        assert_eq!(c.layer_packs(l).len() - 7 + c.expert_packs(l).len(), 2 + 8);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
