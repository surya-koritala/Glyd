//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) as a description: the four streams mixed around each layer's two blocks (transformers' rounding), Kimi delta attention or the latent attention over the
//! keys its indexer picks (src/mla_glm.rs's block, run through `GlmHost`), a dense MLP in the first layers and a mixture of experts in the rest, the mean of the streams and the final norm. Its weights load by
//! model_glm.rs's loader and the blocks by their factories, handed over as they are; the arithmetic is kernels/glm.cu's, as model_glm.rs runs it (`glmdyn` checks the two against each other).

use super::{Dyn, Loaded, ModelDesc, Op, Opts, Slot, Stage, View};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::glm::{ConfigGlm, GlmKind};
use crate::model35::Mat;
use crate::model_glm::{WeightsGlm, GM_D};
use crate::model_nh::{MoeBlock, NhHost};
use crate::moe::MoeLayer;
use crate::weights::Layout;
use std::path::Path;

/// A GLM block of experts on the dynamic engine: its core's layer, run by `Op::Moe` (the router, the experts, the shared expert).
struct GlmMoeIn(Box<dyn crate::model_glm::MoeBlock>);

impl MoeBlock for GlmMoeIn {
    unsafe fn forward(&self, _: &dyn NhHost, _: u64, _: usize, _: u64) -> Result<()> {
        err("GLM's experts run on the dynamic engine")
    }
    fn graphable(&self) -> bool {
        true
    }
    fn bytes(&self) -> usize {
        self.0.bytes()
    }
    fn resident(&self) -> Option<(&MoeLayer, Option<(&Mat, &Mat)>)> {
        self.0.layer().map(|l| (l, None))
    }
}

/// The model, its experts at `level` (penguin's lossless layout, or the block layouts Int6, Int5, Int4), its other matrices at `o.layout`.
pub fn load(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    load_stage(dir, o, level, Stage::whole(ConfigGlm::load(dir)?.layers))
}

/// Stage `st` of it (a model split across GPUs: its layers, the embedding where it is the first, the head and the final norm where the last; a layer of another stage is an empty one here).
pub fn load_stage(dir: &Path, o: &Opts, level: Layout, st: Stage) -> Result<Dyn> {
    let dev = Dev::new(o.device)?;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let lib8 = crate::ffi::Lib::find()?;
    let mla = crate::mla_glm::factory();
    let only: Vec<usize> = (st.layers.0..st.layers.1).collect();
    let whole = st.first && st.last;
    let w = {
        let moe = crate::moe_glm::factory_f8(level, &dev, &lib8);
        WeightsGlm::load_ends(dir, threads, o.layout, dev.cc_pair(), dev.total_mem as u64, (!whole).then_some(&only[..]), Some((st.first, st.last)), Some(&moe), Some(&mla))?
    };
    let lm = w.lm_head().is_some();
    let WeightsGlm { cfg: c, layout, layers: lys, embed, head, norm_f, .. } = w;
    if c.kda.dim != GM_D || c.hc.mult != 4 {
        return err(format!("Kimi delta heads of {} and {} streams: this engine runs heads of {GM_D} and 4 streams", c.kda.dim, c.hc.mult));
    }
    let (hid, qd, heads) = (c.hidden, c.qkv(), c.kda.heads);
    let mut l_ = Loaded::default();
    let (mut states, mut layers) = (Vec::new(), Vec::new());
    let put = |v: &mut Vec<Buf>, b: Buf| {
        v.push(b);
        v.len() - 1
    };
    for (l, ly) in lys.into_iter().enumerate() {
        let Some(ly) = ly else {
            if (st.layers.0..st.layers.1).contains(&l) {
                return err(format!("layer {l} is not loaded"));
            }
            // (another stage's layer: nothing here, never run)
            layers.push(Vec::new());
            continue;
        };
        let [ha, hf] = ly.hc;
        let (na, nf) = (put(&mut l_.small, ly.norm_a), put(&mut l_.small, ly.norm_f));
        let pre = |small: &mut Vec<Buf>, h: crate::model_glm::Hc, norm: usize| Op::GlmPre { f: put(small, h.proj), base: put(small, h.base), scale: put(small, h.scale), norm };
        let mut ops = vec![pre(&mut l_.small, ha, na)];
        let m0 = l_.mats.len();
        let nm = ly.mats.len();
        l_.mats.extend(ly.mats);
        match ly.kind {
            GlmKind::Kda => {
                // (matrices: qkv, o, f_a, f_b, g_a, g_b, b; small: the convolutions' weights, A_log, dt_bias, the gated norm's weight)
                let s0 = l_.small.len();
                l_.small.extend(ly.small);
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * 3 * qd * 2);
                states.push(heads * GM_D * GM_D * 4);
                ops.extend([
                    Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "linear attention: qkv product" },
                    Op::KdaConv { w: s0, state: conv, c: 3 * qd, row: 3 * qd, w32: false },
                    Op::Linear { w: m0 + 2, x: Slot::H, y: Slot::Proj, name: "linear attention: gates" },
                    Op::Linear { w: m0 + 3, x: Slot::Proj, y: Slot::Qo, name: "linear attention: gates" },
                    Op::Linear { w: m0 + 6, x: Slot::H, y: Slot::Ba, name: "linear attention: gates" },
                    Op::Linear { w: m0 + 4, x: Slot::H, y: Slot::Proj, name: "linear attention: gates" },
                    Op::Linear { w: m0 + 5, x: Slot::Proj, y: Slot::Iq, name: "linear attention: gates" },
                    Op::KdaDelta { a_log: s0 + 1, dt_bias: s0 + 2, state: ssm, heads, bound: c.kda.bound, ba: View::slot(Slot::Ba) },
                    Op::KdaOut { w: s0 + 3, heads, w32: false, iq: View::slot(Slot::Iq) },
                    Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "linear attention: o product" },
                ]);
            }
            GlmKind::Mla => {
                let b = ly.mla.ok_or_else(|| Error(format!("layer {l}: a latent attention layer with no block")))?;
                ops.push(Op::Mla { block: l_.mla.len() });
                l_.mla.push(b);
            }
        }
        ops.extend([Op::GlmPost, pre(&mut l_.small, hf, nf)]);
        match ly.moe {
            Some(b) => {
                if b.layer().is_none() {
                    return err(format!("layer {l}: a block of experts that is not all on the GPU: the dynamic engine runs those alone"));
                }
                ops.push(Op::Moe { block: l_.moe.len() });
                l_.moe.push(Box::new(GlmMoeIn(b)));
            }
            None => {
                // (the dense MLP's gate and up, then its down: the layer's last two matrices)
                let (gu, down) = (m0 + nm - 2, m0 + nm - 1);
                ops.extend([
                    Op::Linear { w: gu, x: Slot::H, y: Slot::Gu, name: "MLP: gate and up product" },
                    Op::SwiGluClamp { inter: c.inter, limit: c.experts.limit },
                    Op::Linear { w: down, x: Slot::M, y: Slot::O, name: "MLP: down product" },
                ]);
            }
        }
        ops.push(Op::GlmPost);
        layers.push(ops);
    }
    if st.last {
        let norm = put(&mut l_.small, norm_f.ok_or_else(|| Error("the final norm is not loaded".into()))?);
        layers.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::GlmHead { norm });
    }
    // (a stage without the embedding or the head keeps their places, empty, so that their indices hold)
    let empty = || -> Result<Mat> { Ok(Mat::Dense(Buf::new(2)?, 0, 0)) };
    let e = l_.mats.len();
    l_.mats.push(match embed {
        Some(p) => Mat::Packed(p),
        None if !st.first => empty()?,
        None => return err("the embedding is not loaded"),
    });
    let lm_i = match head {
        Some(h) => {
            l_.mats.push(Mat::Packed(h));
            e + 1
        }
        None if lm || !st.last => e,
        None => return err("the LM head is not loaded"),
    };
    let mut widths = [0usize; super::SLOTS];
    for (s, wd) in [(Slot::X, hid), (Slot::S, 4 * hid), (Slot::H, hid), (Slot::O, hid), (Slot::Mix, 3 * qd), (Slot::Xc, 3 * qd), (Slot::Proj, c.kda.dim), (Slot::Qo, qd), (Slot::Iq, qd), (Slot::Ba, heads.max(8)), (Slot::Core, qd), (Slot::Att, qd), (Slot::Gu, 2 * c.inter), (Slot::M, c.inter)] {
        widths[s as usize] = wd;
    }
    let d = ModelDesc {
        family: "GLM-5.3-Flash",
        hidden: hid,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
        rings: Vec::new(),
        first: Op::GlmStreams,
        layers,
        states,
        final_norm: None,
        decoded: false,
        skinny: false,
        embed: e,
        lm: lm_i,
        softcap: 0.0,
        eos: c.eos.clone(),
        post: None,
        v4: None,
        hc: (c.hc.eps, c.hc.iters as i32),
        logit_mult: 0.0,
    };
    Dyn::from_parts(dev, d, l_, Vec::new(), layout, st, o)
}
