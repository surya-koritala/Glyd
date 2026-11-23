//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) as a description: the four streams mixed around each layer's two blocks (transformers' rounding), Kimi delta attention or the latent attention over the
//! keys its indexer picks (the layer kind's, `Op::Mla`: src/dynamic/mla.rs), a dense MLP in the first layers and a mixture of experts in the rest, the mean of the streams and the final norm. Its weights load by
//! model_glm.rs's loader (a latent attention's by mla_glm.rs's `MlaWeights`) and the experts' blocks by their factory, handed over as they are; the arithmetic is kernels/glm.cu's and kernels/mla.cu's, as
//! model_glm.rs runs it (`glmdyn` checks the two against each other).

use super::{Dyn, IdxKeys, IdxPick, IdxScore, Loaded, MlaIn, MlaIndexer, MlaOp, ModelDesc, Op, Opts, Ring, Slot, Stage, View};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::glm::{ConfigGlm, GlmKind};
use crate::kern35::Mat;
use crate::kern_nh::{MoeBlock, NhHost};
use crate::model_glm::{WeightsGlm, GM_D};
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
    load_stage_hb(dir, o, level, st, None)
}

/// `load_stage` with a hummingbird directory's experts in the expert cache (`caching`: the stage's places and memory, src/hbg_store.rs).
pub fn load_stage_hb(dir: &Path, o: &Opts, level: Layout, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let dev = Dev::new(o.device)?;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let lib8 = crate::ffi::Lib::find()?;
    let only: Vec<usize> = (st.layers.0..st.layers.1).collect();
    let whole = st.first && st.last;
    let w = {
        let moe = crate::moe_glm::factory_cached(level, &dev, &lib8, caching.copied());
        WeightsGlm::load_ends(
            dir,
            threads,
            o.layout,
            dev.cc_pair(),
            dev.total_mem as u64,
            (!whole).then_some(&only[..]),
            Some((st.first, st.last)),
            Some(&moe),
            None,
        )?
    };
    // (the latent attention layers' weights: the kind's operation's matrices and small tensors)
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let lm = w.lm_head().is_some();
    let WeightsGlm { cfg: c, layout, layers: lys, embed, head, norm_f, .. } = w;
    if c.kda.dim != GM_D || c.hc.mult != 4 {
        return err(format!("Kimi delta heads of {} and {} streams: this engine runs heads of {GM_D} and 4 streams", c.kda.dim, c.hc.mult));
    }
    let (hid, qd, heads) = (c.hidden, c.qkv(), c.kda.heads);
    let mut l_ = Loaded::default();
    let (mut states, mut layers, mut dense, mut rings) = (Vec::new(), Vec::new(), false, Vec::new());
    let mla = latent(&c);
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
        let pre = |small: &mut Vec<Buf>, h: crate::model_glm::Hc, norm: usize| Op::GlmPre {
            f: put(small, h.proj),
            base: put(small, h.base),
            scale: put(small, h.scale),
            norm,
        };
        let mut ops = vec![pre(&mut l_.small, ha, na)];
        // (a latent attention layer's four matrices first, then the loader's: the dense MLP's)
        let latent_w = if ly.kind == GlmKind::Mla { Some(crate::mla_glm::MlaWeights::load(l, &c, &ck, layout)?) } else { None };
        let m0 = l_.mats.len();
        let nm = ly.mats.len() + if latent_w.is_some() { 4 } else { 0 };
        let (want_m, want_s) = (per_layer(ly.kind, ly.moe.is_some()), if ly.kind == GlmKind::Kda { 4 } else { 0 });
        if (nm, ly.small.len()) != (want_m, want_s) {
            return err(format!("layer {l}: {nm} matrices and {} small tensors loaded, where its kind has {want_m} and {want_s}", ly.small.len()));
        }
        let latent_w = latent_w.map(|w| {
            l_.mats.extend([w.q_a, w.q_b, w.kv_a, w.o]);
            [w.q_a_ln, w.kv_a_ln, w.kv_b, w.wq_b, w.wk, w.k_norm_w, w.k_norm_b, w.wproj, w.ape, w.ix_gate]
        });
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
                let s0 = l_.small.len();
                l_.small.extend(latent_w.ok_or_else(|| Error(format!("layer {l}: a latent attention layer with no weights")))?);
                let op = mla.placed(m0, s0, rings.len());
                rings.extend(op.rings());
                ops.push(Op::Mla(op));
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
                dense = true;
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
    for (s, wd) in [
        (Slot::X, hid),
        (Slot::S, 4 * hid),
        (Slot::H, hid),
        (Slot::O, hid),
        (Slot::Mix, 3 * qd),
        (Slot::Xc, 3 * qd),
        (Slot::Proj, c.kda.dim),
        (Slot::Qo, qd),
        (Slot::Iq, qd),
        (Slot::Ba, heads.max(8)),
        (Slot::Core, qd),
        (Slot::Att, qd),
        // (the dense MLP's, where a layer has one: as the description lowered from the layer kinds sizes them)
        (Slot::Gu, if dense { 2 * c.inter } else { 0 }),
        (Slot::M, if dense { c.inter } else { 0 }),
    ] {
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
        rings,
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
    // (the loader's own operations, as it indexed what it loaded, held to the layer kinds' description of the config: one model, or it is not loaded)
    let d = super::lower::agree(lowered(dir, &c, &st)?, &d, &dir.join("config.json").display().to_string())?;
    Dyn::from_parts(dev, d, l_, Vec::new(), layout, st, o)
}

/// A layer's matrices as the loader holds them: a Kimi delta layer's seven (qkv, o, f_a, f_b, g_a, g_b, b) or a latent attention layer's four (q_a, q_b, kv_a, o),
/// then a dense MLP's gate and up and its down; a mixture of experts' are its block's.
pub(crate) fn per_layer(kind: GlmKind, moe: bool) -> usize {
    (if kind == GlmKind::Kda { 7 } else { 4 }) + if moe { 0 } else { 2 }
}

/// GLM-5.3-Flash's latent attention as the layer kind's operation (`MlaOp`, its matrices, small tensors and rings at 0: `MlaOp::placed` puts a layer's): q's and the
/// key-value latent's products apart, no shared key part, the norms' epsilon the config's, and the indexer over pools of its keys (its scores in f32, the picks given
/// as their tokens and the open pool's).
pub(crate) fn latent(c: &ConfigGlm) -> MlaOp {
    let (m, ix) = (&c.mla, &c.indexer);
    MlaOp {
        heads: m.heads,
        nope: m.nope,
        rope: 0,
        v: m.v,
        kv_rank: m.kv_rank,
        q_rank: m.q_rank,
        input: MlaIn::Apart { q_a: 0, kv_a: 0 },
        q_b: 0,
        o: 0,
        q_norm: 0,
        kv_norm: 0,
        kv_b: 0,
        q_norm512: false,
        gate: false,
        scale: (m.nope as f32).powf(-0.5),
        eps: c.eps,
        ring: 0,
        indexer: Some(MlaIndexer {
            pick: IdxPick {
                heads: ix.heads,
                ratio: ix.kpool,
                top_k: ix.topk / ix.kpool.max(1),
                expand: true,
                tail: ix.tail,
                wscale: (ix.heads as f32).powf(-0.5),
                score: IdxScore::F32 { dim: ix.dim, scale: (ix.dim as f32).powf(-0.5) },
                keys: IdxKeys::Ring(1),
            },
            dim: ix.dim,
            q_b: 0,
            k: 0,
            k_norm: 0,
            k_bias: 0,
            weights: 0,
            ape: 0,
            gate: 0,
        }),
    }
}

/// GLM-5.3-Flash's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::glm`), held to `describe`.
pub fn lowered(dir: &Path, c: &ConfigGlm, st: &Stage) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let lowered = super::lower::glm(&kinds, c.eos.clone(), c.max_pos, st)?;
    super::lower::agree(lowered, &describe(c, st)?, &label)
}

/// The executor's description of GLM-5.3-Flash for stage `st`, as the loader indexes what it loads. Each layer's small tensors: its two norms, the attention's
/// stream mix (its projection, base and scale), a Kimi delta layer's convolutions, A_log, dt_bias and gated norm, then the feed-forward's stream mix; its
/// matrices `per_layer`'s. The final norm after the last layer's.
pub fn describe(c: &ConfigGlm, st: &Stage) -> Result<ModelDesc> {
    if c.kda.dim != GM_D || c.hc.mult != 4 {
        return err(format!("Kimi delta heads of {} and {} streams: this engine runs heads of {GM_D} and 4 streams", c.kda.dim, c.hc.mult));
    }
    let (hid, qd, heads) = (c.hidden, c.qkv(), c.kda.heads);
    let mla = latent(c);
    let (mut n_small, mut n_mats, mut n_moe, mut states, mut layers, mut rings) = (0, 0, 0, Vec::new(), Vec::new(), Vec::new());
    for l in 0..c.layers {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let s = n_small;
        n_small += 5;
        let mut ops = vec![Op::GlmPre { f: s + 2, base: s + 3, scale: s + 4, norm: s }];
        let m0 = n_mats;
        let nm = per_layer(c.kinds[l], c.moe[l]);
        n_mats += nm;
        match c.kinds[l] {
            GlmKind::Kda => {
                let s0 = n_small;
                n_small += 4;
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * 3 * qd * 2);
                states.push(heads * GM_D * GM_D * 4);
                ops.extend(kda_ops(m0, s0, conv, ssm, qd, heads, c.kda.bound));
            }
            GlmKind::Mla => {
                let op = mla.placed(m0, n_small, rings.len());
                n_small += op.small_count();
                rings.extend(op.rings());
                ops.push(Op::Mla(op));
            }
        }
        let h = n_small;
        n_small += 3;
        ops.extend([Op::GlmPost, Op::GlmPre { f: h, base: h + 1, scale: h + 2, norm: s + 1 }]);
        if c.moe[l] {
            ops.push(Op::Moe { block: n_moe });
            n_moe += 1;
        } else {
            ops.extend(dense_ops(m0 + nm - 2, c.inter, c.experts.limit));
        }
        ops.push(Op::GlmPost);
        layers.push(ops);
    }
    if st.last {
        layers.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::GlmHead { norm: n_small });
    }
    let e = n_mats;
    let lm = if st.last && !c.tie { e + 1 } else { e };
    // (the dense MLP's width where a layer has one, else none: as the description lowered from the layer kinds sizes its slots)
    let inter = if (0..c.layers).any(|l| !c.moe[l]) { c.inter } else { 0 };
    Ok(glm_desc(hid, qd, heads, inter, c.vocab, c.eps, c.max_pos, c.eos.clone(), (c.hc.eps, c.hc.iters as i32), layers, (states, rings), e, lm))
}

/// A Kimi delta layer's operations: its matrices from `m0`, its small tensors from `s0`, its convolution's and recurrence's states.
pub(crate) fn kda_ops(m0: usize, s0: usize, conv: usize, ssm: usize, qd: usize, heads: usize, bound: f32) -> Vec<Op> {
    vec![
        Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "linear attention: qkv product" },
        Op::KdaConv { w: s0, state: conv, c: 3 * qd, row: 3 * qd, w32: false },
        Op::Linear { w: m0 + 2, x: Slot::H, y: Slot::Proj, name: "linear attention: gates" },
        Op::Linear { w: m0 + 3, x: Slot::Proj, y: Slot::Qo, name: "linear attention: gates" },
        Op::Linear { w: m0 + 6, x: Slot::H, y: Slot::Ba, name: "linear attention: gates" },
        Op::Linear { w: m0 + 4, x: Slot::H, y: Slot::Proj, name: "linear attention: gates" },
        Op::Linear { w: m0 + 5, x: Slot::Proj, y: Slot::Iq, name: "linear attention: gates" },
        Op::KdaDelta { a_log: s0 + 1, dt_bias: s0 + 2, state: ssm, heads, bound, ba: View::slot(Slot::Ba) },
        Op::KdaOut { w: s0 + 3, heads, w32: false, iq: View::slot(Slot::Iq) },
        Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "linear attention: o product" },
    ]
}

/// A dense MLP's operations: its gate and up product at `gu`, its down product after it.
pub(crate) fn dense_ops(gu: usize, inter: usize, limit: f32) -> Vec<Op> {
    vec![
        Op::Linear { w: gu, x: Slot::H, y: Slot::Gu, name: "MLP: gate and up product" },
        Op::SwiGluClamp { inter, limit },
        Op::Linear { w: gu + 1, x: Slot::M, y: Slot::O, name: "MLP: down product" },
    ]
}

/// The rest of GLM-5.3-Flash's description: its slots' widths and fixed fields.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_desc(
    hid: usize,
    qd: usize,
    heads: usize,
    inter: usize,
    vocab: usize,
    eps: f32,
    max_pos: usize,
    eos: Vec<i64>,
    hc: (f32, i32),
    layers: Vec<Vec<Op>>,
    (states, rings): (Vec<usize>, Vec<Ring>),
    embed: usize,
    lm: usize,
) -> ModelDesc {
    let mut widths = [0usize; super::SLOTS];
    for (s, wd) in [
        (Slot::X, hid),
        (Slot::S, 4 * hid),
        (Slot::H, hid),
        (Slot::O, hid),
        (Slot::Mix, 3 * qd),
        (Slot::Xc, 3 * qd),
        (Slot::Proj, GM_D),
        (Slot::Qo, qd),
        (Slot::Iq, qd),
        (Slot::Ba, heads.max(8)),
        (Slot::Core, qd),
        (Slot::Att, qd),
        (Slot::Gu, 2 * inter),
        (Slot::M, inter),
    ] {
        widths[s as usize] = wd;
    }
    ModelDesc {
        family: "GLM-5.3-Flash",
        hidden: hid,
        vocab,
        eps,
        max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: Op::GlmStreams,
        layers,
        states,
        final_norm: None,
        decoded: false,
        skinny: false,
        embed,
        lm,
        softcap: 0.0,
        eos,
        post: None,
        v4: None,
        hc,
        logit_mult: 0.0,
    }
}
