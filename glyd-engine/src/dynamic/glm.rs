//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) as a description: the four streams mixed around each layer's two blocks (transformers' rounding), Kimi delta attention or the latent attention over the
//! keys its indexer picks (the layer kind's, `Op::Mla`: src/dynamic/mla.rs), a dense MLP in the first layers and a mixture of experts in the rest, the mean of the streams and the final norm. Loaded by the
//! executor's loader (`Dyn::new_placed`): the lowered description (`lower::glm`) names every matrix and small tensor, and each layer's experts are the Moe kind's block (`experts::held`): at the lossless
//! level its experts as released in FP8, with its shared expert where that is FP8, at a block level packed at it. The arithmetic is kernels/glm.cu's and kernels/mla.cu's.

use super::{
    Dyn, HeadMix, IdxKeys, IdxPick, IdxScore, MlaIn, MlaIndexer, MlaOp, ModelDesc, MoeOp, Op, Opts, Placing, ResidentMoe, Ring, Slot, Stage, StreamsMix, View,
};
use crate::dev::Dev;
use crate::dynamic::experts::MoeBlock;
use crate::ffi::{err, Error, Result};
use crate::glm::{ConfigGlm, GlmKind};
use crate::kern::GM_D;
use crate::moe::Act;
use crate::runtime::kinds::Rounding;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// The model, its experts at `level` (penguin's lossless layout, or the block layouts Int6, Int5, Int4), its other matrices at `o.layout`.
pub fn load(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    load_stage(dir, o, level, Stage::whole(ConfigGlm::load(dir)?.layers))
}

/// Stage `st` of it (a model split across GPUs: its layers, the embedding where it is the first, the head and the final norm where the last; a layer of another stage is an empty one here).
pub fn load_stage(dir: &Path, o: &Opts, level: Layout, st: Stage) -> Result<Dyn> {
    load_stage_hb(dir, o, level, st, None)
}

/// The dynamic route's stage (src/dynamic/split.rs `StageLoad`): `load_stage_hb` with its experts at `o.layout`'s level (penguin's lossless where none is given).
pub fn stage(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    load_stage_hb(dir, o, o.layout.unwrap_or(Layout::Tiered), st, caching)
}

/// `load_stage` with its experts as `caching` says, where it is given (the stage's places and memory, src/hbg_store.rs): a hummingbird directory's in the expert
/// cache, the release's in the cache's places (FP8 as released each packed on its own; at a block level each expert's packs). Loaded by the executor's loader
/// from the lowered description (`lower::glm`, every tensor named); each layer's experts the Moe kind's block (`experts::held`). Every layer is across the
/// thread's group of GPUs where it has one (src/dynamic/tp.rs); the stage's last layers' matrices in the computer's memory where the cache says so
/// (src/dense_ram.rs). A development build's GLYD_GLM_F8=0: FP8 experts as the bf16 they dequantise to.
pub fn load_stage_hb(dir: &Path, o: &Opts, level: Layout, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let c = ConfigGlm::load(dir)?;
    if c.kda.dim != GM_D || c.hc.mult != 4 {
        return err(format!("Kimi delta heads of {} and {} streams: this engine runs heads of {GM_D} and 4 streams", c.kda.dim, c.hc.mult));
    }
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let d = lowered(dir, &c, &st).map_err(|e| e.context(&label))?;
    let (dev, lib) = (Dev::new(o.device)?, crate::ffi::Lib::find()?);
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let store = caching.map(crate::hbg_store::store_for);
    let f8 = (crate::tune!("GLYD_GLM_F8").as_deref() != Some("0")).then_some(super::experts::F8 { scales: &kinds.scales, dev: &dev, lib: &lib });
    let held = super::experts::Held { hb: hb.as_ref(), cache: caching.zip(store.as_ref()), ordinal: o.device, f8, dense: None, codebooks: None, codes: None };
    let hidden = kinds.desc.hidden;
    let experts = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_layers(&kinds, st.layers, 2)
            .map(|(ix, m, roles)| {
                crate::status::STATUS.beat();
                Ok(Box::new(ResidentMoe(super::experts::held(m, roles, hidden, ck, level, Some((ix, &held)))?)) as Box<dyn MoeBlock>)
            })
            .collect()
    };
    let placing = Placing { across: true, ram: crate::dense_ram::Placer::last(st.layers.0, st.layers.1, caching.map_or(0, |x| x.dense)) };
    Dyn::new_placed(dir, d, None, st, o, &experts, &placing)
}

/// GLM-5.3 (`glm_moe_dsa`, the full-size layout: no streams; the latent attention on every layer, its rotary and its indexer its own or shared) from `dir`, by
/// its layer kinds alone: the description lowered (`lower::latent_moe`), each layer's experts the Moe kind's resident block (`experts::held`: the sigmoid
/// router with its choice bias and scale, the experts and the shared expert); every expert on the GPU. A checkpoint as released, in FP8 with fp32 block
/// scales (`weight_scale_inv`), is held as released at the lossless level: the attention's and the dense MLP's matrices (`Mat::F8`), kv_b_proj and the
/// indexer's products as their E4M3 codes and scales (`Part::scales`), the experts and the shared expert (`experts::F8`); at a block level its matrices are
/// packed from the bf16 they dequantise to. A development build's GLYD_GLM_F8=0: the experts as that bf16 at the lossless level too.
pub fn load_plain(dir: &Path, o: &Opts) -> Result<Dyn> {
    plain(dir, o, None, None, None)
}

/// `load_plain` with the model's draft head (its MTP layer, DeepSeek-V3's joined form: `lower::draft_head`, src/dynamic/head.rs) drafting `k` tokens a round
/// (at most `VERIFY_ROWS` - 1), its experts held as the model's.
pub fn load_plain_drafting(dir: &Path, o: &Opts, k: usize) -> Result<Dyn> {
    plain(dir, o, None, Some(k), None)
}

/// The dynamic route's stage of the model (src/dynamic/split.rs `StageLoad`): stage `st`'s layers, every expert of them on its GPU, or in the expert cache as
/// `caching` says where it is given (src/tiering.rs: its places a layer on the GPU, every expert in the computer's memory; FP8 experts as released each packed
/// on its own, `experts::F8`), and the stage's last layers' matrices in the computer's memory where it says so (src/dense_ram.rs: refused where they are released
/// in FP8, whichever the family).
pub fn stage_plain(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    plain(dir, o, Some(st), None, caching)
}

/// The layers and end tokens of the model in `dir`, as `load_plain` reads them (the dynamic route's `Loader::ends`).
pub fn plain_ends(dir: &Path) -> Result<(usize, Vec<i64>)> {
    let (kinds, _, eos, _) = plain_config(dir)?;
    Ok((kinds.desc.blocks.len() / 2, eos))
}

/// The model's description, its label, its end tokens (generation_config.json's, else the config's) and its most positions.
fn plain_config(dir: &Path) -> Result<(crate::runtime::describe::Described, String, Vec<i64>, usize)> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let cfg = crate::runtime::config::ModelConfig::load(dir)?;
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let mut eos = crate::runtime::config::eos_ids(&cfg.text, generation.as_deref());
    if eos.is_empty() {
        eos = crate::runtime::config::eos_ids(&cfg.top, None);
    }
    let max_pos = cfg.shared.max_position_embeddings.ok_or_else(|| Error("no max_position_embeddings".into()))?;
    Ok((kinds, label, eos, max_pos))
}

/// The latent attention with plain residuals lowered for the whole of a description (`lower::latent_moe`): a draft head's blocks' lowering.
fn latent_whole(d: &crate::runtime::describe::Described, eos: Vec<i64>, max_pos: usize) -> Result<ModelDesc> {
    super::lower::latent_moe(d, eos, max_pos, &Stage::whole(d.desc.blocks.len() / 2))
}

/// `load_plain` of stage `st` (the whole model where None), with its draft head drafting `draft` tokens a round where given, its experts in the expert cache as
/// `caching` says where it is given (not with a draft head: its verify pass takes the experts on the GPU).
fn plain(dir: &Path, o: &Opts, st: Option<Stage>, draft: Option<usize>, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    if draft.is_some() && caching.is_some() {
        return err("GLM-5.3's draft head with its experts in the expert cache: its verify pass takes the experts on the GPU");
    }
    let (kinds, label, eos, max_pos) = plain_config(dir)?;
    let layers = kinds.desc.blocks.len() / 2;
    let st = st.unwrap_or(Stage::whole(layers));
    let d = super::lower::latent_moe(&kinds, eos.clone(), max_pos, &st).map_err(|e| e.context(&label))?;
    let head = match draft {
        Some(_) => Some(super::lower::draft_head(&kinds, eos, max_pos, latent_whole).map_err(|e| e.context(&label))?),
        None => None,
    };
    let hidden = kinds.desc.hidden;
    let (dev, lib) = (Dev::new(o.device)?, crate::ffi::Lib::find()?);
    let f8 = (!kinds.scales.is_empty() && crate::tune!("GLYD_GLM_F8").as_deref() != Some("0")).then_some(super::experts::F8 {
        scales: &kinds.scales,
        dev: &dev,
        lib: &lib,
    });
    let store = caching.map(crate::hbg_store::store_for);
    let held = super::experts::Held { hb: None, cache: caching.zip(store.as_ref()), ordinal: o.device, f8, dense: None, codebooks: None, codes: None };
    let experts = |ck: &Checkpoint, level: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_layers(&kinds, st.layers, 2)
            .map(|(ix, m, roles)| {
                crate::status::STATUS.beat(); // (a block of experts takes seconds to read)
                Ok(Box::new(ResidentMoe(super::experts::held(m, roles, hidden, ck, level, Some((ix, &held)))?)) as Box<dyn MoeBlock>)
            })
            .collect()
    };
    // (the head's experts as the model's, its layer the one after the model's)
    // (its block of experts numbered after the model's: the expert cache's next layer)
    let model_moe = super::experts::of_layers(&kinds, (0, layers), 2).count();
    let head_experts = |ck: &Checkpoint, level: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_head(&kinds, 0)
            .map(|(b, m, roles)| {
                let ix = super::experts::Ix { layer: layers + b / 2, moe: model_moe + b / 2 };
                Ok(Box::new(ResidentMoe(super::experts::held(m, roles, hidden, ck, level, Some((ix, &held)))?)) as Box<dyn MoeBlock>)
            })
            .collect()
    };
    let spec = head.zip(draft).map(|(h, k)| super::DraftSpec { dir, desc: h, k, experts: Some(&head_experts) });
    let placing = Placing { across: false, ram: crate::dense_ram::Placer::last(st.layers.0, st.layers.1, caching.map_or(0, |x| x.dense)) };
    Dyn::new_placed(dir, d, spec, st, o, &experts, &placing)
}

/// A layer's matrices as the loader holds them: a Kimi delta layer's seven (qkv, o, f_a, f_b, g_a, g_b, b) or a latent attention layer's four (q_a, q_b, kv_a, o),
/// then a dense MLP's gate and up and its down; a mixture of experts' are its block's.
pub(crate) fn per_layer(kind: GlmKind, moe: bool) -> usize {
    (if kind == GlmKind::Kda { 7 } else { 4 }) + if moe { 0 } else { 2 }
}

/// GLM-5.3-Flash's latent attention as the layer kind's operation (`MlaOp`, its matrices, small tensors and rings at 0: `MlaOp::placed` puts a layer's): q's and the
/// key-value latent's products apart, no shared key part, the norms' epsilon the config's and their rounding the release's (`Rounding::Twice`: the normed value to
/// bf16, times the weight, rounded), and the indexer over pools of its keys (its scores in f32, the picks given as their tokens and the open pool's).
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
        q_round: Rounding::Twice,
        gate: false,
        scale: (m.nope as f32).powf(-0.5),
        eps: c.eps,
        ring: 0,
        rot: None,
        lent: None,
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
                weighted: true,
                cand: None,
            },
            dim: ix.dim,
            q_b: 0,
            k: 0,
            k_norm: 0,
            k_bias: 0,
            weights: 0,
            ape: 0,
            gate: 0,
            key: super::KeyNorm::Layer(1e-6),
        }),
    }
}

/// GLM-5.3-Flash's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::glm`, every tensor named), held to
/// `describe` with the names left out (the family's own description names none).
pub fn lowered(dir: &Path, c: &ConfigGlm, st: &Stage) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let lowered = super::lower::glm(&kinds, c.eos.clone(), c.max_pos, st)?;
    super::lower::agree(super::lower::nameless(&lowered), &describe(c, st)?, &label)?;
    Ok(lowered)
}

/// The executor's description of GLM-5.3-Flash for stage `st`, in the order its loader reads what it loads (`lower::glm` names them). Each layer's small
/// tensors: its two norms, the attention's stream mix (its projection, base and scale), a Kimi delta layer's convolutions, A_log, dt_bias and gated norm or a
/// latent layer's ten, then the feed-forward's stream mix; its matrices `per_layer`'s. The final norm after the last layer's.
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
        let mut ops = vec![Op::StreamsPre { mix: StreamsMix::Hc { f: s + 2, base: s + 3, scale: s + 4 }, norm: s, round: Rounding::Twice }];
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
        ops.extend([
            Op::StreamsPost { round: Rounding::Twice, gated: false },
            Op::StreamsPre { mix: StreamsMix::Hc { f: h, base: h + 1, scale: h + 2 }, norm: s + 1, round: Rounding::Twice },
        ]);
        if c.moe[l] {
            ops.push(Op::Moe(MoeOp::of(n_moe)));
            n_moe += 1;
        } else {
            ops.extend(dense_ops(m0 + nm - 2, c.inter, c.experts.limit));
        }
        ops.push(Op::StreamsPost { round: Rounding::Twice, gated: false });
        layers.push(ops);
    }
    if st.last {
        layers.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::StreamsHead {
            mix: HeadMix::Mean,
            norm: n_small,
            round: Rounding::Twice,
        });
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
        Op::GatedAct { inter, act: Act::SwiGlu { limit: Some(limit) } },
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
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: Op::Streams,
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
        host: Vec::new(),
    }
}
