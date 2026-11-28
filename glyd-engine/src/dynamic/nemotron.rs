//! Nemotron-H (`NemotronHForCausalLM`: Nemotron 3 Super, Nemotron 3.5 Lightning) as a description: Mamba-2 layers, attention layers without a positional encoding,
//! mixture-of-experts layers (with a latent, Super's; or at the model's width, Lightning's); every layer x + mixer(norm(x)) (the arithmetic is the layer kinds'
//! kernels'). Loaded by the executor's loader (`Dyn::new_placed`): the lowered description (`lower::nemotron`) names every matrix and small tensor, the matrices
//! packed in the layout the level gives every matrix but the routed experts' (`nemotron::dense_layout`: six-bit codes at wren and hummingbird), and each
//! mixture-of-experts layer's experts the Moe kind's block (`experts::block`, a block a layer: the latent's own projections around the experts, their second
//! matrices in the level's row layout, a glyd-v7 directory's packs as saved, the expert cache's places where the stage has one, the directory's codebook rows at
//! hummingbird), which `Op::Moe` runs.

use super::attn::{AttnOp, QkNorm};
use super::{Dyn, ModelDesc, MoeOp, NormW, Op, Opts, Placing, Ring, Slot, Stage};
use crate::dynamic::experts::MoeBlock;
use crate::ffi::{err, Result};
use crate::kern::{MAMBA2_HEAD, MAMBA2_STATE};
use crate::nemotron::NhKind;
use crate::offload::ExpertStore;
use crate::runtime::config::ModelConfig;
use crate::runtime::describe::Described;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;
use std::rc::Rc;

/// The dynamic route's stage (src/dynamic/split.rs `StageLoad`): `load_stage` with its experts at `o.layout`'s level (penguin's lossless where none is given),
/// all on the GPU. A caching is refused: Nemotron-H's expert cache is the lossy route's (src/cli.rs gives `load_cached` its places and store), not the dynamic
/// route's caching.
pub fn stage(dir: &Path, m: &ModelConfig, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    if caching.is_some() {
        return err(format!(
            "{}: Nemotron-H's experts are cached as the lossy route plans them (src/cli.rs), not as the dynamic route's caching says",
            dir.display()
        ));
    }
    load_cached_of(dir, m, o, o.layout.unwrap_or(Layout::Tiered), None, st)
}

/// The model in `dir` with its matrices at `layout` and its experts at `level` (the experts all on the GPU).
pub fn load(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    let layers = crate::nemotron::ConfigNemotronH::load(dir)?.layers;
    load_stage(dir, o, level, Stage::whole(layers))
}

/// The whole model with its experts at `level`, all on the GPU or in the expert cache `cache` (`load_cached_of`): what the one-GPU route's plan loads (src/cli.rs
/// `open_lossy`). `m`: `dir`'s config.json, parsed once.
pub fn load_whole_of(dir: &Path, m: &ModelConfig, o: &Opts, level: Layout, cache: Option<(usize, &Rc<ExpertStore>)>) -> Result<Dyn> {
    let layers = m.read_with(dir, crate::nemotron::ConfigNemotronH::of)?.layers;
    load_cached_of(dir, m, o, level, cache, Stage::whole(layers))
}

/// Stage `st` of it (a model split across GPUs: its layers, the embedding where it is the first, the head where the last; a layer of another
/// stage is an empty one here).
pub fn load_stage(dir: &Path, o: &Opts, level: Layout, st: Stage) -> Result<Dyn> {
    load_cached(dir, o, level, None, st)
}

/// Stage `st` with its experts at `level`, all on the GPU or in the expert cache (`cache`: places a layer for them on the GPU and the store every expert is in,
/// src/offload.rs; a layer of no more experts than places all on the GPU), by the executor's loader: the lowered description names every matrix and small tensor of
/// the stage, each matrix packed in the layout `o.layout`'s level gives every matrix but the routed experts' (`nemotron::dense_layout`; a glyd-v7 directory's packs
/// as saved, each in the layout its entry names), the stage's layers on its GPU (none in the computer's memory: the planner's row keeps none there), the Mamba-2
/// scan's buffers sized from its operations; each mixture-of-experts layer's experts the Moe kind's block (`experts::block`: its matrices besides the experts at
/// `dense_layout` of the level; its layer in the cache its number among the model's mixture-of-experts layers, so that a predictor's next is the next of them).
pub fn load_cached(dir: &Path, o: &Opts, level: Layout, cache: Option<(usize, &Rc<ExpertStore>)>, st: Stage) -> Result<Dyn> {
    load_cached_of(dir, &ModelConfig::load(dir)?, o, level, cache, st)
}

/// `load_cached` of a config.json parsed once (`m`: `dir`'s).
pub fn load_cached_of(dir: &Path, m: &ModelConfig, o: &Opts, level: Layout, cache: Option<(usize, &Rc<ExpertStore>)>, st: Stage) -> Result<Dyn> {
    let c = m.read_with(dir, crate::nemotron::ConfigNemotronH::of)?;
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let d = lowered_of(&kinds, &label, &c, &st)?;
    let caching = cache.map(|(places, store)| crate::hbg_store::HbCaching { places, ram: store.host_memory(), offload: store.offload(), dense: 0 });
    let codebooks = Rc::new(crate::moe_hb::HbShared::default());
    let held = super::experts::Held {
        hb: None,
        cache: caching.as_ref().zip(cache.map(|x| x.1)),
        ordinal: o.device,
        f8: None,
        dense: Some(crate::nemotron::dense_layout(level)),
        codebooks: Some(&codebooks),
        codes: None,
    };
    let hidden = kinds.desc.hidden;
    let experts = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_layers(&kinds, st.layers, 1)
            .map(|(ix, m, roles)| {
                crate::status::STATUS.beat(); // (a mixture-of-experts layer's block takes seconds to read)
                super::experts::block(m, roles, hidden, ck, level, (ix, &held))
            })
            .collect()
    };
    let o = Opts { layout: o.layout.map(crate::nemotron::dense_layout), ..o.clone() };
    Dyn::new_placed(dir, m, d, None, st, &o, &experts, &Placing::default())
}

/// A layer's matrices and small tensors as its operations index them: a Mamba-2 layer's in and out products, its norm, conv1d's weight and bias, A_log, D,
/// dt_bias and the gated norm's weight; an attention layer's qkv and o products and its norm; a mixture-of-experts layer's norm (its experts are its block's).
pub(crate) fn per_layer(k: NhKind) -> (usize, usize) {
    match k {
        NhKind::Mamba => (2, 7),
        NhKind::Attention => (2, 1),
        NhKind::Moe => (0, 1),
    }
}

/// Nemotron-H's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::nemotron`, every tensor named), held
/// to `describe` below with the names left out (the family's own description names none); `c` the same config as the family's reader takes it, for its end
/// tokens and its context. What the loader runs.
pub fn lowered(dir: &Path, c: &crate::nemotron::ConfigNemotronH, st: &Stage) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    lowered_of(&kinds, &label, c, st)
}

/// `lowered` of the description `kinds` (named `label` in an error).
fn lowered_of(kinds: &Described, label: &str, c: &crate::nemotron::ConfigNemotronH, st: &Stage) -> Result<ModelDesc> {
    let lowered = super::lower::nemotron(kinds, c.eos.clone(), c.max_pos, st)?;
    super::lower::agree(super::lower::nameless(&lowered), &describe(c, st)?, label)?;
    Ok(lowered)
}

/// Nemotron-H's attention of heads of 128 (nq query heads over nk key-value heads, ring `ring`): no norms and no positional encoding, the keys and values as the
/// product gives them, scale 128^-0.5, the scores fp32.
pub(crate) fn attention(nq: usize, nk: usize, ring: usize) -> AttnOp {
    AttnOp {
        src: Slot::Mix,
        stride: (nq + 2 * nk) * 128,
        q_step: 128,
        k_off: nq * 128,
        v_off: Some((nq + nk) * 128),
        d: 128,
        nq,
        nk,
        writes: true,
        ring,
        window: 0,
        norm: QkNorm::None,
        v_norm: false,
        rope: None,
        q_scale: 1.0,
        scale: 128f32.powf(-0.5),
        softcap: 0.0,
        gate: None,
        round: false,
        sbf16: false,
        picked: false,
        sinks: None,
        v_scale: 1.0,
    }
}

/// The executor's description of Nemotron-H for stage `st`, its matrices, small tensors and blocks of experts indexed in the order `lower::nemotron` names them
/// (this description names none: the lowering is held to it with its names left out).
pub fn describe(c: &crate::nemotron::ConfigNemotronH, st: &Stage) -> Result<ModelDesc> {
    if c.m_head_dim != MAMBA2_HEAD || c.m_state != MAMBA2_STATE || c.conv_kernel != 4 || c.head_dim != 128 {
        return err(format!("Mamba-2 heads of {} channels with a state of {}, a convolution of {}, attention heads of {}: this engine runs heads of {MAMBA2_HEAD}, a state of {MAMBA2_STATE}, a convolution of 4 and attention heads of 128", c.m_head_dim, c.m_state, c.conv_kernel, c.head_dim));
    }
    let (hid, di, cd, row) = (c.hidden, c.inner(), c.conv_dim(), c.in_rows());
    let qkv_w = (c.heads + 2 * c.kv_heads) * c.head_dim;
    let (mut n_mats, mut n_small, mut n_moe, mut rings, mut states, mut layers) = (0, 0, 0, Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    for (l, &kind) in c.kinds.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let (b, s0) = (n_mats, n_small);
        let (m, n) = per_layer(kind);
        (n_mats, n_small) = (n_mats + m, n_small + n);
        let mut ops = Vec::new();
        if l == 0 {
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, weight: NormW::Plain, eps: c.eps });
        } else {
            ops.push(Op::NormAdd { y: Slot::O, w: s0, plus1: false });
        }
        match kind {
            NhKind::Mamba => {
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * cd * 2);
                states.push(c.m_heads * MAMBA2_HEAD * MAMBA2_STATE * 4);
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "Mamba-2: in product" },
                    Op::MambaConv { w: s0 + 1, bias: s0 + 2, state: conv, cd, row, x_off: di },
                    Op::MambaScan { dt_bias: s0 + 5, a_log: s0 + 3, d: s0 + 4, state: ssm, cd, row, dt_off: di + cd, nh: c.m_heads, ng: c.m_groups },
                    Op::MambaNorm { w: s0 + 6, row, di, ng: c.m_groups },
                    Op::Linear { w: b + 1, x: Slot::Core, y: Slot::O, name: "Mamba-2: out product" },
                ]);
            }
            NhKind::Attention => {
                let ring = rings.len();
                rings.push(Ring::kv(128, c.kv_heads, 0));
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                    Op::Attn(attention(c.heads, c.kv_heads, ring)),
                    Op::Linear { w: b + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                ]);
            }
            NhKind::Moe => {
                ops.push(Op::Moe(MoeOp::of(n_moe)));
                n_moe += 1;
            }
        }
        layers.push(ops);
    }
    let e = n_mats;
    let lm = if c.tie { e } else { e + 1 };
    let mut widths = [0usize; super::SLOTS];
    for (s, w) in [
        (Slot::X, hid),
        (Slot::H, hid),
        (Slot::O, hid),
        (Slot::Mix, row.max(qkv_w)),
        (Slot::Xc, cd),
        (Slot::Core, di),
        (Slot::Qo, c.heads * c.head_dim),
        (Slot::Att, c.heads * c.head_dim),
        (Slot::Hlast, hid),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Nemotron-H",
        hidden: hid,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        // (named by the lowering, `lower::nemotron`)
        mats: Vec::new(),
        small: Vec::new(),
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        // (a later stage never runs the first operation)
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, weight: NormW::Plain, eps: c.eps }),
        layers,
        states,
        final_norm: Some((n_small, Slot::O, false)),
        decoded: true,
        skinny: false,
        embed: e,
        lm,
        softcap: 0.0,
        eos: c.eos.clone(),
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
        host: Vec::new(),
        context: Vec::new(),
    })
}
