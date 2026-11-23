//! Nemotron-H (`NemotronHForCausalLM`: Nemotron 3 Super, Nemotron 3.5 Lightning) as a description: Mamba-2 layers, attention layers without a positional encoding,
//! mixture-of-experts layers (with a latent, Super's; or at the model's width, Lightning's); every layer x + mixer(norm(x)) (the arithmetic is the kernels', as
//! kern_nh.rs runs them). Its weights load by kern_nh.rs's loader, the experts by moe_nh.rs's blocks, handed over as they are: a block with all its experts on the GPU
//! is run by the executor, one with an expert cache (or at the hummingbird level) runs its own launches over the executor's buffers (`NhHost`).

use super::attn::{AttnOp, QkNorm};
use super::{Dyn, ModelDesc, Op, Opts, Ring, Slot, Stage};
use crate::dev::Dev;
use crate::ffi::{err, Error, Result};
use crate::kern_nh::{MoeFactory, SsdWs, WeightsNh, NH_N, NH_P};
use crate::nemotron::NhKind;
use crate::weights::Layout;
use std::path::Path;

/// The model in `dir` with its matrices at `layout` and its experts at `level` (the experts all on the GPU).
pub fn load(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    let layers = crate::nemotron::ConfigNemotronH::load(dir)?.layers;
    load_stage(dir, o, level, Stage::whole(layers))
}

/// Stage `st` of it (a model split across GPUs: its layers, the embedding where it is the first, the head where the last; a layer of another
/// stage is an empty one here).
pub fn load_stage(dir: &Path, o: &Opts, level: Layout, st: Stage) -> Result<Dyn> {
    load_with(dir, o, &crate::moe_nh::factory(level), st)
}

/// Stage `st` with the blocks of experts `make` makes (moe_nh.rs's factories: the experts all on the GPU, or an expert cache's places for them, src/offload.rs).
pub fn load_with(dir: &Path, o: &Opts, make: MoeFactory<'_>, st: Stage) -> Result<Dyn> {
    let dev = Dev::new(o.device)?;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let only: Vec<usize> = (st.layers.0..st.layers.1).collect();
    let whole = st.first && st.last;
    let w = WeightsNh::load_ends(dir, threads, o.layout, dev.cc_pair(), dev.total_mem as u64, (!whole).then_some(&only[..]), (st.first, st.last), Some(make))?;
    let WeightsNh { cfg: c, layout, layers: lys, embed, head, norm_f, .. } = w;
    let d = lowered(dir, &c, &st)?;
    // (the loader's matrices, small tensors and blocks of experts in the order `describe` indexes them: a layer's, then the final norm, the embedding and the head)
    let (mut mats, mut small, mut moe) = (Vec::new(), Vec::new(), Vec::new());
    for (l, ly) in lys.into_iter().enumerate() {
        let Some(ly) = ly else { continue };
        let (m, n) = per_layer(ly.kind);
        if ly.mats.len() != m || ly.small.len() + 1 != n || ly.moe.is_some() != (ly.kind == NhKind::Moe) {
            return err(format!(
                "layer {l}: {} matrices, {} small tensors and a block of experts {} loaded, where its kind has {m} and {n}",
                ly.mats.len(),
                ly.small.len() + 1,
                ly.moe.is_some()
            ));
        }
        mats.extend(ly.mats);
        small.push(ly.norm);
        small.extend(ly.small);
        if let Some(b) = ly.moe {
            moe.push(b);
        }
    }
    // (a stage without the head or the embedding keeps their places, empty, so that their indices hold)
    small.push(match norm_f {
        Some(b) => b,
        None if !st.last => crate::dev::Buf::new(4)?,
        None => return err("the final norm is not loaded"),
    });
    mats.push(match embed {
        Some(p) => crate::kern35::Mat::Packed(p),
        None if !st.first && !(st.last && c.tie) => crate::kern35::Mat::Dense(crate::dev::Buf::new(2)?, 0, 0),
        None => return err("the embedding is not loaded"),
    });
    if let Some(h) = head {
        mats.push(crate::kern35::Mat::Packed(h));
    }
    if (mats.len(), small.len()) != (d.lm + 1, d.final_norm.map_or(0, |f| f.0 + 1)) {
        return err(format!(
            "{} matrices and {} small tensors loaded, where the description indexes {} and {}",
            mats.len(),
            small.len(),
            d.lm + 1,
            d.final_norm.map_or(0, |f| f.0 + 1)
        ));
    }
    let ch = o.chunk.max(1);
    let mut dy = Dyn::from_loaded(dev, d, mats, small, moe, Vec::new(), layout, st, o)?;
    dy.ssd = Some(SsdWs::new(&c, ch)?);
    Ok(dy)
}

/// A layer's matrices and small tensors as the loader holds them: a Mamba-2 layer's in and out products, its norm, conv1d's weight and bias, A_log, D, dt_bias
/// and the gated norm's weight; an attention layer's qkv and o products and its norm; a mixture-of-experts layer's norm (its experts are its block's).
pub(crate) fn per_layer(k: NhKind) -> (usize, usize) {
    match k {
        NhKind::Mamba => (2, 7),
        NhKind::Attention => (2, 1),
        NhKind::Moe => (0, 1),
    }
}

/// Nemotron-H's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::nemotron`); `c` the same config as the
/// family's reader takes it, for its end tokens and its context. What the loader runs; `describe` below is the description it is held to.
pub fn lowered(dir: &Path, c: &crate::nemotron::ConfigNemotronH, st: &Stage) -> Result<ModelDesc> {
    let p = dir.join("config.json");
    let text = std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let cfg: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let lowered = super::lower::nemotron(&crate::runtime::describe::describe(&cfg)?, c.eos.clone(), c.max_pos, st)?;
    super::lower::agree(lowered, &describe(c, st)?, &p.display().to_string())
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
    }
}

/// The executor's description of Nemotron-H for stage `st` (kern_nh.rs's loader fills its matrices, small tensors and blocks of experts; they are indexed here in
/// its order).
pub fn describe(c: &crate::nemotron::ConfigNemotronH, st: &Stage) -> Result<ModelDesc> {
    if c.m_head_dim != NH_P || c.m_state != NH_N || c.conv_kernel != 4 || c.head_dim != 128 {
        return err(format!("Mamba-2 heads of {} channels with a state of {}, a convolution of {}, attention heads of {}: this engine runs heads of {NH_P}, a state of {NH_N}, a convolution of 4 and attention heads of 128", c.m_head_dim, c.m_state, c.conv_kernel, c.head_dim));
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
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, plus1: false });
        } else {
            ops.push(Op::NormAdd { y: Slot::O, w: s0, plus1: false });
        }
        match kind {
            NhKind::Mamba => {
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * cd * 2);
                states.push(c.m_heads * NH_P * NH_N * 4);
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
                rings.push(Ring { d: 128, nk: c.kv_heads, window: 0 });
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                    Op::Attn(attention(c.heads, c.kv_heads, ring)),
                    Op::Linear { w: b + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                ]);
            }
            NhKind::Moe => {
                ops.push(Op::Moe { block: n_moe });
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
        // (the loader's own lists: kern_nh.rs names and packs the tensors)
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
        rings,
        // (a later stage never runs the first operation)
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, plus1: false }),
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
    })
}
