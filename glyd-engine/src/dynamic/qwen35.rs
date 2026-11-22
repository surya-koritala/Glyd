//! Qwen3.5 (dense: 9B, 27B; Qwen3.6-27B, Qwen3.8-27B) as a description: Gated DeltaNet layers and gated full-attention layers, the (1 + w) norms,
//! the SiLU MLP, the final norm on the last row (models/qwen3_5; the arithmetic is the kernels', as src/kern35.rs names them). Its weights load by
//! src/kern35.rs's loader (the small matrices it keeps dense, the fp32 parameters, the gated norm's weight as vLLM holds it), handed over as they are.

use super::{Dyn, ModelDesc, Op, Opts, Ring, Rope, Slot, Stage};
use crate::dev::Dev;
use crate::ffi::{err, Result};
use crate::kern35::{Mat, Weights35};
use crate::qwen35::Kind;
use std::path::Path;

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    let layers = crate::qwen35::Config35::load(dir)?.layers;
    load_stage(dir, o, Stage::whole(layers))
}

/// Stage `st` of it (a model split across GPUs: its layers, the embedding where it is the first, the head where the last; a layer of another
/// stage is an empty one here).
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    let dev = Dev::new(o.device)?;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let whole = st.first && st.last;
    let w = Weights35::load_stage(
        dir,
        o.ctx + 16,
        threads,
        o.layout,
        dev.cc_pair(),
        dev.total_mem as u64,
        None,
        (!whole).then_some((st.layers, st.first, st.last)),
    )?;
    let Weights35 { cfg: c, layout, layers: lys, embed, head, norm, cos, sin, .. } = w;
    let d = lowered(dir, &c, &st)?;
    // (the loader's matrices and small tensors, in the order `describe` indexes them: a layer's, then the final norm, the embedding and the head)
    let (mut mats, mut small) = (Vec::new(), Vec::new());
    for (l, ly) in lys.into_iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            continue;
        }
        let (m, n) = per_layer(ly.kind);
        if ly.mats.len() != m || ly.small.len() + 2 != n {
            return err(format!("layer {l}: {} matrices and {} small tensors loaded, where its kind has {m} and {n}", ly.mats.len(), ly.small.len() + 2));
        }
        mats.extend(ly.mats);
        small.push(ly.in_norm);
        small.push(ly.post_norm);
        small.extend(ly.small);
    }
    small.push(norm);
    mats.push(Mat::Packed(embed));
    if let Some(h) = head {
        mats.push(Mat::Packed(h));
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
    Dyn::from_loaded(dev, d, mats, small, Vec::new(), vec![(cos, sin)], layout, st, o)
}

/// A layer's matrices and small tensors as the loader holds them: a linear-attention layer's qkvz, ba and out products and its MLP's two; its input and
/// post-attention norms, the convolution, A_log, dt_bias and the gated norm's weight. A full-attention layer's qkv and o products and its MLP's two; its norms and q
/// and k norms.
pub(crate) fn per_layer(k: Kind) -> (usize, usize) {
    match k {
        Kind::Linear => (5, 6),
        Kind::Full => (4, 4),
    }
}

/// Qwen3.5's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::qwen35`); `c` the same config as the
/// family's reader takes it, for its end tokens and its context. What the loader runs; `describe` below is the description it is held to.
pub fn lowered(dir: &Path, c: &crate::qwen35::Config35, st: &Stage) -> Result<ModelDesc> {
    let p = dir.join("config.json");
    let text = std::fs::read_to_string(&p).map_err(|e| crate::ffi::Error(format!("{}: {e}", p.display())))?;
    let cfg: serde_json::Value = serde_json::from_str(&text).map_err(|e| crate::ffi::Error(format!("{}: {e}", p.display())))?;
    super::lower::qwen35(&crate::runtime::describe::describe(&cfg)?, c.eos.clone(), c.max_pos, st)
}

/// The executor's description of Qwen3.5's layout for stage `st` (src/kern35.rs's loader fills its matrices and small tensors; they are indexed here in its order).
pub fn describe(c: &crate::qwen35::Config35, st: &Stage) -> Result<ModelDesc> {
    if c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 {
        return err(format!(
            "head {}, linear heads {}/{}, convolution {}: this engine runs head 256, linear heads of 128 and a convolution of 4",
            c.head_dim, c.lin_k_dim, c.lin_v_dim, c.conv_kernel
        ));
    }
    let (nq, nk, nv) = (c.heads, c.kv_heads, c.lin_v_heads);
    let (cd, vd) = (c.conv_dim(), c.value_dim());
    let full_w = 2 * nq * 256 + 2 * nk * 256;
    let row = cd + vd;
    let (mut n_mats, mut n_small, mut rings, mut states, mut layers) = (0, 0, Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    for (l, &kind) in c.kinds.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            // (another stage's layer: nothing here, never run)
            layers.push(Vec::new());
            continue;
        }
        let (b, s0) = (n_mats, n_small);
        let (m, n) = per_layer(kind);
        (n_mats, n_small) = (n_mats + m, n_small + n);
        let mut ops = Vec::new();
        if l == 0 {
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, plus1: true });
        } else {
            ops.push(Op::NormAdd { y: Slot::D, w: s0, plus1: true });
        }
        let mlp = match kind {
            Kind::Linear => {
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * cd * 2);
                states.push(nv * 128 * 128 * 4);
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "linear attention: qkvz product" },
                    Op::Linear { w: b + 1, x: Slot::H, y: Slot::Ba, name: "linear attention: ba product" },
                    Op::GdnConv { w: s0 + 2, state: conv, cd, row },
                    Op::GdnDelta { a_log: s0 + 3, dt_bias: s0 + 4, state: ssm, cd, nk: c.lin_k_heads, nv },
                    Op::GatedNorm { w: s0 + 5, row, nv, z_off: cd },
                    Op::Linear { w: b + 2, x: Slot::Core, y: Slot::O, name: "linear attention: out product" },
                ]);
                b + 3
            }
            Kind::Full => {
                let ring = rings.len();
                rings.push(Ring { d: 256, nk, window: 0 });
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                    Op::Prep35 { stride: full_w, q_norm: s0 + 2, k_norm: s0 + 3, half: c.rotary_dim / 2, ring, nq, nk },
                    Op::Attn35 { stride: full_w, ring, nq, nk },
                    Op::Linear { w: b + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                ]);
                b + 2
            }
        };
        ops.extend([
            Op::NormAdd { y: Slot::O, w: s0 + 1, plus1: true },
            Op::GateUpSilu { w: mlp, inter: c.inter },
            Op::Linear { w: mlp + 1, x: Slot::M, y: Slot::D, name: "MLP: down product" },
        ]);
        layers.push(ops);
    }
    let final_norm = n_small;
    let e = n_mats;
    let lm = if c.tie { e } else { e + 1 };
    let mut widths = [0usize; super::SLOTS];
    let q_w = nq * 256;
    for (s, w) in [
        (Slot::X, c.hidden),
        (Slot::H, c.hidden),
        (Slot::Mix, row.max(full_w)),
        (Slot::Ba, 2 * nv),
        (Slot::Xc, cd),
        (Slot::Core, vd),
        (Slot::Qo, q_w),
        (Slot::Att, q_w),
        (Slot::O, c.hidden),
        (Slot::Gu, 2 * c.inter),
        (Slot::M, c.inter),
        (Slot::D, c.hidden),
        (Slot::Hlast, c.hidden),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Qwen3.5",
        hidden: c.hidden,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        // (the loader's own lists: src/kern35.rs names and packs the tensors)
        mats: Vec::new(),
        small: Vec::new(),
        ropes: vec![Rope { theta: c.rope_theta, dim: c.rotary_dim, pairs: c.rotary_dim / 2 }],
        rings,
        // (a later stage never runs the first operation)
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, plus1: true }),
        layers,
        states,
        final_norm: Some((final_norm, Slot::D, true)),
        decoded: true,
        // (the skinny kernels: 1.5% faster on the 4080 SUPER, their sums in another order; off until a quality check against the reference)
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
