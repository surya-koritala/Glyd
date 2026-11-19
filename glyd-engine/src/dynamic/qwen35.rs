//! Qwen3.5 (dense: 9B, 27B; Qwen3.6-27B, Qwen3.8-27B) as a description: Gated DeltaNet layers and gated full-attention layers, the (1 + w) norms,
//! the SiLU MLP, the final norm on the last row (models/qwen3_5; the arithmetic is the kernels', as model35.rs runs them). Its weights load by
//! model35.rs's loader (the small matrices it keeps dense, the fp32 parameters, the gated norm's weight as vLLM holds it), handed over as they are.

use super::{Dyn, ModelDesc, Op, Opts, Ring, Rope, Slot, Stage};
use crate::dev::Dev;
use crate::ffi::{err, Result};
use crate::model35::{Mat, Weights35};
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
    let w = Weights35::load_stage(dir, o.ctx + 16, threads, o.layout, dev.cc_pair(), dev.total_mem as u64, None, (!whole).then_some((st.layers, st.first, st.last)))?;
    let Weights35 { cfg: c, layout, layers: lys, embed, head, norm, cos, sin, .. } = w;
    if c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 {
        return err(format!("head {}, linear heads {}/{}, convolution {}: this engine runs head 256, linear heads of 128 and a convolution of 4", c.head_dim, c.lin_k_dim, c.lin_v_dim, c.conv_kernel));
    }
    let (nq, nk, nv) = (c.heads, c.kv_heads, c.lin_v_heads);
    let (cd, vd) = (c.conv_dim(), c.value_dim());
    let full_w = 2 * nq * 256 + 2 * nk * 256;
    let row = cd + vd;
    let (mut mats, mut small, mut rings, mut states, mut layers) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    for (l, ly) in lys.into_iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            // (another stage's layer: nothing here, never run)
            layers.push(Vec::new());
            continue;
        }
        let b = mats.len();
        mats.extend(ly.mats);
        let s0 = small.len();
        small.push(ly.in_norm);
        small.push(ly.post_norm);
        small.extend(ly.small);
        let mut ops = Vec::new();
        if l == 0 {
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, plus1: true });
        } else {
            ops.push(Op::NormAdd { y: Slot::D, w: s0, plus1: true });
        }
        let mlp = match ly.kind {
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
    let final_norm = small.len();
    small.push(norm);
    let e = mats.len();
    mats.push(Mat::Packed(embed));
    let lm = match head {
        Some(h) => {
            mats.push(Mat::Packed(h));
            e + 1
        }
        None => e,
    };
    let mut widths = [0usize; super::SLOTS];
    let q_w = nq * 256;
    for (s, w) in [(Slot::X, c.hidden), (Slot::H, c.hidden), (Slot::Mix, row.max(full_w)), (Slot::Ba, 2 * nv), (Slot::Xc, cd), (Slot::Core, vd), (Slot::Qo, q_w), (Slot::Att, q_w), (Slot::O, c.hidden), (Slot::Gu, 2 * c.inter), (Slot::M, c.inter), (Slot::D, c.hidden), (Slot::Hlast, c.hidden)] {
        widths[s as usize] = w;
    }
    let d = ModelDesc {
        family: "Qwen3.5",
        hidden: c.hidden,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        // (the loader's own lists: model35.rs names and packs the tensors)
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
    };
    Dyn::from_loaded(dev, d, mats, small, Vec::new(), vec![(cos, sin)], layout, st, o)
}
