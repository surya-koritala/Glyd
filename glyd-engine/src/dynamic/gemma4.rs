//! Gemma 4 (dense: 12B, 31B) as a description: sliding and global attention layers, the sandwich norms, the per-layer scalar, gelu(tanh), the tied
//! and soft-capped head (models/gemma4/modeling_gemma4.py; the arithmetic is the kernels', as src/gemma4.rs names them).

use super::attn::{AttnOp, QkNorm, RopeAt};
use super::{ModelDesc, Op, Ring, Rope, Slot};
use crate::gemma4::{ConfigG4, KindG4};

/// Gemma 4's attention of a layer of head `d`, `nk` key-value heads (the values the keys' projection where !`has_v`): q and k normed with weights `q_norm`, `k_norm`
/// and each rounding of transformers' bf16 arithmetic, the values normed without one, the rotary of table 0 (sliding) or 1 (global) on `pairs` of the head's
/// half pairs, scale 1, scores the bf16 of their dot products. `writes`: a draft model's layers read the big model's ring `ring` and write none.
pub(crate) fn attention(c: &ConfigG4, kind: KindG4, nq: usize, q_norm: usize, k_norm: usize, ring: usize, writes: bool) -> Op {
    let (d, nk, has_v) = c.attn_of(kind);
    let sliding = kind == KindG4::Sliding;
    Op::Attn(AttnOp {
        src: Slot::Proj,
        stride: if writes { c.proj_width(kind) } else { nq * d },
        q_step: d,
        k_off: nq * d,
        v_off: (writes && has_v).then_some((nq + nk) * d),
        d,
        nq,
        nk,
        writes,
        ring,
        window: if sliding { c.window } else { 0 },
        norm: QkNorm::Weight { q: q_norm, k: k_norm },
        v_norm: writes,
        rope: Some(RopeAt { table: if sliding { 0 } else { 1 }, span: d, pairs: if sliding { c.head_dim / 2 } else { c.global_rot } }),
        q_scale: 1.0,
        scale: 1.0,
        softcap: 0.0,
        gate: None,
        round: true,
        sbf16: true,
        picked: None,
    })
}

/// Gemma 4's executor description from the config.json in `dir` (R3a): the layer kinds the description builder reads from it, lowered to the executor's operations
/// (`lower::gemma4`) and sized by the shapes it gives; `c` the same config as the family's reader takes it, for its end tokens and its context. What every loader of
/// the executor's Gemma 4 runs; `describe` below is the description it is held to.
pub fn lowered(dir: &std::path::Path, c: &ConfigG4) -> crate::ffi::Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let lowered = super::lower::gemma4(&kinds, &|n: &str| kinds.shapes.get(n).cloned(), c.eos.clone(), c.max_pos)?;
    super::lower::agree(lowered, &describe(c), &label)
}

pub fn describe(c: &ConfigG4) -> ModelDesc {
    let p = &c.prefix;
    let mats = c.items();
    // (items(): per layer the projection, o, gate and up, down; then the embedding, the tied head)
    let embed = mats.len() - 1;
    let mut small: Vec<String> = Vec::new();
    let mut name = |n: String| -> usize {
        small.push(n);
        small.len() - 1
    };
    let ropes = vec![
        Rope { theta: c.sliding_theta, dim: c.head_dim, pairs: c.head_dim / 2 },
        Rope { theta: c.global_theta, dim: c.global_head_dim, pairs: c.global_rot },
    ];
    let mut rings = Vec::new();
    let in_norms: Vec<usize> = (0..c.layers).map(|l| name(format!("{p}layers.{l}.input_layernorm.weight"))).collect();
    let final_norm = name(format!("{p}norm.weight"));
    let mut layers = Vec::with_capacity(c.layers);
    for (l, &kind) in c.kinds.iter().enumerate() {
        let pl = format!("{p}layers.{l}");
        let (d, nk, _) = c.attn_of(kind);
        let sliding = kind == KindG4::Sliding;
        let ring = rings.len();
        rings.push(Ring { d, nk, window: if sliding { c.window } else { 0 }, values: true });
        let (q_norm, k_norm) = (name(format!("{pl}.self_attn.q_norm.weight")), name(format!("{pl}.self_attn.k_norm.weight")));
        let (post_attn, pre_ff) = (name(format!("{pl}.post_attention_layernorm.weight")), name(format!("{pl}.pre_feedforward_layernorm.weight")));
        let (post_ff, scalar) = (name(format!("{pl}.post_feedforward_layernorm.weight")), name(format!("{pl}.layer_scalar")));
        let next = if l + 1 < c.layers { in_norms[l + 1] } else { final_norm };
        let w = 4 * l;
        layers.push(vec![
            Op::Linear { w, x: Slot::H, y: Slot::Proj, name: if sliding { "qkv (sliding)" } else { "qk (global)" } },
            attention(c, kind, c.heads, q_norm, k_norm, ring, true),
            Op::Linear { w: w + 1, x: Slot::Att, y: Slot::O, name: "o" },
            Op::AddNorm { y: Slot::O, post: post_attn, scalar: None, next: pre_ff },
            Op::Linear { w: w + 2, x: Slot::H, y: Slot::Gu, name: "gate_up" },
            Op::GeluMul { inter: c.inter },
            Op::Linear { w: w + 3, x: Slot::M, y: Slot::D, name: "down" },
            Op::AddNorm { y: Slot::D, post: post_ff, scalar: Some(scalar), next },
        ]);
    }
    let proj_w = c.kinds.iter().map(|&k| c.proj_width(k)).max().unwrap_or(0);
    let q_w = c.heads * c.global_head_dim;
    let mut widths = [0usize; super::SLOTS];
    for (s, w) in [
        (Slot::X, c.hidden),
        (Slot::H, c.hidden),
        (Slot::Proj, proj_w),
        (Slot::Qo, q_w),
        (Slot::Att, q_w),
        (Slot::O, c.hidden),
        (Slot::Gu, 2 * c.inter),
        (Slot::M, c.inter),
        (Slot::D, c.hidden),
    ] {
        widths[s as usize] = w;
    }
    ModelDesc {
        family: "Gemma 4",
        hidden: c.hidden,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats,
        small,
        ropes,
        rings,
        first: Op::EmbedNorm { scale: c.embed_scale(), norm: in_norms[0] },
        layers,
        states: Vec::new(),
        final_norm: None,
        decoded: false,
        skinny: true,
        embed,
        lm: embed,
        softcap: c.softcap,
        eos: c.eos.clone(),
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    }
}

/// A Gemma 4 "-assistant" draft model (`dc`, its config) of the big model `c`: its input the drafted-from token's scaled embedding and the big model's last
/// hidden state, projected down (pre); layers whose queries read the big model's last sliding and last global layers' keys and values (no keys of its own);
/// its tied head, and the hidden state projected back up (post) for the next draft. Rings and rotary tables are the big model's (its description's indices).
pub fn describe_draft(dc: &ConfigG4, c: &ConfigG4) -> ModelDesc {
    let p = &dc.prefix;
    let mut mats = Vec::new();
    for (l, &k) in dc.kinds.iter().enumerate() {
        let (a, d) = (format!("{p}layers.{l}.self_attn"), dc.attn_of(k).0);
        mats.push(crate::weights::Item { members: vec![format!("{a}.q_proj.weight")], rows: dc.heads * d, cols: dc.hidden });
        mats.push(crate::weights::Item { members: vec![format!("{a}.o_proj.weight")], rows: dc.hidden, cols: dc.heads * d });
        let mlp = format!("{p}layers.{l}.mlp");
        mats.push(crate::weights::Item {
            members: vec![format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")],
            rows: 2 * dc.inter,
            cols: dc.hidden,
        });
        mats.push(crate::weights::Item { members: vec![format!("{mlp}.down_proj.weight")], rows: dc.hidden, cols: dc.inter });
    }
    let (pre, post, embed) = (mats.len(), mats.len() + 1, mats.len() + 2);
    mats.push(crate::weights::Item { members: vec!["pre_projection.weight".into()], rows: dc.hidden, cols: 2 * c.hidden });
    mats.push(crate::weights::Item { members: vec!["post_projection.weight".into()], rows: c.hidden, cols: dc.hidden });
    mats.push(crate::weights::Item { members: vec![format!("{p}embed_tokens.weight")], rows: dc.vocab, cols: dc.hidden });
    let mut small: Vec<String> = Vec::new();
    let mut name = |n: String| -> usize {
        small.push(n);
        small.len() - 1
    };
    let in_norms: Vec<usize> = (0..dc.layers).map(|l| name(format!("{p}layers.{l}.input_layernorm.weight"))).collect();
    let final_norm = name(format!("{p}norm.weight"));
    // (the big model's last layer of each kind: its ring is the one the draft model's queries read)
    let ring_of = |k: KindG4| c.kinds.iter().rposition(|&x| x == k).expect("a layer of each kind");
    let mut layers = Vec::with_capacity(dc.layers);
    for (l, &kind) in dc.kinds.iter().enumerate() {
        let pl = format!("{p}layers.{l}");
        let q_norm = name(format!("{pl}.self_attn.q_norm.weight"));
        let (post_attn, pre_ff) = (name(format!("{pl}.post_attention_layernorm.weight")), name(format!("{pl}.pre_feedforward_layernorm.weight")));
        let (post_ff, scalar) = (name(format!("{pl}.post_feedforward_layernorm.weight")), name(format!("{pl}.layer_scalar")));
        let next = if l + 1 < dc.layers { in_norms[l + 1] } else { final_norm };
        let w = 4 * l;
        let mut ops = Vec::new();
        if l == 0 {
            ops.push(Op::Linear { w: pre, x: Slot::In, y: Slot::X, name: "draft" });
            ops.push(Op::Norm { x: Slot::X, w: in_norms[0], y: Slot::H });
        }
        ops.extend([
            Op::Linear { w, x: Slot::H, y: Slot::Proj, name: "draft" },
            attention(c, kind, dc.heads, q_norm, q_norm, ring_of(kind), false),
            Op::Linear { w: w + 1, x: Slot::Att, y: Slot::O, name: "draft" },
            Op::AddNorm { y: Slot::O, post: post_attn, scalar: None, next: pre_ff },
            Op::Linear { w: w + 2, x: Slot::H, y: Slot::Gu, name: "draft" },
            Op::GeluMul { inter: dc.inter },
            Op::Linear { w: w + 3, x: Slot::M, y: Slot::D, name: "draft" },
            Op::AddNorm { y: Slot::D, post: post_ff, scalar: Some(scalar), next },
        ]);
        layers.push(ops);
    }
    let qd = dc.heads * c.global_head_dim.max(c.head_dim);
    let mut widths = [0usize; super::SLOTS];
    for (s, w) in [
        (Slot::X, dc.hidden),
        (Slot::H, dc.hidden),
        (Slot::Proj, qd),
        (Slot::Qo, qd),
        (Slot::Att, qd),
        (Slot::O, dc.hidden),
        (Slot::Gu, 2 * dc.inter),
        (Slot::M, dc.inter),
        (Slot::D, dc.hidden),
        (Slot::In, 2 * c.hidden),
        (Slot::Hid, c.hidden),
    ] {
        widths[s as usize] = w;
    }
    ModelDesc {
        family: "Gemma 4 draft",
        hidden: dc.hidden,
        vocab: dc.vocab,
        eps: dc.eps,
        max_pos: c.max_pos,
        widths,
        mats,
        small,
        ropes: Vec::new(),
        rings: Vec::new(),
        first: Op::DraftIn { scale: c.embed_scale() },
        layers,
        states: Vec::new(),
        final_norm: None,
        decoded: false,
        skinny: true,
        embed,
        lm: embed,
        softcap: 0.0,
        eos: c.eos.clone(),
        post: Some(post),
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    }
}
