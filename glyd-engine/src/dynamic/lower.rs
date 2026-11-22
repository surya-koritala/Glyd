//! A model's description in layer kinds (`runtime::describe`: what a config.json says) lowered to the executor's operations (`ModelDesc`), so that a family runs from
//! its config alone and not from a description written for it (R2, R3a; docs/design/runtime.md section 3). The matrices' shapes come from the checkpoint (`shape`: a
//! tensor's dims by name), which the kinds do not hold. Each family's lowering is held to the executor's own description of it (the tests below) until that one goes.

use super::{ModelDesc, Op, Ring, Rope, Slot, SLOTS};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{Body, Ffn, Mixer};
use crate::weights::Item;

/// A tensor's dims by its name in the checkpoint (None: not there).
pub type Shape<'a> = &'a dyn Fn(&str) -> Option<Vec<usize>>;

/// The names of a block's role (one tensor, or rows stacked).
fn role<'a>(roles: &'a [(String, Source)], r: &str) -> Option<&'a Source> {
    roles.iter().find(|(n, _)| n == r).map(|(_, s)| s)
}

/// The matrix of `members` stacked by rows: its rows, its columns (every member's the same).
fn item(members: Vec<String>, shape: Shape) -> Result<Item> {
    let mut rows = 0;
    let mut cols = None;
    for m in &members {
        let s = shape(m).ok_or_else(|| Error(format!("the checkpoint has no tensor {m}")))?;
        if s.len() != 2 || cols.is_some_and(|c| c != s[1]) {
            return err(format!("{m}: {s:?}, where the matrix it is stacked in has {:?} columns", cols));
        }
        rows += s[0];
        cols = Some(s[1]);
    }
    Ok(Item { members, rows, cols: cols.unwrap_or(0) })
}

/// `x` rounded to the nearest bf16 (ties to even).
fn bf16_round(x: f32) -> f32 {
    f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) & 0xffff_0000)
}

/// Gemma 4 (dense): pairs of blocks, sliding or global attention with sandwich norms, then the gated MLP with its sandwich norms and the per-layer scalar; the
/// embedding scaled and normed into the first layer, the tied and soft-capped head. `eos` and `max_pos` are the generation config's and the config's.
pub fn gemma4(d: &Described, shape: Shape, eos: Vec<i64>, max_pos: usize) -> Result<ModelDesc> {
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("Gemma 4: {} blocks, where each layer is an attention block and an MLP block", m.blocks.len()));
    }
    let layers = m.blocks.len() / 2;
    let one = |roles: &[(String, Source)], r: &str| -> Result<String> {
        match role(roles, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("Gemma 4: the role {r} is {other:?}, where one tensor is")),
        }
    };
    let mut small: Vec<String> = Vec::new();
    let mut name = |n: String| -> usize {
        small.push(n);
        small.len() - 1
    };
    // (the input norms of every layer first, then the final norm, as the executor's description of Gemma 4 holds them)
    let in_norms: Vec<usize> = (0..layers).map(|l| one(&d.blocks[2 * l], "norm_in").map(&mut name)).collect::<Result<_>>()?;
    let final_norm = name(d.final_norm.clone());
    let (mut mats, mut rings, mut ropes, mut ops) = (Vec::new(), Vec::new(), Vec::<Rope>::new(), Vec::new());
    let (mut sliding_rope, mut global_rope) = (None, None);
    let (mut proj_w, mut q_w, mut inter_w) = (0, 0, 0);
    for l in 0..layers {
        let (ab, fb) = (&m.blocks[2 * l], &m.blocks[2 * l + 1]);
        let (ar, fr) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        let a = match &ab.body {
            Body::Mixer(Mixer::Attention(a)) => a,
            b => return err(format!("Gemma 4: layer {l}'s first block is {b:?}, not attention")),
        };
        if !matches!(fb.body, Body::Ffn(Ffn::Mlp(_))) {
            return err(format!("Gemma 4: layer {l}'s second block is {:?}, not an MLP", fb.body));
        }
        let rope = a.rope.as_ref().ok_or_else(|| Error(format!("Gemma 4: layer {l} has no rotary")))?;
        let sliding = a.window.is_some();
        let slot = if sliding { &mut sliding_rope } else { &mut global_rope };
        let r = *slot.get_or_insert_with(|| {
            ropes.push(Rope { theta: rope.theta, dim: rope.dim, pairs: rope.freqs });
            ropes.len() - 1
        });
        let (dh, nk, has_v) = (a.head_dim, a.kv_heads, !a.k_eq_v);
        let ring = rings.len();
        rings.push(Ring { d: dh, nk, window: a.window.unwrap_or(0) });
        let mut qkv = vec![one(ar, "q")?, one(ar, "k")?];
        if has_v {
            qkv.push(one(ar, "v")?);
        }
        let w = mats.len();
        let qkv = item(qkv, shape)?;
        let stride = qkv.rows;
        mats.push(qkv);
        mats.push(item(vec![one(ar, "o")?], shape)?);
        let gate_up = item(vec![one(fr, "gate")?, one(fr, "up")?], shape)?;
        let inter = gate_up.rows / 2;
        mats.push(gate_up);
        mats.push(item(vec![one(fr, "down")?], shape)?);
        let (q_norm, k_norm) = (name(one(ar, "q_norm")?), name(one(ar, "k_norm")?));
        let (post_attn, pre_ff) = (name(one(ar, "norm_out")?), name(one(fr, "norm_in")?));
        let (post_ff, scalar) = (name(one(fr, "norm_out")?), name(one(fr, "res_scale")?));
        let next = if l + 1 < layers { in_norms[l + 1] } else { final_norm };
        let window = a.window.unwrap_or(0);
        ops.push(vec![
            Op::Linear { w, x: Slot::H, y: Slot::Proj, name: if sliding { "qkv (sliding)" } else { "qk (global)" } },
            Op::QkvPrep { d: dh, nq: a.heads, nk, has_v, stride, q_norm, k_norm, rope: r, rot: rope.freqs, ring },
            Op::Attention { d: dh, nq: a.heads, nk, window, ring },
            Op::Linear { w: w + 1, x: Slot::Att, y: Slot::O, name: "o" },
            Op::AddNorm { y: Slot::O, post: post_attn, scalar: None, next: pre_ff },
            Op::Linear { w: w + 2, x: Slot::H, y: Slot::Gu, name: "gate_up" },
            Op::GeluMul { inter },
            Op::Linear { w: w + 3, x: Slot::M, y: Slot::D, name: "down" },
            Op::AddNorm { y: Slot::D, post: post_ff, scalar: Some(scalar), next },
        ]);
        (proj_w, q_w, inter_w) = (proj_w.max(stride), q_w.max(a.heads * dh), inter_w.max(inter));
    }
    // (the executor's description holds the sliding table first, then the global one; a model of one kind of layer has the one)
    let embed = mats.len();
    mats.push(item(vec![d.embed.clone()], shape)?);
    let mut widths = [0usize; SLOTS];
    for (s, w) in [
        (Slot::X, m.hidden),
        (Slot::H, m.hidden),
        (Slot::Proj, proj_w),
        (Slot::Qo, q_w),
        (Slot::Att, q_w),
        (Slot::O, m.hidden),
        (Slot::Gu, 2 * inter_w),
        (Slot::M, inter_w),
        (Slot::D, m.hidden),
    ] {
        widths[s as usize] = w;
    }
    if d.head.is_some() || !m.head.tied {
        return err("Gemma 4: a head of its own, where the executor's Gemma 4 ties it to the embedding");
    }
    let eps = m.blocks.first().map_or(1e-6, |b| b.norm_in.eps);
    Ok(ModelDesc {
        family: "Gemma 4",
        hidden: m.hidden,
        vocab: m.vocab,
        eps,
        max_pos,
        widths,
        mats,
        small,
        ropes,
        rings,
        // (the scale in the activations' type, bf16, as transformers casts it: 62.0 for sqrt(3840), where the f32 reference keeps 61.97)
        first: Op::EmbedNorm { scale: bf16_round(m.embed.scale), norm: in_norms.first().copied().ok_or_else(|| Error("Gemma 4: no layers".into()))? },
        layers: ops,
        states: Vec::new(),
        final_norm: None,
        decoded: false,
        skinny: true,
        embed,
        lm: embed,
        softcap: m.head.softcap.unwrap_or(0.0),
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    })
}

/// Qwen3.5's layout (dense: Qwen3.5, 3.6 and 3.8 of 256-wide heads): pairs of blocks, a Gated DeltaNet or a gated full attention, then the SiLU MLP, the (1 + w)
/// norms, the final norm on the last row. The matrices and small tensors are src/kern35.rs's loader's (`dynamic::qwen35::describe` indexes them in its order);
/// stage `st` of a model split across GPUs.
pub fn qwen35(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::qwen35::per_layer;
    use crate::qwen35::Kind;
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("Qwen3.5: {} blocks, where each layer is a mixer block and an MLP block", m.blocks.len()));
    }
    let layers = m.blocks.len() / 2;
    let (mut lin, mut att) = (None, None);
    let mut kinds = Vec::with_capacity(layers);
    for l in 0..layers {
        if !matches!(m.blocks[2 * l + 1].body, Body::Ffn(Ffn::Mlp(_))) {
            return err(format!("Qwen3.5: layer {l}'s second block is {:?}, not an MLP (a mixture of experts is not lowered yet)", m.blocks[2 * l + 1].body));
        }
        match &m.blocks[2 * l].body {
            Body::Mixer(Mixer::Linear(x)) => {
                lin.get_or_insert(x);
                kinds.push(Kind::Linear);
            }
            Body::Mixer(Mixer::Attention(a)) => {
                att.get_or_insert(a);
                kinds.push(Kind::Full);
            }
            b => return err(format!("Qwen3.5: layer {l}'s first block is {b:?}")),
        }
    }
    let (lin, att) =
        (lin.ok_or_else(|| Error("Qwen3.5: no linear-attention layer".into()))?, att.ok_or_else(|| Error("Qwen3.5: no full-attention layer".into()))?);
    if att.head_dim != 256 || lin.k_dim != 128 || lin.v_dim != 128 || lin.conv != 4 {
        return err(format!(
            "head {}, linear heads {}/{}, convolution {}: this engine runs head 256, linear heads of 128 and a convolution of 4",
            att.head_dim, lin.k_dim, lin.v_dim, lin.conv
        ));
    }
    let rope = att.rope.as_ref().ok_or_else(|| Error("Qwen3.5: attention with no rotary".into()))?;
    let rotary = 2 * rope.freqs;
    let gate = role(&d.blocks[1], "gate").ok_or_else(|| Error("Qwen3.5: no MLP gate".into()))?;
    let gate = match gate {
        Source::Tensor(n) => n,
        g => return err(format!("Qwen3.5: an MLP gate of {g:?}")),
    };
    let inter = d.shapes.get(gate).map(|s| s[0]).ok_or_else(|| Error(format!("Qwen3.5: no shape for {gate}")))?;
    let (nq, nk, nv) = (att.heads, att.kv_heads, lin.v_heads);
    let (vd, cd) = (nv * lin.v_dim, 2 * lin.k_heads * lin.k_dim + nv * lin.v_dim);
    let full_w = 2 * nq * 256 + 2 * nk * 256;
    let row = cd + vd;
    let (mut n_mats, mut n_small, mut rings, mut states, mut ops_all) = (0, 0, Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    for (l, &kind) in kinds.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            ops_all.push(Vec::new());
            continue;
        }
        let (b, s0) = (n_mats, n_small);
        let (mm, nn) = per_layer(kind);
        (n_mats, n_small) = (n_mats + mm, n_small + nn);
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
                    Op::GdnDelta { a_log: s0 + 3, dt_bias: s0 + 4, state: ssm, cd, nk: lin.k_heads, nv },
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
                    Op::Prep35 { stride: full_w, q_norm: s0 + 2, k_norm: s0 + 3, half: rotary / 2, ring, nq, nk },
                    Op::Attn35 { stride: full_w, ring, nq, nk },
                    Op::Linear { w: b + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                ]);
                b + 2
            }
        };
        ops.extend([
            Op::NormAdd { y: Slot::O, w: s0 + 1, plus1: true },
            Op::GateUpSilu { w: mlp, inter },
            Op::Linear { w: mlp + 1, x: Slot::M, y: Slot::D, name: "MLP: down product" },
        ]);
        ops_all.push(ops);
    }
    let e = n_mats;
    let lm = if d.head.is_none() { e } else { e + 1 };
    let q_w = nq * 256;
    let mut widths = [0usize; SLOTS];
    for (s, w) in [
        (Slot::X, m.hidden),
        (Slot::H, m.hidden),
        (Slot::Mix, row.max(full_w)),
        (Slot::Ba, 2 * nv),
        (Slot::Xc, cd),
        (Slot::Core, vd),
        (Slot::Qo, q_w),
        (Slot::Att, q_w),
        (Slot::O, m.hidden),
        (Slot::Gu, 2 * inter),
        (Slot::M, inter),
        (Slot::D, m.hidden),
        (Slot::Hlast, m.hidden),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Qwen3.5",
        hidden: m.hidden,
        vocab: m.vocab,
        eps: m.blocks.first().map_or(1e-6, |b| b.norm_in.eps),
        max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: vec![Rope { theta: rope.theta, dim: rotary, pairs: rotary / 2 }],
        rings,
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, plus1: true }),
        layers: ops_all,
        states,
        final_norm: Some((n_small, Slot::D, true)),
        decoded: true,
        skinny: false,
        embed: e,
        lm,
        softcap: 0.0,
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tensors' shapes and the config of a header fixture (tests/fixtures/headers: the Hub's safetensors headers).
    fn header(name: &str) -> (String, std::collections::HashMap<String, Vec<usize>>) {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers").join(name);
        let h: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let mut shapes = std::collections::HashMap::new();
        for g in h["groups"].as_array().unwrap() {
            let mask = g["mask"].as_str().unwrap();
            let dims: Vec<usize> = g["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let layers: Vec<String> = g["layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap().to_string()).collect();
            let ls = if mask.contains("{L}") { layers } else { vec![String::new()] };
            for l in &ls {
                shapes.insert(mask.replace("{L}", l), dims.clone());
            }
        }
        (h["config"].as_str().unwrap().to_string(), shapes)
    }

    /// Qwen3.5's layout lowered from each dense release's config (Qwen3.5 2B to 27B, Qwen3.6-27B, Qwen3.8-27B: those of 256-wide heads the executor runs), whole and
    /// as a split's stages, is the executor's own description of it.
    #[test]
    fn qwen35_lowered_is_the_executors_own_description() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_configs.json");
        let all: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let mut seen = 0;
        for (repo, cfg) in all.as_object().unwrap() {
            // (the mixtures of experts are refused by the family's reader, and not lowered yet)
            let Ok(c) = crate::qwen35::Config35::parse(&cfg.to_string(), repo, None) else { continue };
            if c.head_dim != 256 {
                continue;
            }
            let kinds = crate::runtime::describe::describe(cfg).unwrap();
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, c.layers / 2), first: true, last: false },
                super::super::Stage { layers: (c.layers / 2, c.layers), first: false, last: true },
            ] {
                let own = super::super::qwen35::describe(&c, &st).unwrap();
                let lowered = qwen35(&kinds, c.eos.clone(), c.max_pos, &st).unwrap_or_else(|e| panic!("{repo}: {}", e.msg));
                let (a, b) = (format!("{own:#?}"), format!("{lowered:#?}"));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("{repo}: the lowered description differs at line {first:?}");
                }
            }
            seen += 1;
        }
        assert!(seen >= 4, "{seen} releases lowered");
    }

    /// The random Qwen3.5 the logits checks run lowers to the executor's own description of it too.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_qwen35_lowers_as_its_own_description() {
        let dir = std::env::temp_dir().join(format!("glyd-lower-q35-{}", std::process::id()));
        crate::logits::qwen35_random(&dir).unwrap();
        let c = crate::qwen35::Config35::load(&dir).unwrap();
        let st = super::super::Stage::whole(c.layers);
        let (own, lowered) =
            (super::super::qwen35::describe(&c, &st).unwrap(), super::super::qwen35::lowered(&dir, &c, &st).unwrap_or_else(|e| panic!("{}", e.msg)));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(format!("{own:#?}"), format!("{lowered:#?}"));
    }

    /// The random Gemma 4 the logits checks run (src/logits.rs, a window of 32, a global layer that shares its keys and values) lowers to the executor's own
    /// description of it as well: the path the GPU job's checks take.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_gemma4_lowers_as_its_own_description() {
        let dir = std::env::temp_dir().join(format!("glyd-lower-g4-{}", std::process::id()));
        crate::logits::gemma4_random(&dir).unwrap();
        let c = crate::gemma4::ConfigG4::load(&dir).unwrap();
        let (own, lowered) = (super::super::gemma4::describe(&c), super::super::gemma4::lowered(&dir, &c).unwrap_or_else(|e| panic!("{}", e.msg)));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(format!("{own:#?}"), format!("{lowered:#?}"));
    }

    /// Gemma 4 lowered from its released config's kinds, sized by the shapes the description gives, is to the last field the executor's own description of it
    /// (src/dynamic/gemma4.rs); and those shapes are the checkpoint's (the Hub's safetensors header), every one.
    #[test]
    fn gemma4_lowered_is_the_executors_own_description() {
        let (text, shapes) = header("gemma-4-12B-it.json");
        let c = crate::gemma4::ConfigG4::parse(&text, None, false).unwrap();
        let own = super::super::gemma4::describe(&c);
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
        assert_eq!(kinds.shapes.len(), kinds.names().len(), "a shape for every tensor described");
        for (n, s) in &kinds.shapes {
            assert_eq!(Some(s), shapes.get(n), "{n}");
        }
        let lowered = gemma4(&kinds, &|n: &str| kinds.shapes.get(n).cloned(), c.eos.clone(), c.max_pos).unwrap_or_else(|e| panic!("{}", e.msg));
        let (a, b) = (format!("{own:#?}"), format!("{lowered:#?}"));
        if a != b {
            let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
            panic!("the lowered description differs from the executor's at line {first:?}");
        }
    }
}
