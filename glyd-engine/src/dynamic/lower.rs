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
