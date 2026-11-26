//! A model's description in layer kinds (`runtime::describe`: what a config.json says) lowered to the executor's operations (`ModelDesc`), so that a family runs from
//! its config alone and not from a description written for it (R2, R3a; docs/design/runtime.md section 3). The matrices' shapes come from the checkpoint (`shape`: a
//! tensor's dims by name), which the kinds do not hold. Each family's lowering is held to the executor's own description of it (the tests below) until that one goes.

use super::attn::AttnOp;
use super::{HeadMix, ModelDesc, MoeOp, NormW, Op, Ring, Rope, Slot, StreamsMix, SLOTS};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{Body, Ffn, Mixer};
use crate::weights::Item;

/// A tensor's dims by its name in the checkpoint (None: not there).
pub type Shape<'a> = &'a dyn Fn(&str) -> Option<Vec<usize>>;

/// `lowered`, where it is the family's own description of the same config (`own`, which the family's loader sizes its buffers by): two readings of one config.json
/// that differ would give operations of other widths than the buffers they run on, so a model whose two readings differ is not loaded, the first difference named.
pub fn agree(lowered: ModelDesc, own: &ModelDesc, what: &str) -> Result<ModelDesc> {
    let (a, b) = (format!("{own:?}"), format!("{lowered:?}"));
    if a == b {
        return Ok(lowered);
    }
    let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
    let near = |s: &str| s.get(at.saturating_sub(60)..(at + 60).min(s.len())).unwrap_or("").to_string();
    err(format!(
        "{what}: its config's layer kinds and the family's reader of it describe different models (at {:?}, where the reader has {:?}); not loaded",
        near(&b),
        near(&a)
    ))
}

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
    if m.embed.per_layer.is_some() {
        return err("Gemma 4: per-layer inputs (E2B, E4B): not an operation of the executor yet");
    }
    // (the executor's head: the product's bf16 output, then g4_softcap's three rounded operations)
    use crate::runtime::kinds::{Collapse, Rounding};
    if m.head.rounding != Rounding::Twice || m.head.scale != 1.0 || m.head.collapse != Collapse::None {
        return err(format!("Gemma 4: a head of {:?}, where the executor's rounds the logits to bf16 between operations, unscaled", m.head));
    }
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
    // (each attention block's ring, for the layers that share its keys and values: `Attn::kv_from`)
    let mut ring_of = std::collections::BTreeMap::new();
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
            ropes.push(Rope { theta: rope.theta, dim: rope.dim, pairs: rope.freqs, trig64: false });
            ropes.len() - 1
        });
        let (dh, nk, has_v) = (a.head_dim, a.kv_heads, !a.k_eq_v);
        // (a layer that shares an earlier layer's keys and values reads that layer's ring and writes none: its product is the queries alone)
        let ring = match a.kv_from {
            None => {
                rings.push(Ring::kv(dh, nk, a.window.unwrap_or(0)));
                ring_of.insert(2 * l, rings.len() - 1);
                rings.len() - 1
            }
            Some(j) => {
                *ring_of.get(&j).ok_or_else(|| Error(format!("Gemma 4: layer {l} shares the keys and values of block {j}, which makes none before it")))?
            }
        };
        let mut qkv = vec![one(ar, "q")?];
        if a.kv_from.is_none() {
            qkv.push(one(ar, "k")?);
            if has_v {
                qkv.push(one(ar, "v")?);
            }
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
        let q_norm = name(one(ar, "q_norm")?);
        let k_norm = if a.kv_from.is_none() { name(one(ar, "k_norm")?) } else { q_norm };
        let (post_attn, pre_ff) = (name(one(ar, "norm_out")?), name(one(fr, "norm_in")?));
        let (post_ff, scalar) = (name(one(fr, "norm_out")?), name(one(fr, "res_scale")?));
        let next = if l + 1 < layers { in_norms[l + 1] } else { final_norm };
        let mut attn = AttnOp::of(a, Slot::Proj, ring, Some(r), Some((q_norm, k_norm)))?;
        if a.kv_from.is_some() {
            (attn.stride, attn.v_off, attn.v_norm, attn.writes) = (a.heads * dh, None, false, false);
        }
        if attn.stride != stride {
            return err(format!("Gemma 4: layer {l}'s q, k and v product is {stride} wide, where its attention reads {}", attn.stride));
        }
        ops.push(vec![
            Op::Linear { w, x: Slot::H, y: Slot::Proj, name: if sliding { "qkv (sliding)" } else { "qk (global)" } },
            Op::Attn(attn),
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
        forms: Vec::new(),
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
        host: Vec::new(),
        context: Vec::new(),
    })
}

/// The pre-norm dense decoder (the description builder's `llama` reading): pairs of blocks, an attention of the kind's parameters
/// (its q and k norms where the kind has them, a window where it slides) and the gated SiLU MLP, each `x + block(rmsnorm(x) w)`; the final norm on the last rows,
/// the head the embedding's matrix where tied. Each matrix of a layer is its checkpoint tensors stacked (q, k and v; gate and up), sized by the shapes the
/// description gives. The products are the library's at every row count, a long chunk's decoded and multiplied dense where its route says so: the products,
/// norms, activation, embedding and head of the dense engine this replaces, kernel for kernel. A kind this lowering does not take (a bias, a rope with
/// scaling, a norm or activation of another kind) is refused, named.
pub fn dense(d: &Described, eos: Vec<i64>, max_pos: usize) -> Result<ModelDesc> {
    use crate::runtime::kinds::{Act, Collapse, NormKind, NormWeight, Residual, RopeScaling};
    let m = &d.desc;
    let what = super::dense::FAMILY;
    if m.blocks.len() % 2 != 0 || m.blocks.is_empty() {
        return err(format!("{what}: {} blocks, where each layer is an attention block and an MLP block", m.blocks.len()));
    }
    let eps = m.blocks[0].norm_in.eps;
    let plain = |n: &crate::runtime::kinds::Norm, at: &str| -> Result<()> {
        if n.kind != NormKind::Rms || n.weight != NormWeight::Plain || n.eps != eps {
            return err(format!("{what}: {at}'s norm is {n:?}, where the executor's takes an RMS norm times its weight, eps {eps}"));
        }
        Ok(())
    };
    plain(&m.final_norm, "the final norm")?;
    if m.embed.scale != 1.0 || m.embed.norm.is_some() || m.embed.streams != 1 {
        return err(format!("{what}: an embedding of {:?}, where the executor's reads the rows as they are", m.embed));
    }
    if m.head.softcap.is_some() || m.head.scale != 1.0 || m.head.collapse != Collapse::None {
        return err(format!("{what}: a head of {:?}, where the executor's is the product alone", m.head));
    }
    let one = |roles: &[(String, Source)], r: &str| -> Result<String> {
        match role(roles, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("{what}: the role {r} is {other:?}, where one tensor is")),
        }
    };
    let shape = |n: &str| d.shapes.get(n).cloned();
    let layers = m.blocks.len() / 2;
    let (mut mats, mut small, mut rings, mut ropes, mut ops) = (Vec::new(), Vec::<String>::new(), Vec::new(), Vec::<Rope>::new(), Vec::with_capacity(layers));
    let name = |small: &mut Vec<String>, n: String| -> usize {
        small.push(n);
        small.len() - 1
    };
    let (mut proj_w, mut q_w, mut inter_w, mut first) = (0, 0, 0, None);
    for l in 0..layers {
        let (ab, fb) = (&m.blocks[2 * l], &m.blocks[2 * l + 1]);
        let (ar, fr) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        let a = match &ab.body {
            Body::Mixer(Mixer::Attention(a)) => a,
            b => return err(format!("{what}: layer {l}'s first block is {b:?}, not attention")),
        };
        match &fb.body {
            Body::Ffn(Ffn::Mlp(mlp)) if mlp.act == Act::Silu && mlp.gated && !mlp.bias && mlp.clamp.is_none() => {}
            b => return err(format!("{what}: layer {l}'s second block is {b:?}, where the executor's is the gated SiLU MLP")),
        }
        for b in [ab, fb] {
            plain(&b.norm_in, &format!("layer {l}"))?;
            if b.norm_out.is_some() || b.residual != Residual::Add {
                return err(format!("{what}: layer {l}'s block has a norm after it or a residual of {:?}, where the executor's adds its output", b.residual));
            }
        }
        if a.q_norm.is_some_and(|n| n.eps != eps) {
            return err(format!("{what}: layer {l}'s q and k norms have eps {:?}, where the model's is {eps}", a.q_norm.map(|n| n.eps)));
        }
        let table = match &a.rope {
            None => None,
            Some(r) if r.scaling != RopeScaling::None || r.mrope.is_some() || r.freq_dim != r.dim => {
                return err(format!("{what}: layer {l}'s rope is {r:?}, where the executor's tables are the plain rotary's"));
            }
            Some(r) => {
                let t = Rope { theta: r.theta, dim: r.dim, pairs: r.freqs, trig64: false };
                Some(match ropes.iter().position(|x| (x.theta, x.dim, x.pairs) == (t.theta, t.dim, t.pairs)) {
                    Some(i) => i,
                    None => {
                        ropes.push(t);
                        ropes.len() - 1
                    }
                })
            }
        };
        let (norm_in, post) = (name(&mut small, one(ar, "norm_in")?), name(&mut small, one(fr, "norm_in")?));
        let qk = match a.q_norm {
            Some(_) => Some((name(&mut small, one(ar, "q_norm")?), name(&mut small, one(ar, "k_norm")?))),
            None => None,
        };
        let ring = rings.len();
        rings.push(Ring::kv(a.head_dim, a.kv_heads, a.window.unwrap_or(0)));
        let attn = AttnOp::of(a, Slot::Proj, ring, table, qk)?;
        let w = mats.len();
        let qkv = item(vec![one(ar, "q")?, one(ar, "k")?, one(ar, "v")?], &shape)?;
        let stride = qkv.rows;
        if stride != attn.stride {
            return err(format!("{what}: layer {l}'s q, k and v product is {stride} wide, where its attention reads {}", attn.stride));
        }
        mats.push(qkv);
        mats.push(item(vec![one(ar, "o")?], &shape)?);
        let gate_up = item(vec![one(fr, "gate")?, one(fr, "up")?], &shape)?;
        let inter = gate_up.rows / 2;
        mats.push(gate_up);
        mats.push(item(vec![one(fr, "down")?], &shape)?);
        let mut layer = Vec::with_capacity(8);
        match l {
            0 => first = Some(Op::RmsNorm { x: Slot::X, w: norm_in, y: Slot::H, weight: NormW::Plain, eps }),
            _ => layer.push(Op::NormAdd { y: Slot::D, w: norm_in, plus1: false }),
        }
        layer.extend([
            Op::Linear { w, x: Slot::H, y: Slot::Proj, name: "attention: qkv product" },
            Op::Attn(attn),
            Op::Linear { w: w + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
            Op::NormAdd { y: Slot::O, w: post, plus1: false },
            Op::GateUpSilu { w: w + 2, inter },
            Op::Linear { w: w + 3, x: Slot::M, y: Slot::D, name: "MLP: down product" },
        ]);
        ops.push(layer);
        (proj_w, q_w, inter_w) = (proj_w.max(stride), q_w.max(a.heads * a.head_dim), inter_w.max(inter));
    }
    let final_norm = name(&mut small, d.final_norm.clone());
    let embed = mats.len();
    mats.push(item(vec![d.embed.clone()], &shape)?);
    let lm = match (&d.head, m.head.tied) {
        (None, true) => embed,
        (Some(h), false) => {
            mats.push(item(vec![h.clone()], &shape)?);
            embed + 1
        }
        (h, tied) => return err(format!("{what}: a head of {h:?} where tied is {tied}")),
    };
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
        (Slot::Hlast, m.hidden),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: what,
        hidden: m.hidden,
        vocab: m.vocab,
        eps,
        max_pos,
        widths,
        mats,
        small,
        forms: Vec::new(),
        ropes,
        rings,
        first: first.ok_or_else(|| Error(format!("{what}: no layers")))?,
        layers: ops,
        states: Vec::new(),
        final_norm: Some((final_norm, Slot::D, false)),
        decoded: true,
        skinny: false,
        embed,
        lm,
        softcap: 0.0,
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
        host: Vec::new(),
        context: Vec::new(),
    })
}

/// Qwen3.8-Flash-Next (`qwen4_exp_text`, Condor's model) on the shared operations, from its layer kinds alone: every block inside the gated residuals (`Op::StreamsPre`
/// with `StreamsMix::Gated`, `Op::StreamsPost` gated) over the four streams the embedding row becomes (`Op::Streams`), the final mixer (`HeadMix::Gated`); Gated DeltaNet
/// with the sigmoid gate (`Op::GatedNorm` `sigmoid`); the sparse attention (`Op::IdxMean`, `Op::IdxPick` of `IdxKeys::Pools` unweighted, `Op::Attn` `picked`); the experts
/// (`Op::Moe`, block k the k-th layer's); the n-gram embedding before its layer (`Op::NgramRows`, its product, `Op::NgramAdd`). The matrices are the checkpoint's by
/// name (the table of n-grams held as saved, on the GPU or on the host as its kind says, the embedding too: `tables::held`); the small tensors as saved (A_log fp32, as the delta rule reads it: `dynamic::qwen4x`). `max_pos`: the positions
/// each sequence's states hold (the indexer's keys and pools, the sparse attention's keys and values: a row a position).
pub fn qwen4x(d: &Described, eos: Vec<i64>, max_pos: usize) -> Result<ModelDesc> {
    use crate::runtime::kinds::{Act, Collapse, IndexForm, Ngram, NormWeight, Residual, RopeScaling};
    let m = &d.desc;
    let what = "Qwen3.8-Flash-Next";
    let one = |roles: &[(String, Source)], r: &str| -> Result<String> {
        match role(roles, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("{what}: the role {r} is {other:?}, where one tensor is")),
        }
    };
    let rows_of = |roles: &[(String, Source)], r: &str| -> Result<Vec<String>> {
        match role(roles, r) {
            Some(Source::Rows(v)) => Ok(v.clone()),
            Some(Source::Tensor(n)) => Ok(vec![n.clone()]),
            other => err(format!("{what}: the role {r} is {other:?}, where tensors stacked by rows are")),
        }
    };
    let shape = |n: &str| d.shapes.get(n).cloned();
    let (d_, eps) = (m.hidden, m.final_norm.eps);
    let pn = |n: &crate::runtime::kinds::Norm, at: &str| -> Result<()> {
        if n.weight != NormWeight::PlusOne || n.eps != eps {
            return err(format!("{what}: {at}'s norm is {n:?}, where the kernels take a (1 + w) RMS norm, eps {eps}"));
        }
        Ok(())
    };
    pn(&m.final_norm, "the final mixer")?;
    let hc = m.embed.streams;
    if !(1..=4).contains(&hc)
        || m.embed.scale != 1.0
        || m.embed.norm.is_some()
        || m.head.collapse != Collapse::GatedMix
        || m.head.softcap.is_some()
        || m.head.scale != 1.0
    {
        return err(format!("{what}: an embedding of {:?} and a head of {:?}: the kernels take 1 to 4 gated streams and their mixer", m.embed, m.head));
    }
    let (mut mats, mut small, mut states, mut ropes, mut rings) =
        (Vec::new(), Vec::<String>::new(), Vec::<usize>::new(), Vec::<Rope>::new(), Vec::<Ring>::new());
    let mut ops: Vec<Vec<Op>> = Vec::new();
    let mut layer: Vec<Op> = Vec::new();
    let (mut mix_w, mut attn_w, mut rank_w, mut ple_w, mut n_moe) = (0, 0, 0, 0, 0);
    let (mut cd_w, mut vd_w, mut nv_w, mut ix_w, mut ixd_w) = (0, 0, 0, 0, 0);
    let name = |small: &mut Vec<String>, n: String| -> usize {
        small.push(n);
        small.len() - 1
    };
    let mat = |mats: &mut Vec<crate::weights::Item>, members: Vec<String>| -> Result<usize> {
        mats.push(item(members, &shape)?);
        Ok(mats.len() - 1)
    };
    let state = |states: &mut Vec<usize>, bytes: usize| -> usize {
        states.push(bytes);
        states.len() - 1
    };
    // (the gated residuals' ops around a block of `roles`: the norm per stream, the bottleneck, the inject product)
    let gated =
        |mats: &mut Vec<crate::weights::Item>, small: &mut Vec<String>, roles: &[(String, Source)], b: &crate::runtime::kinds::Block| -> Result<(Op, usize)> {
            let Residual::GatedStreams(g) = b.residual else {
                return err(format!("{what}: a block of residual {:?}, where every block's is the gated residuals", b.residual));
            };
            if g.streams != hc {
                return err(format!("{what}: gated residuals over {} streams, where the embedding makes {hc}", g.streams));
            }
            pn(&b.norm_in, "a gated residual")?;
            let norm = name(small, one(roles, "norm_in")?);
            let (down, up, inject) =
                (mat(mats, vec![one(roles, "mix_down")?])?, mat(mats, vec![one(roles, "mix_up")?])?, mat(mats, vec![one(roles, "inject")?])?);
            Ok((Op::StreamsPre { mix: StreamsMix::Gated { down, up, inject }, norm, round: crate::runtime::kinds::Rounding::Twice }, g.lowrank))
        };
    let post = Op::StreamsPost { round: crate::runtime::kinds::Rounding::Twice, gated: true };
    // (each n-gram block's table and where its kind holds it: `tables::held`)
    let mut tables = Vec::new();
    for (b, roles) in m.blocks.iter().zip(&d.blocks) {
        match &b.body {
            Body::Ngram(n) => {
                let Ngram { order, heads, dim, conv, dilation, streams, eos: end, .. } = *n;
                if (order, conv, dilation, streams, hc) != (3, 4, 3, 4, 4)
                    || n.form != crate::runtime::kinds::NgramForm::Segmented
                    || n.norm.weight != NormWeight::PlusOne
                    || dim % (2 * heads) != 0
                    || b.residual != Residual::Add
                {
                    return err(format!(
                        "{what}: an n-gram embedding of {n:?}: the kernels take bigrams and trigrams, a convolution of 4 taps 3 apart, four streams"
                    ));
                }
                pn(&b.norm_in, "the n-gram embedding's query")?;
                let table = mat(&mut mats, rows_of(roles, "ngram")?)?;
                tables.push((table, n.table));
                let hash =
                    [name(&mut small, one(roles, "ngram_mult")?), name(&mut small, one(roles, "ngram_sizes")?), name(&mut small, one(roles, "ngram_offsets")?)];
                let kv = mat(&mut mats, vec![one(roles, "key")?, one(roles, "value")?])?;
                let norms = [name(&mut small, one(roles, "norm_in")?), name(&mut small, one(roles, "norm_key")?), name(&mut small, one(roles, "norm_conv")?)];
                let taps = name(&mut small, one(roles, "conv")?);
                let (hist, cv) = (state(&mut states, 8), state(&mut states, 9 * 4 * d_ * 2));
                let conv = super::ngram::NgramConv { norm: norms[2], taps, state: cv, taps_n: conv, dilation };
                layer.extend([
                    Op::NgramRows(super::ngram::NgramRows {
                        table,
                        hash,
                        hist,
                        per: heads,
                        row: dim / (2 * heads),
                        eos: end,
                        order,
                        map: None,
                        segments: true,
                    }),
                    Op::Linear { w: kv, x: Slot::In, y: Slot::Gu, name: "n-gram embedding: key and value product" },
                    Op::NgramAdd(super::ngram::NgramAdd { q: norms[0], k: norms[1], conv: Some(conv), plus_one: true, eps: n.norm.eps }),
                ]);
                ple_w = ple_w.max(dim);
            }
            Body::Mixer(mx) => {
                let (pre, rank) = gated(&mut mats, &mut small, roles, b)?;
                rank_w = rank_w.max(rank);
                layer.push(pre);
                match mx {
                    Mixer::Linear(l) => {
                        if l.gate_act != Act::Sigmoid || l.k_dim != 128 || l.v_dim != 128 || l.conv != 4 || l.norm_eps != eps {
                            return err(format!("{what}: a linear attention of {l:?}: the kernels take heads of 128, a convolution of 4 and the sigmoid gate"));
                        }
                        let (nv, cd) = (l.v_heads, 2 * l.k_heads * 128 + l.v_heads * 128);
                        let row = cd + nv * 128;
                        let qkvz = mat(&mut mats, vec![one(roles, "qkv")?, one(roles, "gate")?])?;
                        let ba = mat(&mut mats, vec![one(roles, "beta")?, one(roles, "decay")?])?;
                        let o = mat(&mut mats, vec![one(roles, "o")?])?;
                        let [conv, a_log, dt_bias, gn] = ["conv", "a_log", "dt_bias", "norm"].map(|r| one(roles, r).map(|t| name(&mut small, t)));
                        let (cs, ss) = (state(&mut states, 3 * cd * 2), state(&mut states, nv * 128 * 128 * 4));
                        layer.extend([
                            Op::Linear { w: qkvz, x: Slot::H, y: Slot::Mix, name: "linear attention: qkvz product" },
                            Op::Linear { w: ba, x: Slot::H, y: Slot::Ba, name: "linear attention: ba product" },
                            Op::GdnConv { w: conv?, state: cs, cd, row },
                            Op::GdnDelta { a_log: a_log?, dt_bias: dt_bias?, state: ss, cd, nk: l.k_heads, nv },
                            Op::GatedNorm { w: gn?, row, nv, z_off: cd, sigmoid: true },
                            Op::Linear { w: o, x: Slot::Core, y: Slot::O, name: "linear attention: out product" },
                        ]);
                        (mix_w, cd_w, vd_w, nv_w) = (mix_w.max(row), cd_w.max(cd), vd_w.max(nv * 128), nv_w.max(nv));
                    }
                    Mixer::Attention(a) => {
                        let ix = a.indexer.as_ref().ok_or_else(|| Error(format!("{what}: an attention with no indexer")))?;
                        let IndexForm::Mean { norm: inorm } = ix.form else {
                            return err(format!("{what}: an indexer of {:?}, where the model's pools are mean-pooled", ix.form));
                        };
                        pn(&inorm, "the indexer")?;
                        let r = a.rope.as_ref().ok_or_else(|| Error(format!("{what}: an attention with no rotary")))?;
                        let (idim, iblock) = (ix.head_dim, ix.block);
                        if r.scaling != RopeScaling::None
                            || r.freq_dim != r.dim
                            || crate::kern::qwen4x::index_fits(idim, iblock).is_err()
                            || r.dim > idim
                            || ix.top_k % iblock != 0
                        {
                            return err(format!(
                                "{what}: a rotary of {r:?} and an indexer of {ix:?}: the kernels take the plain rotary within the index heads, of a multiple of 32 up to 1,024"
                            ));
                        }
                        let rt = Rope { theta: r.theta, dim: r.dim, pairs: r.freqs, trig64: false };
                        let table = match ropes.iter().position(|x| (x.theta, x.dim, x.pairs) == (rt.theta, rt.dim, rt.pairs)) {
                            Some(i) => i,
                            None => {
                                ropes.push(rt);
                                ropes.len() - 1
                            }
                        };
                        let qk = (name(&mut small, one(roles, "q_norm")?), name(&mut small, one(roles, "k_norm")?));
                        // (its keys and values, the indexer's raw keys and its pools' keys: rings of the executor's pages, as every attention kind's; the pools an entry of `iblock` tokens a row)
                        let ring = rings.len();
                        rings.extend([Ring::kv(a.head_dim, a.kv_heads, 0), Ring::keys(idim, 1, 0), Ring::entries(idim, iblock)]);
                        let att = AttnOp::picked(a, Slot::Mix, ring, Some(table), Some(qk))?;
                        let stride = att.stride;
                        let qkv = mat(&mut mats, vec![one(roles, "q")?, one(roles, "k")?, one(roles, "v")?])?;
                        if mats[qkv].rows != stride {
                            return err(format!("{what}: a q, k and v product of {} rows, where its attention reads {stride}", mats[qkv].rows));
                        }
                        let index = mat(&mut mats, vec![one(roles, "index_qk")?])?;
                        let o = mat(&mut mats, vec![one(roles, "o")?])?;
                        let (iqn, ikn) = (name(&mut small, one(roles, "index_q_norm")?), name(&mut small, one(roles, "index_k_norm")?));
                        let (keys, pools) = (ring + 1, ring + 2);
                        let pick = super::IdxPick {
                            heads: ix.heads,
                            ratio: ix.block,
                            top_k: ix.top_k / ix.block,
                            expand: true,
                            tail: true,
                            wscale: 1.0,
                            score: super::IdxScore::F32 { dim: idim, scale: (idim as f32).powf(-0.5) },
                            keys: super::IdxKeys::Pools(pools),
                            weighted: false,
                            cand: None,
                        };
                        layer.extend([
                            Op::Linear { w: qkv, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                            Op::Linear { w: index, x: Slot::H, y: Slot::Qr, name: "indexer: product" },
                            Op::IdxMean { heads: ix.heads, dim: idim, block: iblock, q_norm: iqn, k_norm: ikn, rope: table, keys, pools },
                            Op::IdxPick(pick),
                            Op::Attn(att),
                            Op::Linear { w: o, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                        ]);
                        (mix_w, attn_w, ix_w, ixd_w) = (mix_w.max(stride), attn_w.max(a.heads * a.head_dim), ix_w.max(ix.heads), ixd_w.max(idim));
                    }
                    mx => return err(format!("{what}: a mixer of {mx:?}")),
                }
                layer.push(post.clone());
            }
            Body::Ffn(Ffn::Moe(_)) => {
                let (pre, rank) = gated(&mut mats, &mut small, roles, b)?;
                rank_w = rank_w.max(rank);
                layer.extend([pre, Op::Moe(MoeOp::of(n_moe)), post.clone()]);
                n_moe += 1;
                ops.push(std::mem::take(&mut layer));
            }
            f => return err(format!("{what}: a block of {f:?}")),
        }
    }
    if !layer.is_empty() || ops.is_empty() {
        return err(format!("{what}: a layer that does not end with its experts"));
    }
    let final_norm = name(&mut small, d.final_norm.clone());
    let xm = |r: &str| d.extra.iter().find(|(n, _)| n == r).and_then(|(_, s)| if let Source::Tensor(t) = s { Some(t.clone()) } else { None });
    let (down, up) = (
        xm("out_mix_down").ok_or_else(|| Error(format!("{what}: no final mixer")))?,
        xm("out_mix_up").ok_or_else(|| Error(format!("{what}: no final mixer")))?,
    );
    let (down, up) = (mat(&mut mats, vec![down])?, mat(&mut mats, vec![up])?);
    if let Some(last) = ops.last_mut() {
        last.push(Op::StreamsHead { mix: HeadMix::Gated { down, up }, norm: final_norm, round: crate::runtime::kinds::Rounding::Twice });
    }
    let embed = mat(&mut mats, vec![d.embed.clone()])?;
    let lm = match &d.head {
        Some(h) => mat(&mut mats, vec![h.clone()])?,
        None => embed,
    };
    let w = hc * d_;
    let mut widths = [0usize; SLOTS];
    for (s, x) in [
        (Slot::X, d_),
        (Slot::H, d_),
        (Slot::S, w),
        (Slot::Hq, w),
        (Slot::Mid, rank_w),
        (Slot::Cp, w),
        (Slot::Kv, hc),
        (Slot::O, d_),
        (Slot::Mix, mix_w),
        (Slot::Ba, 2 * nv_w),
        (Slot::Xc, cd_w),
        (Slot::Core, vd_w),
        (Slot::Qo, attn_w),
        (Slot::Att, attn_w),
        (Slot::Qr, (ix_w + 1) * ixd_w),
        (Slot::Iq, ix_w * ixd_w),
        (Slot::In, ple_w),
        (Slot::Gu, if ple_w > 0 { 5 * d_ } else { 0 }),
        (Slot::D, if ple_w > 0 { w } else { 0 }),
        (Slot::M, if ple_w > 0 { w } else { 0 }),
    ] {
        widths[s as usize] = x;
    }
    Ok(ModelDesc {
        family: what,
        hidden: d_,
        vocab: m.vocab,
        eps,
        max_pos,
        widths,
        mats,
        small,
        forms: Vec::new(),
        ropes,
        rings,
        first: Op::Streams,
        layers: ops,
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
        hc: (0.0, 0),
        logit_mult: 0.0,
        host: super::tables::held(m, embed, lm, &tables),
        context: Vec::new(),
    })
}

/// A model's lowering of its blocks (its family's: `qwen4x`, or `latent_moe` of the whole model): what a draft head's blocks are lowered by.
pub type Blocks = fn(&Described, Vec<i64>, usize) -> Result<ModelDesc>;

/// What a model's lowering gives a draft head's blocks back as (`draft_head`): the executor's description, and where the loader is told of the head's own
/// tensors: by name in the description's lists (`ModelDesc` itself: `mats`, `small`), or in a plan of the compressed attention's lowering (`lower_comp::Lowered`).
pub trait Lowered {
    fn desc(&mut self) -> &mut ModelDesc;
    /// The head's own matrix `name` of `rows` x `cols` (held as the checkpoint has it, bf16 and not packed, where `dense`: a table looked up by row, or a
    /// product's matrix of columns no product route of the packed layouts takes, which `Mat::Dense`'s small product multiplies): its index, and whether it
    /// is in the FP8 matrices' list (`Op::F8Lin`'s) rather than the matrices'.
    fn own_mat(&mut self, name: &str, rows: usize, cols: usize, dense: bool, shape: Shape) -> Result<(usize, bool)>;
    /// The head's own small tensor `name`: its index.
    fn own_small(&mut self, name: &str) -> usize;
    /// The model's embedding and LM head, which the head reads as the model's: not loaded with it.
    fn share_ends(&mut self) {}
}

impl Lowered for ModelDesc {
    fn desc(&mut self) -> &mut ModelDesc {
        self
    }

    fn own_mat(&mut self, name: &str, rows: usize, cols: usize, dense: bool, shape: Shape) -> Result<(usize, bool)> {
        if dense {
            return err(format!("{name}: a matrix held as it is among a description's matrices by name, which the executor packs"));
        }
        self.mats.push(item(vec![name.to_string()], shape)?);
        let it = &self.mats[self.mats.len() - 1];
        if (it.rows, it.cols) != (rows, cols) {
            return err(format!("{name} of {}x{}, where the head reads {rows}x{cols}", it.rows, it.cols));
        }
        Ok((self.mats.len() - 1, false))
    }

    fn own_small(&mut self, name: &str) -> usize {
        self.small.push(name.to_string());
        self.small.len() - 1
    }
}

/// The description's first draft head (`kinds::MtpHead`) as a draft part on the shared operations, for every family that has one: its blocks lowered as the
/// model's layers are (`blocks`, the model's lowering, on a model of the head's blocks and final norm), after `Op::MtpIn`, the head's input from the model's
/// state its row continues (Hid) and the next token's embedding (the model's X), by the head's form (`MtpFront`):
/// - `MtpInput::Streams` (Qwen3.8-Flash-Next's): the head's streams from the model's last streams, its own final mixer (the model's collapse with the head's
///   roles, `HeadMix::Gated`) into the model's LM head; its norms (1 + w).
/// - `MtpInput::Joined` (DeepSeek-V3's, GLM-5.3's): its residual from eh_proj of the two normed rows joined; the first block's norm then reads it as the
///   model's first layer's reads the embedding; its final norm's output the LM head's input and the next draft's state; its norms plain, rounded as
///   transformers' bf16 RMSNorm (`Rounding::Twice`).
/// - DSpark's (`kinds::DSpark`, every DeepSeek release's drafter): a block of tokens in one pass (`block_head`).
///
/// The model's embedding and LM head are named (the executor shares the model's, never loading them twice: `DraftSpec`); its rings are its own, from 0 (the
/// executor puts them after the model's).
pub fn draft_head<L: Lowered>(d: &Described, eos: Vec<i64>, max_pos: usize, blocks: impl Fn(&Described, Vec<i64>, usize) -> Result<L>) -> Result<L> {
    use super::MtpFront;
    use crate::runtime::kinds::{MtpInput, NormKind, NormWeight, Rounding};
    let what = "a draft head";
    let (Some(h), Some((roles, own))) = (d.desc.mtp.first(), d.mtp.first()) else {
        return err(format!("{what}: the description names no draft head"));
    };
    if h.dspark.is_some() {
        return block_head(d, eos, max_pos, blocks);
    }
    let eps = d.desc.final_norm.eps;
    let (weight, rounding) = match h.input {
        MtpInput::Streams => (NormWeight::PlusOne, h.norm_embed.rounding),
        MtpInput::Joined => (NormWeight::Plain, Rounding::Twice),
    };
    for (n, at) in [(&h.norm_embed, "its embedding's norm"), (&h.norm_hidden, "its state's norm")] {
        if n.kind != NormKind::Rms || n.weight != weight || n.rounding != rounding || n.eps != eps {
            return err(format!("{what}: {at} is {n:?}, where the kernels take an RMS norm of a {weight:?} weight rounded {rounding:?}, eps {eps}"));
        }
    }
    let one = |r: &str| -> Result<String> {
        match role(own, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("{what}: the role {r} is {other:?}, where one tensor is")),
        }
    };
    // (the head as a model of its blocks and its final norm; a head of streams' final mixer the model's collapse with its own roles)
    let mut hd = d.clone();
    hd.desc.blocks = h.blocks.clone();
    hd.desc.final_norm = h.final_norm;
    hd.desc.mtp = Vec::new();
    hd.blocks = roles.clone();
    hd.final_norm = one("final_norm")?;
    if h.input == MtpInput::Streams {
        hd.extra = vec![("out_mix_down".into(), Source::Tensor(one("out_mix_down")?)), ("out_mix_up".into(), Source::Tensor(one("out_mix_up")?))];
    }
    hd.mtp = Vec::new();
    let mut l = blocks(&hd, eos, max_pos).map_err(|e| e.context(what))?;
    let shape = |n: &str| d.shapes.get(n).cloned();
    let (ne, nh) = (l.own_small(&one("norm_embed")?), l.own_small(&one("norm_hidden")?));
    let hidden = l.desc().hidden;
    let fronts: Vec<(&str, usize)> = match h.input {
        MtpInput::Streams => vec![("fc_embedding", hidden), ("fc_hidden", hidden)],
        MtpInput::Joined => vec![("eh_proj", 2 * hidden)],
    };
    let mut k = Vec::new();
    for &(r, cols) in &fronts {
        match l.own_mat(&one(r)?, hidden, cols, false, &shape).map_err(|e| e.context(what))? {
            (i, false) => k.push(i),
            (_, true) => return err(format!("{what}: its front's {r} held FP8, where the front's products are the matrices'")),
        }
    }
    let m = l.desc();
    let front = match h.input {
        MtpInput::Streams => {
            m.widths[Slot::Hid as usize] = m.width(Slot::S);
            MtpFront::Streams { embed: k[0], hidden: k[1] }
        }
        MtpInput::Joined => {
            // (the head's input is its residual, as the embedding is the model's: the lowered first operation, the first block's norm of it, runs after it)
            let first = std::mem::replace(&mut m.first, Op::Nop);
            if !matches!(first, Op::Nop) {
                m.layers.first_mut().ok_or_else(|| Error(format!("{what}: no blocks")))?.insert(0, first);
            }
            (m.widths[Slot::Hid as usize], m.widths[Slot::In as usize]) = (m.hidden, 2 * m.hidden);
            MtpFront::Joined { eh: k[0] }
        }
    };
    m.first = Op::MtpIn { norm_embed: ne, norm_hidden: nh, front };
    Ok(l)
}

/// DSpark's draft head (`kinds::DSpark`; vLLM models/deepseek_v4/nvidia/dspark.py, SGLang models/deepseek_v4_dspark.py): its blocks the model's decoder layers
/// of the window alone, lowered as the model's are (`blocks`) on a model of them with the head's final norm and its collapse's own roles (a hyper-head's
/// `out_hc_*`); its front `MtpFront::Block` (the block's tokens' embeddings as its streams, the Markov head's two matrices after the LM head); its context
/// (`ModelDesc::context`): the model's taps of a row (Hid, the target layers' inputs side by side) to FP8 where the release's products quantize their input,
/// `main_proj`, `main_norm` (the head's `norm_hidden` kind), to FP8, then each layer's own key product and key operation (normed, rotated, into its window) of
/// that main_x. A layer's attention reads its window to the block's last row (`Dyn::ahead`).
fn block_head<L: Lowered>(d: &Described, eos: Vec<i64>, max_pos: usize, blocks: impl Fn(&Described, Vec<i64>, usize) -> Result<L>) -> Result<L> {
    use super::MtpFront;
    use crate::runtime::kinds::{NormKind, NormWeight};
    let what = "a DSpark draft head";
    let (Some(h), Some((roles, own))) = (d.desc.mtp.first(), d.mtp.first()) else {
        return err(format!("{what}: the description names no draft head"));
    };
    let ds = h.dspark.as_ref().ok_or_else(|| Error(format!("{what}: no DSpark form")))?;
    if ds.block < 2 || ds.block > super::VERIFY_ROWS || ds.targets.is_empty() || ds.markov_rank == 0 {
        return err(format!(
            "{what}: a block of {} tokens over {} target layers, a Markov rank of {}: blocks of 2 to {} and some of each",
            ds.block,
            ds.targets.len(),
            ds.markov_rank,
            super::VERIFY_ROWS
        ));
    }
    let n = &h.norm_hidden;
    if n.kind != NormKind::Rms || n.weight != NormWeight::Plain {
        return err(format!("{what}: main_x's norm is {n:?}, where the kernels take an RMS norm of a plain weight"));
    }
    let one = |r: &str| -> Result<String> {
        match role(own, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("{what}: the role {r} is {other:?}, where one tensor is")),
        }
    };
    // (the head as a model of its blocks, its final norm, and its collapse's own roles where the model's collapse has some)
    let mut hd = d.clone();
    hd.desc.blocks = h.blocks.clone();
    hd.desc.final_norm = h.final_norm;
    hd.desc.mtp = Vec::new();
    hd.blocks = roles.clone();
    hd.final_norm = one("final_norm")?;
    hd.extra = own.iter().filter(|(r, _)| r.starts_with("out_hc_")).cloned().collect();
    hd.mtp = Vec::new();
    let mut l = blocks(&hd, eos, max_pos).map_err(|e| e.context(what))?;
    l.share_ends();
    let shape = |n: &str| d.shapes.get(n).cloned();
    let (hidden, vocab, taps) = (d.desc.hidden, d.desc.vocab, ds.targets.len());
    let main = l.own_mat(&one("main_proj")?, hidden, taps * hidden, false, &shape).map_err(|e| e.context(what))?;
    let main_norm = l.own_small(&one("main_norm")?);
    // (the Markov head's matrices held as they are: the first a table looked up by row, the second multiplied by `Mat::Dense`'s small product, its columns
    // (the rank) no product route of the packed layouts)
    let (me, mh) = (
        l.own_mat(&one("markov_embed")?, vocab, ds.markov_rank, true, &shape).map_err(|e| e.context(what))?,
        l.own_mat(&one("markov_head")?, vocab, ds.markov_rank, true, &shape).map_err(|e| e.context(what))?,
    );
    if me.1 || mh.1 {
        return err(format!("{what}: its Markov head's matrices held FP8, where its products are the matrices'"));
    }
    let m = l.desc();
    let lin = |(w, f8): (usize, bool), x: Slot, y: Slot, name: &'static str| if f8 { Op::F8Lin { w, x, y, name } } else { Op::Linear { w, x, y, name } };
    let mut context = vec![
        Op::Fp8 { x: Slot::Hid, y: Slot::In, block: 128 },
        lin(main, Slot::In, Slot::X, "draft head: main projection"),
        Op::RmsNorm { x: Slot::X, w: main_norm, y: Slot::H, weight: NormW::Plain, eps: n.eps },
        Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
    ];
    // (each layer: its key product of Hq and its key operation, as the layer's own; its attention over the window alone)
    for (i, ops) in m.layers.iter().enumerate() {
        let key = ops.iter().position(|op| matches!(op, Op::Linear { y: Slot::Kv, .. } | Op::F8Lin { y: Slot::Kv, .. }));
        let kv = ops.iter().position(|op| matches!(op, Op::CompKv { .. }));
        let window_only = ops.iter().all(|op| !matches!(op, Op::CompAttn { cache: Some(_), .. } | Op::Compress { .. } | Op::IdxPick(_)));
        match (key, kv) {
            (Some(a), Some(b)) if a < b && window_only && matches!(ops[a], Op::Linear { x: Slot::Hq, .. } | Op::F8Lin { x: Slot::Hq, .. }) => {
                context.extend([ops[a].clone(), ops[b].clone()]);
            }
            _ => return err(format!("{what}: its layer {i} is not a compressed attention of the window alone whose key product reads Hq")),
        }
    }
    m.context = context;
    m.widths[Slot::Hid as usize] = taps * hidden;
    m.widths[Slot::In as usize] = m.widths[Slot::In as usize].max(taps * hidden);
    // (the lowering's first operation, the streams from the embedding: the block's front makes them of the model's X)
    if !matches!(m.first, Op::Streams) {
        return err(format!("{what}: its blocks begin with {:?}, where they begin with the streams of the embedding", m.first));
    }
    m.first = Op::MtpIn {
        norm_embed: main_norm,
        norm_hidden: main_norm,
        front: MtpFront::Block { block: ds.block, noise: ds.noise, markov: (me.0, mh.0), rank: ds.markov_rank },
    };
    Ok(l)
}

/// Qwen3.5's layout (dense: Qwen3.5, 3.6 and 3.8 of 256-wide heads): pairs of blocks, a Gated DeltaNet or a gated full attention, then the SiLU MLP, the (1 + w)
/// norms, the final norm on the last row. The matrices and small tensors are named in the order the loader holds them (the tests hold the result to the
/// hand-written `dynamic::qwen35::describe`); stage `st` of a model split across GPUs. A layer whose feed-forward is a mixture of experts (the layout's with experts: a softmax router, the gated shared expert)
/// runs `Op::Moe` in the MLP's place, block k the stage's k-th such layer's (`dynamic::experts`), and names no MLP matrices.
pub fn qwen35(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::qwen35::per_layer;
    use crate::qwen35::Kind;
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("Qwen3.5: {} blocks, where each layer is a mixer block and an MLP block", m.blocks.len()));
    }
    let layers = m.blocks.len() / 2;
    let eps = m.blocks.first().map_or(1e-6, |b| b.norm_in.eps);
    let (mut lin, mut att) = (None, None);
    let mut kinds = Vec::with_capacity(layers);
    // (each layer's feed-forward: the SiLU MLP, or a mixture of experts, its block the layer kind's: `dynamic::experts`)
    let mut sparse = Vec::with_capacity(layers);
    for l in 0..layers {
        match &m.blocks[2 * l + 1].body {
            Body::Ffn(Ffn::Mlp(_)) => sparse.push(false),
            Body::Ffn(Ffn::Moe(e)) => {
                super::experts::cfg(e, m.hidden, 0).map_err(|x| x.context(format!("Qwen3.5: layer {l}")))?;
                sparse.push(true)
            }
            b => return err(format!("Qwen3.5: layer {l}'s second block is {b:?}, not an MLP or a mixture of experts")),
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
    // (the MLP's width, the first dense layer's: 0 where every layer's feed-forward is experts)
    let inter = match sparse.iter().position(|&s| !s) {
        Some(l) => {
            let gate = match role(&d.blocks[2 * l + 1], "gate").ok_or_else(|| Error("Qwen3.5: no MLP gate".into()))? {
                Source::Tensor(n) => n,
                g => return err(format!("Qwen3.5: an MLP gate of {g:?}")),
            };
            d.shapes.get(gate).map(|s| s[0]).ok_or_else(|| Error(format!("Qwen3.5: no shape for {gate}")))?
        }
        None => 0,
    };
    let mut n_moe = 0;
    let (nq, nk, nv) = (att.heads, att.kv_heads, lin.v_heads);
    let (vd, cd) = (nv * lin.v_dim, 2 * lin.k_heads * lin.k_dim + nv * lin.v_dim);
    let full_w = 2 * nq * 256 + 2 * nk * 256;
    let row = cd + vd;
    let (mut n_mats, mut n_small, mut rings, mut states, mut ops_all) = (0, 0, Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    // (the tensors, named in the order the operations index them: a layer's matrices, its norms and its mixer's small tensors)
    let shape = |n: &str| d.shapes.get(n).cloned();
    let one = |roles: &[(String, Source)], r: &str| -> Result<String> {
        match role(roles, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("Qwen3.5: the role {r} is {other:?}, where one tensor is")),
        }
    };
    let (mut mats, mut small) = (Vec::new(), Vec::<String>::new());
    for (l, &kind) in kinds.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            ops_all.push(Vec::new());
            continue;
        }
        let (ar, fr) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        let mixer: (&[&[&str]], &[&str]) = match kind {
            Kind::Linear => (&[&["qkv", "gate"], &["beta", "decay"], &["o"]], &["conv", "a_log", "dt_bias", "norm"]),
            Kind::Full => (&[&["q", "k", "v"], &["o"]], &["q_norm", "k_norm"]),
        };
        for m in mixer.0 {
            mats.push(item(m.iter().map(|r| one(ar, r)).collect::<Result<_>>()?, &shape)?);
        }
        if !sparse[l] {
            mats.push(item(vec![one(fr, "gate")?, one(fr, "up")?], &shape)?);
            mats.push(item(vec![one(fr, "down")?], &shape)?);
        }
        small.extend([one(ar, "norm_in")?, one(fr, "norm_in")?]);
        for r in mixer.1 {
            small.push(one(ar, r)?);
        }
        let (b, s0) = (n_mats, n_small);
        let (mm, nn) = per_layer(kind);
        // (a layer of experts has no MLP's two matrices: its block holds its own)
        (n_mats, n_small) = (n_mats + mm - if sparse[l] { 2 } else { 0 }, n_small + nn);
        let mut ops = Vec::new();
        if l == 0 {
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, weight: NormW::PlusOne, eps });
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
                    Op::GatedNorm { w: s0 + 5, row, nv, z_off: cd, sigmoid: false },
                    Op::Linear { w: b + 2, x: Slot::Core, y: Slot::O, name: "linear attention: out product" },
                ]);
                b + 3
            }
            Kind::Full => {
                let ring = rings.len();
                rings.push(Ring::kv(256, nk, 0));
                let a = AttnOp::of(att, Slot::Mix, ring, Some(0), Some((s0 + 2, s0 + 3)))?;
                if a.stride != full_w {
                    return err(format!("Qwen3.5: an attention product of {full_w} where its attention reads {}", a.stride));
                }
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                    Op::Attn(a),
                    Op::Linear { w: b + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                ]);
                b + 2
            }
        };
        ops.push(Op::NormAdd { y: Slot::O, w: s0 + 1, plus1: true });
        if sparse[l] {
            // (the router and the experts read the normed H; their sum, the gated shared expert's with it, is the layer's update)
            ops.push(Op::Moe(MoeOp { y: Slot::D, ..MoeOp::of(n_moe) }));
            n_moe += 1;
        } else {
            ops.extend([Op::GateUpSilu { w: mlp, inter }, Op::Linear { w: mlp + 1, x: Slot::M, y: Slot::D, name: "MLP: down product" }]);
        }
        ops_all.push(ops);
    }
    let e = n_mats;
    let lm = if d.head.is_none() { e } else { e + 1 };
    small.push(d.final_norm.clone());
    mats.push(item(vec![d.embed.clone()], &shape)?);
    if let Some(h) = &d.head {
        mats.push(item(vec![h.clone()], &shape)?);
    }
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
        eps,
        max_pos,
        widths,
        mats,
        small,
        forms: Vec::new(),
        ropes: vec![Rope { theta: rope.theta, dim: rotary, pairs: rotary / 2, trig64: false }],
        rings,
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, weight: NormW::PlusOne, eps }),
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
        host: Vec::new(),
        context: Vec::new(),
    })
}

/// Nemotron-H (Nemotron 3 Super, Nemotron 3.5 Lightning): a block a layer, Mamba-2, attention without a positional encoding, or a mixture of experts (its block of
/// experts is the Moe kind's, `experts::block`); every layer x + mixer(norm(x)), the final norm on the last row; stage `st` of a model split across GPUs. Every matrix and small
/// tensor the operations use is named from the description's roles, in the order they index them (`dynamic::nemotron::describe`'s), sized by the kinds'
/// widths: a layer's norm, then a Mamba-2 layer's convolution's weight and bias, A_log, D, dt_bias and gated norm; the final norm. Matrices: a Mamba-2 layer's
/// in and out products, an attention layer's q, k and v as one and o; the embedding, then the head where it is not the embedding. A layer's experts are its
/// block's, read apart.
pub fn nemotron(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::nemotron::per_layer;
    use crate::kern::{MAMBA2_HEAD, MAMBA2_STATE};
    use crate::nemotron::NhKind;
    let m = &d.desc;
    let one = |roles: &[(String, Source)], r: &str| match role(roles, r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        x => err(format!("Nemotron-H: the role {r} is {x:?}, where one tensor is")),
    };
    let it = |members: Vec<String>, rows: usize, cols: usize| Item { members, rows, cols };
    let (mut mats, mut small) = (Vec::new(), Vec::new());
    let eps = m.blocks.first().map_or(1e-5, |b| b.norm_in.eps);
    let (mut mamba, mut att) = (None, None);
    let mut kinds = Vec::with_capacity(m.blocks.len());
    for (l, b) in m.blocks.iter().enumerate() {
        kinds.push(match &b.body {
            Body::Mixer(Mixer::Mamba2(x)) => {
                mamba.get_or_insert(x);
                NhKind::Mamba
            }
            Body::Mixer(Mixer::Attention(a)) => {
                att.get_or_insert(a);
                NhKind::Attention
            }
            Body::Ffn(Ffn::Moe(_)) => NhKind::Moe,
            other => return err(format!("Nemotron-H: layer {l} is {other:?}")),
        });
    }
    let mb = mamba.ok_or_else(|| Error("Nemotron-H: no Mamba-2 layer".into()))?;
    let (nq, nk, hd) = att.map_or((0, 0, 128), |a| (a.heads, a.kv_heads, a.head_dim));
    if mb.head_dim != MAMBA2_HEAD || mb.state != MAMBA2_STATE || mb.conv != 4 || hd != 128 {
        return err(format!(
            "Mamba-2 heads of {} channels with a state of {}, a convolution of {}, attention heads of {hd}: this engine runs heads of {MAMBA2_HEAD}, a state of {MAMBA2_STATE}, a convolution of 4 and attention heads of 128",
            mb.head_dim, mb.state, mb.conv
        ));
    }
    let hid = m.hidden;
    let di = mb.heads * mb.head_dim;
    let cd = di + 2 * mb.groups * mb.state;
    let row = di + cd + mb.heads;
    let qkv_w = (nq + 2 * nk) * hd;
    let (mut n_mats, mut n_small, mut n_moe, mut rings, mut states, mut layers) = (0, 0, 0, Vec::new(), Vec::new(), Vec::new());
    let mut first = None;
    for (l, &kind) in kinds.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let (b, s0) = (n_mats, n_small);
        let (mm, nn) = per_layer(kind);
        (n_mats, n_small) = (n_mats + mm, n_small + nn);
        let mut ops = Vec::new();
        if l == 0 {
            first = Some(Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, weight: NormW::Plain, eps });
        } else {
            ops.push(Op::NormAdd { y: Slot::O, w: s0, plus1: false });
        }
        let r = &d.blocks[l];
        small.push(one(r, "norm_in")?);
        match kind {
            NhKind::Mamba => {
                mats.extend([it(vec![one(r, "in")?], row, hid), it(vec![one(r, "o")?], hid, di)]);
                for x in ["conv", "conv_bias", "a_log", "d", "dt_bias", "norm"] {
                    small.push(one(r, x)?);
                }
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * cd * 2);
                states.push(mb.heads * MAMBA2_HEAD * MAMBA2_STATE * 4);
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "Mamba-2: in product" },
                    Op::MambaConv { w: s0 + 1, bias: s0 + 2, state: conv, cd, row, x_off: di },
                    Op::MambaScan { dt_bias: s0 + 5, a_log: s0 + 3, d: s0 + 4, state: ssm, cd, row, dt_off: di + cd, nh: mb.heads, ng: mb.groups },
                    Op::MambaNorm { w: s0 + 6, row, di, ng: mb.groups },
                    Op::Linear { w: b + 1, x: Slot::Core, y: Slot::O, name: "Mamba-2: out product" },
                ]);
            }
            NhKind::Attention => {
                mats.extend([it(["q", "k", "v"].iter().map(|x| one(r, x)).collect::<Result<_>>()?, qkv_w, hid), it(vec![one(r, "o")?], hid, nq * hd)]);
                let ring = rings.len();
                rings.push(Ring::kv(hd, nk, 0));
                let a = att.ok_or_else(|| Error("Nemotron-H: an attention layer of no kind".into()))?;
                let a = AttnOp::of(a, Slot::Mix, ring, None, None)?;
                if a.stride != qkv_w {
                    return err(format!("Nemotron-H: an attention product of {qkv_w} where its attention reads {}", a.stride));
                }
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                    Op::Attn(a),
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
    let lm = if d.head.is_none() { e } else { e + 1 };
    // (the final norm, the embedding and the head: named on every stage, a stage loading the ones it reads)
    small.push(d.final_norm.clone());
    mats.push(it(vec![d.embed.clone()], m.vocab, hid));
    if let Some(h) = &d.head {
        mats.push(it(vec![h.clone()], m.vocab, hid));
    }
    if (mats.len(), small.len()) != (lm + 1, n_small + 1) {
        return err(format!(
            "Nemotron-H: {} matrices and {} small tensors named, where its operations index {} and {}",
            mats.len(),
            small.len(),
            lm + 1,
            n_small + 1
        ));
    }
    let mut widths = [0usize; SLOTS];
    for (s, w) in [
        (Slot::X, hid),
        (Slot::H, hid),
        (Slot::O, hid),
        (Slot::Mix, row.max(qkv_w)),
        (Slot::Xc, cd),
        (Slot::Core, di),
        (Slot::Qo, nq * hd),
        (Slot::Att, nq * hd),
        (Slot::Hlast, hid),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Nemotron-H",
        hidden: hid,
        vocab: m.vocab,
        eps,
        max_pos,
        widths,
        mats,
        small,
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: first.unwrap_or(Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, weight: NormW::Plain, eps }),
        layers,
        states,
        final_norm: Some((n_small, Slot::O, false)),
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
        host: Vec::new(),
        context: Vec::new(),
    })
}

/// Muse Spark: pairs of blocks, the gated attention (q, k, v and the gate in one product, heads of 64, the rotary on the window's layers alone) with its input-scaled
/// norms and residual gate, then the experts in the latent with theirs; the embedding normed without a weight, the soft-capped head times the output multiplier.
/// Every matrix and small tensor is named from the description's roles, in the order the operations index them, a small tensor made from its values where its
/// operation reads it so (`SmallForm`): the executor's loader reads them (`dynamic::muse`), each layer's experts its kind's block (`dynamic::experts`).
pub fn muse(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("Muse Spark: {} blocks, where each layer is an attention block and an experts block", m.blocks.len()));
    }
    // (the executor's head: the product's bf16 output, then muse_softcap's multiply and soft-cap, each rounded)
    if m.head.rounding != crate::runtime::kinds::Rounding::Twice {
        return err(format!("Muse Spark: a head of {:?}, where the executor's rounds the logits to bf16 between operations", m.head));
    }
    let layers = m.blocks.len() / 2;
    let mut atts = Vec::with_capacity(layers);
    let (mut lat, mut mult, mut top_k) = (None, 1.0, 0);
    for l in 0..layers {
        match (&m.blocks[2 * l].body, &m.blocks[2 * l + 1].body) {
            (Body::Mixer(Mixer::Attention(a)), Body::Ffn(Ffn::Moe(e))) => {
                atts.push(a);
                lat = lat.or(e.latent.as_ref().map(|x| x.dim));
                (mult, top_k) = (e.router.logit_scale, top_k.max(e.router.top_k));
            }
            (a, f) => return err(format!("Muse Spark: layer {l} is {a:?} and {f:?}")),
        }
    }
    let a0 = atts.first().ok_or_else(|| Error("Muse Spark: no layers".into()))?;
    let (nq, nk, hd, lat) = (a0.heads, a0.kv_heads, a0.head_dim, lat.ok_or_else(|| Error("Muse Spark: experts without a latent".into()))?);
    let (eps, post_eps) = (m.blocks[0].norm_in.eps, m.blocks[0].norm_out.map_or(1e-6, |n| n.eps));
    if hd != 64 || m.hidden > 8192 || lat > 4096 || top_k > 32 {
        return err(format!(
            "heads of {hd}, a width of {}, a latent of {lat}, {top_k} experts a token: this engine runs heads of 64, widths up to 8192, latents up to 4096 and up to 32 experts a token",
            m.hidden
        ));
    }
    let row = (2 * nq + 2 * nk) * hd;
    let ones = 0;
    let (mut n_small, mut n_mats, mut n_moe, mut rings, mut ops_all) = (1, 0, 0, Vec::new(), Vec::new());
    // (the tensors, named in the order the operations index them; the small tensors made from their values: the ones, the centered norms' 1 + w, the gates' factors)
    use super::SmallForm::{GateAlpha, GateBeta, Ones, PlusOne};
    // (sized by the kinds' widths: the describer gives no shapes for this family's tensors)
    let it = |members: Vec<String>, rows: usize, cols: usize| Item { members, rows, cols };
    let one = |roles: &[(String, Source)], r: &str| -> Result<String> {
        match role(roles, r) {
            Some(Source::Tensor(n)) => Ok(n.clone()),
            other => err(format!("Muse Spark: the role {r} is {other:?}, where one tensor is")),
        }
    };
    let (mut mats, mut small, mut forms) = (Vec::new(), vec![String::new()], vec![(0, Ones(m.hidden.max(lat)))]);
    for (l, a) in atts.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            ops_all.push(Vec::new());
            continue;
        }
        let (ar, fr) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        mats.push(it(["q", "k", "v", "g"].iter().map(|r| one(ar, r)).collect::<Result<_>>()?, row, m.hidden));
        mats.push(it(vec![one(ar, "o")?], m.hidden, nq * hd));
        mats.push(it(vec![one(fr, "latent_down")?], lat, m.hidden));
        mats.push(it(vec![one(fr, "latent_up")?], m.hidden, lat));
        let temp = |b: &crate::runtime::kinds::Block| match b.residual {
            crate::runtime::kinds::Residual::Gated(g) => Ok(g.temperature),
            ref r => err(format!("Muse Spark: layer {l}'s residual is {r:?}, where a gated one is")),
        };
        let (ta, tf) = (temp(&m.blocks[2 * l])?, temp(&m.blocks[2 * l + 1])?);
        for (n, f) in [
            (one(ar, "norm_in")?, PlusOne),
            (one(ar, "res_gate")?, GateAlpha(ta)),
            (one(ar, "res_gate")?, GateBeta(ta)),
            (one(fr, "norm_in")?, PlusOne),
            (one(fr, "latent_norm_in")?, PlusOne),
            (one(fr, "expert_norm")?, PlusOne),
            (one(fr, "norm_out")?, PlusOne),
            (one(fr, "res_gate")?, GateAlpha(tf)),
            (one(fr, "res_gate")?, GateBeta(tf)),
        ] {
            forms.push((small.len(), f));
            small.push(n);
        }
        if st.last && l + 1 == layers {
            small.push(d.final_norm.clone());
        }
        let s0 = n_small;
        n_small += 9;
        let m0 = n_mats;
        n_mats += 4;
        let block = n_moe;
        n_moe += 1;
        let ring = rings.len();
        rings.push(Ring::kv(hd, nk, a.window.unwrap_or(0)));
        let attn = AttnOp::of(a, Slot::Mix, ring, Some(0), None)?;
        if attn.stride != row {
            return err(format!("Muse Spark: an attention product of {row} where its attention reads {}", attn.stride));
        }
        let mut ops = vec![
            Op::RmsNorm { x: Slot::X, w: s0, y: Slot::H, weight: NormW::Plain, eps },
            Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "attention: q, k, v and gate product" },
            Op::Attn(attn),
            Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
            Op::RmsNorm { x: Slot::O, w: ones, y: Slot::O, weight: NormW::Plain, eps: post_eps },
            Op::GatedResidual { alpha: s0 + 1, beta: s0 + 2 },
            Op::RmsNorm { x: Slot::X, w: s0 + 3, y: Slot::H, weight: NormW::Plain, eps },
            Op::Linear { w: m0 + 2, x: Slot::H, y: Slot::Proj, name: "experts: latent in" },
            Op::RmsNorm { x: Slot::Proj, w: s0 + 4, y: Slot::Xc, weight: NormW::Plain, eps },
            Op::Moe(super::experts::latent_normed(block, s0 + 5, post_eps, mult)),
            Op::Linear { w: m0 + 3, x: Slot::M, y: Slot::O, name: "experts: latent out" },
            Op::RmsNorm { x: Slot::O, w: s0 + 6, y: Slot::O, weight: NormW::Input, eps: post_eps },
            Op::GatedResidual { alpha: s0 + 7, beta: s0 + 8 },
        ];
        if st.last && l + 1 == layers {
            ops.push(Op::RmsNorm { x: Slot::X, w: n_small, y: Slot::H, weight: NormW::Plain, eps });
            n_small += 1;
        }
        ops_all.push(ops);
    }
    mats.push(it(vec![d.embed.clone()], m.vocab, m.hidden));
    mats.push(it(vec![d.head.clone().ok_or_else(|| Error("Muse Spark: no head of its own".into()))?], m.vocab, m.hidden));
    let d_ = m.hidden;
    let mut widths = [0usize; SLOTS];
    for (s, w) in [
        (Slot::X, d_),
        (Slot::H, d_),
        (Slot::Mix, row),
        (Slot::Qo, nq * hd),
        (Slot::Att, nq * hd),
        (Slot::O, d_),
        (Slot::Proj, lat),
        (Slot::Xc, lat),
        (Slot::M, lat),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Muse Spark",
        hidden: d_,
        vocab: m.vocab,
        eps,
        max_pos,
        widths,
        mats,
        small,
        forms,
        ropes: vec![Rope { theta: atts.iter().find_map(|a| a.rope.as_ref()).map_or(0.0, |r| r.theta), dim: hd, pairs: hd / 2, trig64: true }],
        rings,
        first: Op::RmsNorm { x: Slot::X, w: ones, y: Slot::X, weight: NormW::Plain, eps },
        layers: ops_all,
        states: Vec::new(),
        final_norm: None,
        decoded: false,
        skinny: false,
        embed: n_mats,
        lm: n_mats + 1,
        softcap: m.head.softcap.unwrap_or(0.0),
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: m.head.scale,
        host: Vec::new(),
        context: Vec::new(),
    })
}

/// GLM-5.3-Flash: the four streams mixed around each layer's two blocks, Kimi delta attention (its forget gate bounded) or the latent attention (its block's),
/// a dense MLP (its SwiGLU clamped) in the first layers and a mixture of experts in the rest (its block's), the mean of the streams and the final norm. Every
/// matrix and small tensor the operations use is named from the description's roles, in the order they index them (`dynamic::glm::describe`'s), sized by the
/// kinds' widths (a dense MLP's by its gate's shape): a layer's small tensors its two norms, the mixer block's stream mix (projection, base, scale), a delta
/// layer's three convolutions as one (`SmallForm::Joined`), A_log, dt_bias and gated norm or a latent layer's two norms, kv_b as saved and its indexer's, then
/// the feed-forward block's stream mix; the last stage's final norm. Matrices: a delta layer's q, k and v as one, o, the forget gate's two, the output gate's two
/// and beta; a latent layer's q_a, q_b, kv_a and o; a dense MLP's gate and up as one, and down; then the embedding and, on the last stage, the head. A layer's
/// experts are its block's, read apart (`dynamic::experts`).
pub fn glm(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::glm::{dense_ops, glm_desc, kda_ops, per_layer};
    use crate::glm::GlmKind;
    use crate::kern::GM_D;
    use crate::runtime::kinds::{Decay, IndexForm, NormKind, Residual};
    let m = &d.desc;
    let h = m.hidden;
    let one = |roles: &[(String, Source)], r: &str| match role(roles, r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        x => err(format!("GLM-5.3-Flash: the role {r} is {x:?}, where one tensor is")),
    };
    let it = |members: Vec<String>, rows: usize, cols: usize| Item { members, rows, cols };
    let (mut mats, mut small, mut forms) = (Vec::new(), Vec::new(), Vec::new());
    if m.blocks.len() % 2 != 0 || m.embed.streams != 4 {
        return err(format!(
            "GLM-5.3-Flash: {} blocks over {} streams, where each layer is a mixer block and a feed-forward block over 4",
            m.blocks.len(),
            m.embed.streams
        ));
    }
    let layers = m.blocks.len() / 2;
    let (hc, round) = match &m.blocks[0].residual {
        Residual::Streams(s) if !s.lagged => ((s.eps, s.sinkhorn_iters as i32), s.rounding),
        r => return err(format!("GLM-5.3-Flash: a residual of {r:?}, not mHC's streams")),
    };
    // (the executor's head: the product's output in bf16 (the release's lm_head), unscaled and uncapped)
    if m.head.rounding != crate::runtime::kinds::Rounding::Once || m.head.scale != 1.0 || m.head.softcap.is_some() {
        return err(format!("GLM-5.3-Flash: a head of {:?}, where the executor's rounds the product's output to bf16, unscaled and uncapped", m.head));
    }
    let lin = m.blocks.iter().find_map(|b| match &b.body {
        Body::Mixer(Mixer::Linear(x)) => Some(x),
        _ => None,
    });
    let (heads, qd, bound) = match lin {
        Some(x) if x.k_dim != GM_D => return err(format!("Kimi delta heads of {}: this engine runs heads of {GM_D}", x.k_dim)),
        Some(x) => match x.decay {
            Decay::Bounded(b) => (x.k_heads, x.k_heads * x.k_dim, b),
            dc => return err(format!("Kimi delta attention with a forget gate of {dc:?}: this engine runs the bounded gate")),
        },
        None => (0, 0, 0.0),
    };
    let (mut inter, mut limit) = (0, 0.0);
    let (mut n_small, mut n_mats, mut n_moe, mut states, mut ops_all, mut rings) = (0, 0, 0, Vec::new(), Vec::new(), Vec::new());
    for l in 0..layers {
        let (kind, mla) = match &m.blocks[2 * l].body {
            Body::Mixer(Mixer::Linear(_)) => (GlmKind::Kda, None),
            Body::Mixer(Mixer::Mla(x)) => (GlmKind::Mla, Some(x)),
            b => return err(format!("GLM-5.3-Flash: layer {l}'s mixer is {b:?}")),
        };
        let moe = match &m.blocks[2 * l + 1].body {
            Body::Ffn(Ffn::Mlp(mlp)) => {
                let gate = match role(&d.blocks[2 * l + 1], "gate") {
                    Some(Source::Tensor(n)) => n,
                    g => return err(format!("GLM-5.3-Flash: layer {l}'s MLP gate is {g:?}")),
                };
                inter = d.shapes.get(gate).map(|s| s[0]).ok_or_else(|| Error(format!("GLM-5.3-Flash: no shape for {gate}")))?;
                limit = mlp.clamp.ok_or_else(|| Error(format!("GLM-5.3-Flash: layer {l}'s MLP has no SwiGLU limit: this engine runs the clamped SwiGLU")))?;
                false
            }
            Body::Ffn(Ffn::Moe(_)) => true,
            b => return err(format!("GLM-5.3-Flash: layer {l}'s feed-forward is {b:?}")),
        };
        if !(st.layers.0..st.layers.1).contains(&l) {
            ops_all.push(Vec::new());
            continue;
        }
        let (a, f) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        for (roles, r) in [(a, "norm_in"), (f, "norm_in"), (a, "hc_fn"), (a, "hc_base"), (a, "hc_scale")] {
            small.push(one(roles, r)?);
        }
        let s = n_small;
        n_small += 5;
        let mut ops = vec![Op::StreamsPre { mix: StreamsMix::Hc { f: s + 2, base: s + 3, scale: s + 4 }, norm: s, round }];
        let m0 = n_mats;
        let nm = per_layer(kind, moe);
        n_mats += nm;
        if kind == GlmKind::Kda {
            // (q, k and v one product; o; the forget gate's two; the output gate's two; beta, a row a head)
            let kd = qd / heads.max(1);
            let qkv = [one(a, "q")?, one(a, "k")?, one(a, "v")?].to_vec();
            mats.extend([
                it(qkv, 3 * qd, h),
                it(vec![one(a, "o")?], h, qd),
                it(vec![one(a, "decay_a")?], kd, h),
                it(vec![one(a, "decay_b")?], qd, kd),
                it(vec![one(a, "gate_a")?], kd, h),
                it(vec![one(a, "gate_b")?], qd, kd),
                it(vec![one(a, "beta")?], heads, h),
            ]);
            let conv = match role(a, "conv") {
                Some(Source::Rows(v)) => v.join("|"),
                x => return err(format!("GLM-5.3-Flash: layer {l}'s convolutions are {x:?}, where the three are")),
            };
            forms.push((small.len(), super::SmallForm::Joined));
            small.push(conv);
            for r in ["a_log", "dt_bias", "norm"] {
                small.push(one(a, r)?);
            }
            let s0 = n_small;
            n_small += 4;
            let (conv, ssm) = (states.len(), states.len() + 1);
            states.push(3 * 3 * qd * 2);
            states.push(heads * GM_D * GM_D * 4);
            ops.extend(kda_ops(m0, s0, conv, ssm, qd, heads, bound));
        } else if let Some(x) = mla {
            // (q's and the key-value latent's products apart; the indexer's picks given as their tokens and the open pool's)
            let op = super::mla::MlaOp::of(x, super::mla::MlaIn::Apart { q_a: m0, kv_a: m0 + 2 }, m0 + 1, m0 + 3, n_small, rings.len(), true, None)?;
            let q_rank = x.q_lora.ok_or_else(|| Error(format!("GLM-5.3-Flash: layer {l}'s latent attention has no query latent")))?;
            mats.extend([
                it(vec![one(a, "q_a")?], q_rank, h),
                it(vec![one(a, "q_b")?], x.heads * (x.nope + x.rope_dim), q_rank),
                it(vec![one(a, "kv_a")?], x.kv_lora + x.rope_dim, h),
                it(vec![one(a, "o")?], h, x.heads * x.v_dim),
            ]);
            for r in ["q_a_norm", "kv_a_norm", "kv_b"] {
                small.push(one(a, r)?);
            }
            // (the indexer's, in the order the operation reads them: `MlaIndexer::at`)
            if let Some(ix) = &x.indexer {
                let layer = matches!(ix.form, IndexForm::Gated { key } if key.kind == NormKind::Layer);
                let pooled = op.indexer.as_ref().is_some_and(super::mla::MlaIndexer::pooled);
                let names = ["indexer_wq_b", "indexer_wk", "indexer_k_norm"]
                    .into_iter()
                    .chain(layer.then_some("indexer_k_norm_bias"))
                    .chain(["indexer_weights"])
                    .chain(if pooled { ["indexer_pool_ape", "indexer_pool_gate"].as_slice() } else { &[] }.iter().copied());
                for r in names {
                    small.push(one(a, r)?);
                }
            }
            n_small += op.small_count();
            rings.extend(op.rings());
            ops.push(Op::Mla(op));
        }
        for r in ["hc_fn", "hc_base", "hc_scale"] {
            small.push(one(f, r)?);
        }
        let hs = n_small;
        n_small += 3;
        ops.extend([
            Op::StreamsPost { round, gated: false },
            Op::StreamsPre { mix: StreamsMix::Hc { f: hs, base: hs + 1, scale: hs + 2 }, norm: s + 1, round },
        ]);
        if moe {
            ops.push(Op::Moe(MoeOp::of(n_moe)));
            n_moe += 1;
        } else {
            mats.extend([it(vec![one(f, "gate")?, one(f, "up")?], 2 * inter, h), it(vec![one(f, "down")?], h, inter)]);
            ops.extend(dense_ops(m0 + nm - 2, inter, limit));
        }
        ops.push(Op::StreamsPost { round, gated: false });
        ops_all.push(ops);
    }
    if st.last {
        ops_all.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::StreamsHead { mix: HeadMix::Mean, norm: n_small, round });
        small.push(d.final_norm.clone());
    }
    let e = n_mats;
    let lm = if st.last && d.head.is_some() { e + 1 } else { e };
    mats.push(it(vec![d.embed.clone()], m.vocab, h));
    if let Some(head) = d.head.as_ref().filter(|_| st.last) {
        mats.push(it(vec![head.clone()], m.vocab, h));
    }
    if (mats.len(), small.len()) != (lm + 1, n_small + usize::from(st.last)) {
        return err(format!(
            "GLM-5.3-Flash: {} matrices and {} small tensors named, where its operations index {} and {}",
            mats.len(),
            small.len(),
            lm + 1,
            n_small
        ));
    }
    let eps = m.blocks.first().map_or(1e-5, |b| b.norm_in.eps);
    let mut out = glm_desc(m.hidden, qd, heads, inter, m.vocab, eps, max_pos, eos, hc, ops_all, (states, rings), e, lm);
    (out.mats, out.small, out.forms) = (mats, small, forms);
    Ok(out)
}

/// Blocks of a latent attention and a feed-forward on the plain residual (GLM-5.3, `glm_moe_dsa`): x += mla(norm(x)), x += ffn(norm(x)) a layer, the
/// feed-forward a SwiGLU MLP or a block of experts (`dynamic::experts`), the final norm, the head; stage `st` (a model split across GPUs). The residual rounds
/// to bf16 at each add and every norm rounds as the kind says (`Rounding::Twice`, transformers' bf16 RMSNorm: `NormW::Rounded`; the reference's residual
/// stays f32). The latent attention's rotary is one table of the description's (interleaved pairs at the default frequencies: `MlaRot`); a `shared` indexer
/// takes the picks of the latent attention before it (`MlaOp::lent`), which must be of the same shape and in the stage. Every matrix and small tensor is
/// named from the description's roles in the order the operations index them: a layer's two norms, its latent attention's q latent norm, key-value latent
/// norm and kv_b, then its own indexer's (`MlaIndexer::at`'s order); the last stage's final norm. Matrices: a layer's q_a, q_b, kv_a, o, then a dense MLP's
/// gate and up as one and its down (a layer of experts holds its own); the embedding, then the head where it is not the embedding.
pub fn latent_moe(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::mla::{MlaIn, MlaOp};
    use crate::moe::Act;
    use crate::runtime::kinds::{Collapse, NormKind, NormWeight, Residual, Rounding};
    let m = &d.desc;
    let h = m.hidden;
    let what = "a latent attention model";
    if m.blocks.len() % 2 != 0 || m.embed.streams != 1 || m.embed.scale != 1.0 || m.embed.norm.is_some() {
        return err(format!(
            "{what}: {} blocks over {} streams, where each layer is a latent attention block and a feed-forward block on one",
            m.blocks.len(),
            m.embed.streams
        ));
    }
    if m.head.rounding != Rounding::Once || m.head.scale != 1.0 || m.head.softcap.is_some() || m.head.collapse != Collapse::None {
        return err(format!("{what}: a head of {:?}, where the executor's rounds the product's output to bf16, unscaled and uncapped", m.head));
    }
    let norms = m.blocks.iter().map(|b| &b.norm_in).chain([&m.final_norm]);
    if let Some(n) = norms.clone().find(|n| n.kind != NormKind::Rms || n.weight != NormWeight::Plain || n.rounding != Rounding::Twice) {
        return err(format!("{what}: a norm of {n:?}, where the executor's plain residual norms are RMS norms with a weight, rounded as transformers' bf16"));
    }
    let eps = m.final_norm.eps;
    if norms.clone().any(|n| n.eps != eps) {
        return err(format!("{what}: norms of more than one epsilon"));
    }
    let layers = m.blocks.len() / 2;
    let one = |roles: &[(String, Source)], r: &str| match role(roles, r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        x => err(format!("{what}: the role {r} is {x:?}, where one tensor is")),
    };
    let it = |members: Vec<String>, rows: usize, cols: usize| Item { members, rows, cols };
    // (the latent attention's rotary: one table, the first layer's kind's)
    let mut ropes = Vec::new();
    let (mut mats, mut small, mut rings, mut ops_all) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut n_mats, mut n_small, mut n_moe, mut inter) = (0, 0, 0, 0);
    let mut first = None;
    // (the pick the latent attention before a layer leaves, for a `shared` one: its shape, and whether it ran in this stage)
    let mut lender: Option<(super::IdxPick, bool)> = None;
    for l in 0..layers {
        let (ab, fb) = (&m.blocks[2 * l], &m.blocks[2 * l + 1]);
        if ab.residual != Residual::Add || fb.residual != Residual::Add || ab.norm_out.is_some() || fb.norm_out.is_some() {
            return err(format!("{what}: layer {l}'s residuals are {:?} and {:?}, where the executor's are the plain add", ab.residual, fb.residual));
        }
        let x = match &ab.body {
            Body::Mixer(Mixer::Mla(x)) => x,
            b => return err(format!("{what}: layer {l}'s first block is {b:?}, not a latent attention")),
        };
        let table = match (&x.rope, ropes.first()) {
            (None, _) => None,
            (Some(r), None) => {
                ropes.push(Rope { theta: r.theta, dim: r.dim, pairs: r.freqs, trig64: false });
                Some(0)
            }
            (Some(r), Some(t)) if (r.theta, r.dim, r.freqs) == (t.theta, t.dim, t.pairs) => Some(0),
            (Some(r), Some(t)) => return err(format!("{what}: layer {l}'s rotary {r:?}, where an earlier layer's is {t:?}: one table")),
        };
        let mine = (st.layers.0..st.layers.1).contains(&l);
        let (a, f) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        let m0 = n_mats;
        // (pools of tokens give the open pool's tokens after the best pools', the release's; a key a token has none)
        let tail = x.indexer.as_ref().is_some_and(|i| i.block > 1);
        let mut op = MlaOp::of(x, MlaIn::Apart { q_a: m0, kv_a: m0 + 2 }, m0 + 1, m0 + 3, n_small + 2, rings.len(), tail, table)?;
        // (a shared indexer's picks are the ones the latent attention before it made, of the same pick (its keys the lender's ring), in this stage)
        if let (Some(p), Some((q, _))) = (op.lent.as_mut(), &lender) {
            p.keys = q.keys;
        }
        match (&op.lent, &lender) {
            (Some(p), Some((q, here))) if p == q && (*here || !mine) => {}
            (Some(p), Some((q, _))) if p != q => return err(format!("{what}: layer {l} takes picks of {p:?} from a layer whose are {q:?}")),
            (Some(_), Some(_)) => return err(format!("{what}: layer {l} takes the picks of a layer of another stage: a split must keep them together")),
            (Some(_), None) => return err(format!("{what}: layer {l}'s indexer is shared, with no latent attention with an indexer before it")),
            (None, _) => {}
        }
        if let Some(p) = op.indexer.as_ref().map(|x| x.pick.clone()).or(op.lent.clone()) {
            lender = Some((p, mine || lender.as_ref().is_some_and(|x| x.1 && op.lent.is_some())));
        }
        let moe = match &fb.body {
            Body::Ffn(Ffn::Mlp(mlp)) => {
                if mlp.clamp.is_some() || !mlp.gated || mlp.bias || mlp.act != crate::runtime::kinds::Act::Silu {
                    return err(format!("{what}: layer {l}'s MLP of {mlp:?}, where the executor's is SwiGLU without a limit"));
                }
                let gate = match role(f, "gate") {
                    Some(Source::Tensor(n)) => n,
                    g => return err(format!("{what}: layer {l}'s MLP gate is {g:?}")),
                };
                inter = d.shapes.get(gate).map(|s| s[0]).ok_or_else(|| Error(format!("{what}: no shape for {gate}")))?;
                false
            }
            Body::Ffn(Ffn::Moe(e)) => {
                super::experts::cfg(e, h, 0).map_err(|x| x.context(format!("{what}: layer {l}")))?;
                true
            }
            b => return err(format!("{what}: layer {l}'s feed-forward is {b:?}")),
        };
        if !mine {
            ops_all.push(Vec::new());
            continue;
        }
        small.extend([one(a, "norm_in")?, one(f, "norm_in")?]);
        let s0 = n_small;
        let q_rank = x.q_lora.ok_or_else(|| Error(format!("{what}: layer {l}'s latent attention has no query latent")))?;
        mats.extend([
            it(vec![one(a, "q_a")?], q_rank, h),
            it(vec![one(a, "q_b")?], x.heads * (x.nope + x.rope_dim), q_rank),
            it(vec![one(a, "kv_a")?], x.kv_lora + x.rope_dim, h),
            it(vec![one(a, "o")?], h, x.heads * x.v_dim),
        ]);
        for r in ["q_a_norm", "kv_a_norm", "kv_b"] {
            small.push(one(a, r)?);
        }
        // (its own indexer's, in the order the operation reads them: `MlaIndexer::at`)
        if let Some(ix) = op.indexer.as_ref() {
            let layer = matches!(ix.key, super::mla::KeyNorm::Layer(_));
            let names = ["indexer_wq_b", "indexer_wk", "indexer_k_norm"]
                .into_iter()
                .chain(layer.then_some("indexer_k_norm_bias"))
                .chain(["indexer_weights"])
                .chain(if ix.pooled() { ["indexer_pool_ape", "indexer_pool_gate"].as_slice() } else { &[] }.iter().copied());
            for r in names {
                small.push(one(a, r)?);
            }
        }
        n_small += 2 + op.small_count();
        n_mats += 4;
        let rounded = |w: usize| Op::RmsNorm { x: Slot::X, w, y: Slot::H, weight: NormW::Rounded, eps };
        let mut ops = Vec::new();
        if l == st.layers.0 && st.first {
            first = Some(rounded(s0));
        } else {
            ops.extend([Op::Add { x: Slot::D }, rounded(s0)]);
        }
        rings.extend(op.rings());
        ops.extend([Op::Mla(op), Op::Add { x: Slot::O }, rounded(s0 + 1)]);
        if moe {
            ops.push(Op::Moe(MoeOp { y: Slot::D, ..MoeOp::of(n_moe) }));
            n_moe += 1;
        } else {
            mats.extend([it(vec![one(f, "gate")?, one(f, "up")?], 2 * inter, h), it(vec![one(f, "down")?], h, inter)]);
            ops.extend([
                Op::Linear { w: n_mats, x: Slot::H, y: Slot::Gu, name: "MLP: gate and up product" },
                Op::GatedAct { inter, act: Act::SwiGlu { limit: None } },
                Op::Linear { w: n_mats + 1, x: Slot::M, y: Slot::D, name: "MLP: down product" },
            ]);
            n_mats += 2;
        }
        ops_all.push(ops);
    }
    if st.last {
        // (the last layer's add, then the final norm into H, which the head reads)
        small.push(d.final_norm.clone());
        let last = ops_all.last_mut().ok_or_else(|| Error(format!("{what}: no layers")))?;
        last.extend([Op::Add { x: Slot::D }, Op::RmsNorm { x: Slot::X, w: n_small, y: Slot::H, weight: NormW::Rounded, eps }]);
        n_small += 1;
    }
    let e = n_mats;
    let lm = if st.last && d.head.is_some() { e + 1 } else { e };
    mats.push(it(vec![d.embed.clone()], m.vocab, h));
    if let Some(head) = d.head.as_ref().filter(|_| st.last) {
        mats.push(it(vec![head.clone()], m.vocab, h));
    }
    if (mats.len(), small.len()) != (lm + 1, n_small) {
        return err(format!("{what}: {} matrices and {} small tensors named, where its operations index {} and {n_small}", mats.len(), small.len(), lm + 1));
    }
    let mut widths = [0usize; SLOTS];
    for (s, w) in [(Slot::X, h), (Slot::H, h), (Slot::O, h), (Slot::D, h), (Slot::Gu, 2 * inter), (Slot::M, inter), (Slot::Hlast, h)] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "GLM-5.3",
        hidden: h,
        vocab: m.vocab,
        eps,
        max_pos,
        widths,
        mats,
        small,
        forms: Vec::new(),
        ropes,
        rings,
        first: first.unwrap_or(Op::Nop),
        layers: ops_all,
        states: Vec::new(),
        final_norm: None,
        decoded: false,
        skinny: false,
        embed: e,
        lm,
        softcap: 0.0,
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
        host: Vec::new(),
        context: Vec::new(),
    })
}

/// `d` with no tensor named (its operations, slots, rings and states alone): what a family's own description of it says, which names none.
pub(crate) fn nameless(d: &ModelDesc) -> ModelDesc {
    ModelDesc { mats: Vec::new(), small: Vec::new(), forms: Vec::new(), ..d.clone() }
}

/// A fused product's rows (several tensors' stacked) are padded with zero rows to a multiple of this: 64 a GPU in groups of up to 8.
pub(crate) const FUSED_ROWS: usize = 512;

/// A fused product's rows (several tensors' stacked): padded with zero rows to a multiple of `FUSED_ROWS`.
pub(crate) fn fused(rows: usize) -> usize {
    rows.div_ceil(FUSED_ROWS) * FUSED_ROWS
}

/// `members` stacked by rows (`item`), then zero rows to a multiple of `to` rows (`weights::ZERO_ROWS`): a fused product whose parts' rows the library does not
/// pack as they are, and that a group of GPUs slices in 64s whatever the group. No product reads the zero rows.
fn padded(members: Vec<String>, shape: Shape, to: usize) -> Result<Item> {
    let mut it = item(members, shape)?;
    it.rows = it.rows.div_ceil(to) * to;
    it.members.push(crate::weights::ZERO_ROWS.to_string());
    Ok(it)
}

/// Kimi-K3: Kimi delta attention (its forget gate bounded) or gated NoPE latent attention (its block's), a dense MLP in the first layers and latent experts with
/// shared ones in the rest (their block's), every block's input a depth mix over the groups of layers before it; SiTU. `w32` the delta layers' dtypes as the
/// checkpoint holds them, `early` whether the cached experts are guessed ahead. Every matrix and small tensor the operations use is named from the
/// description's roles (`kimi_names`); the blocks of experts are read apart, in the layers' order.
pub fn kimi(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage, w32: &[(bool, bool)], early: bool) -> Result<ModelDesc> {
    use super::kimi::{kimi_desc, KimiDims, KimiLayer};
    use crate::runtime::kinds::{Act, Decay, Residual};
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("Kimi-K3: {} blocks, where each layer is a mixer block and a feed-forward block", m.blocks.len()));
    }
    let layers = m.blocks.len() / 2;
    let shape0 = |n: &str| d.shapes.get(n).map(|s| s[0]).ok_or_else(|| Error(format!("Kimi-K3: no shape for {n}")));
    let tensor = |b: usize, r: &str| match role(&d.blocks[b], r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        x => err(format!("Kimi-K3: block {b}'s {r} is {x:?}")),
    };
    let (mut qd, mut nh, mut kd, mut bound) = (0, 0, 0, 0.0);
    let (mut inter, mut situ, mut moe_rows, mut mla, mut mla_ix) = (0, (0.0, 0.0), 0, None, None);
    let (mut ls, mut starts) = (Vec::with_capacity(layers), Vec::with_capacity(layers));
    for l in 0..layers {
        let (mx, ff) = (&m.blocks[2 * l], &m.blocks[2 * l + 1]);
        match (&mx.residual, &ff.residual) {
            (Residual::DepthMix(a), Residual::DepthMix(b)) if !b.starts => starts.push(a.starts),
            r => return err(format!("Kimi-K3: layer {l}'s residuals are {r:?}, not the depth mix")),
        }
        let kda = match &mx.body {
            Body::Mixer(Mixer::Linear(x)) => {
                let &Decay::Bounded(b) = &x.decay else {
                    return err(format!("Kimi delta attention with a forget gate of {:?}: this engine runs the bounded gate", x.decay));
                };
                (qd, nh, kd, bound) = (x.k_heads * x.k_dim, x.k_heads, x.k_dim, b);
                let fa = shape0(&tensor(2 * l, "decay_a")?)?;
                if fa != kd {
                    return err(format!("Kimi-K3: a forget gate of rank {fa}, where the delta heads are {kd}: this engine runs the rank of a head"));
                }
                true
            }
            Body::Mixer(Mixer::Mla(x)) => {
                // (q_a, the gate and kv_a one product; with the newer config's indexer on the layers it names)
                let op = super::mla::MlaOp::of(x, super::mla::MlaIn::Fused { w: 0 }, 0, 0, 0, 0, false, None)?;
                if x.indexer.is_some() {
                    mla_ix = Some(op);
                } else {
                    mla = Some(op);
                }
                false
            }
            b => return err(format!("Kimi-K3: layer {l}'s mixer is {b:?}")),
        };
        let moe = match &ff.body {
            Body::Ffn(Ffn::Mlp(mlp)) => {
                inter = shape0(&tensor(2 * l + 1, "gate")?)?;
                let &Act::Situ { beta, linear } = &mlp.act else { return err(format!("Kimi-K3: layer {l}'s MLP is {:?}, not SiTU", mlp.act)) };
                situ = (beta, linear);
                false
            }
            Body::Ffn(Ffn::Moe(e)) => {
                let lat = e.latent.as_ref().map(|x| x.dim).ok_or_else(|| Error("Kimi-K3: experts without a latent".into()))?;
                let shared = if e.shared.is_some() { shape0(&tensor(2 * l + 1, "shared_gate")?)? } else { 0 };
                moe_rows = lat + 2 * shared;
                true
            }
            b => return err(format!("Kimi-K3: layer {l}'s feed-forward is {b:?}")),
        };
        let ix = matches!(&mx.body, Body::Mixer(Mixer::Mla(x)) if x.indexer.is_some());
        ls.push(KimiLayer { kda, moe, w32: w32.get(l).copied().unwrap_or_default(), ix });
    }
    let eps = m.blocks.first().map_or(1e-6, |b| b.norm_in.eps);
    let mut out =
        kimi_desc(&KimiDims { h: m.hidden, qd, nh, kd, bound, inter, situ, eps, moe_rows, mla, mla_ix }, &ls, &starts, st, early, m.vocab, max_pos, eos);
    (out.mats, out.small, out.forms) = kimi_names(d, st)?;
    Ok(out)
}

/// Kimi-K3's tensors for stage `st`, in the order its operations index them (`dynamic::kimi::kimi_desc`), sized by the description's shapes. Small tensors: each
/// layer's input and post-attention norms, its two depth mixes' norms and projections; then a delta layer's three convolutions as one (`SmallForm::Joined`),
/// A_log, dt_bias and the gated norm, or a latent layer's two norms, kv_b as saved (the absorbed form multiplies by its rows) and its indexer's (the queries'
/// and the keys' products, the keys' norm, its bias where a LayerNorm, the heads' weights); the last stage's output mix's norm and projection and the final
/// norm. Matrices: a delta layer's one product (q, k, v, the full-rank gate, the forget gate's first product and beta, zero rows to 512s), o and the forget
/// gate's second product; a latent layer's one product (q_a, the gate, kv_a, padded alike), q_b and o; a dense MLP's gate and up (padded alike) and down;
/// then the embedding and the head. A layer's experts are its block's, not among them.
#[allow(clippy::type_complexity)]
fn kimi_names(d: &Described, st: &super::Stage) -> Result<(Vec<Item>, Vec<String>, Vec<(usize, super::SmallForm)>)> {
    use crate::runtime::kinds::{IndexForm, NormKind};
    let m = &d.desc;
    let shape = |n: &str| d.shapes.get(n).cloned();
    let one = |roles: &[(String, Source)], r: &str| match role(roles, r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        x => err(format!("Kimi-K3: the role {r} is {x:?}, where one tensor is")),
    };
    let fused =
        |roles: &[(String, Source)], rs: &[&str]| -> Result<Item> { padded(rs.iter().map(|r| one(roles, r)).collect::<Result<_>>()?, &shape, FUSED_ROWS) };
    let (mut mats, mut small, mut forms) = (Vec::new(), Vec::new(), Vec::new());
    for l in st.layers.0..st.layers.1 {
        let (a, f) = (&d.blocks[2 * l], &d.blocks[2 * l + 1]);
        for (roles, r) in [(a, "norm_in"), (f, "norm_in"), (a, "mix_norm"), (a, "mix_proj"), (f, "mix_norm"), (f, "mix_proj")] {
            small.push(one(roles, r)?);
        }
        match &m.blocks[2 * l].body {
            Body::Mixer(Mixer::Linear(_)) => {
                mats.extend([
                    fused(a, &["q", "k", "v", "gate", "decay_a", "beta"])?,
                    item(vec![one(a, "o")?], &shape)?,
                    item(vec![one(a, "decay_b")?], &shape)?,
                ]);
                let conv = match role(a, "conv") {
                    Some(Source::Rows(v)) => v.join("|"),
                    x => return err(format!("Kimi-K3: layer {l}'s convolutions are {x:?}, where the three are")),
                };
                forms.push((small.len(), super::SmallForm::Joined));
                small.push(conv);
                for r in ["a_log", "dt_bias", "norm"] {
                    small.push(one(a, r)?);
                }
            }
            Body::Mixer(Mixer::Mla(x)) => {
                mats.extend([fused(a, &["q_a", "g", "kv_a"])?, item(vec![one(a, "q_b")?], &shape)?, item(vec![one(a, "o")?], &shape)?]);
                for r in ["q_a_norm", "kv_a_norm", "kv_b"] {
                    small.push(one(a, r)?);
                }
                if let Some(ix) = &x.indexer {
                    let layer = matches!(ix.form, IndexForm::Gated { key } if key.kind == NormKind::Layer);
                    for r in
                        ["indexer_wq_b", "indexer_wk", "indexer_k_norm"].into_iter().chain(layer.then_some("indexer_k_norm_bias")).chain(["indexer_weights"])
                    {
                        small.push(one(a, r)?);
                    }
                }
            }
            b => return err(format!("Kimi-K3: layer {l}'s mixer is {b:?}")),
        }
        if let Body::Ffn(Ffn::Mlp(_)) = &m.blocks[2 * l + 1].body {
            mats.extend([fused(f, &["gate", "up"])?, item(vec![one(f, "down")?], &shape)?]);
        }
    }
    if st.last {
        small.extend([one(&d.extra, "out_mix_norm")?, one(&d.extra, "out_mix_proj")?, d.final_norm.clone()]);
    }
    mats.push(item(vec![d.embed.clone()], &shape)?);
    mats.push(item(vec![d.head.clone().unwrap_or_else(|| d.embed.clone())], &shape)?);
    Ok((mats, small, forms))
}

/// A model of compressed attention layers (DeepSeek-V4's and V4.1's forms of the kind; `lower_comp::lower`, which gives its loader's plan too). `vo` what the level and
/// the switches make of its products and caches.
pub fn deepseek_v4(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage, vo: &super::deepseek_v4::V4Opts) -> Result<ModelDesc> {
    super::lower_comp::lower(d, eos, max_pos, st, vo).map(|x| x.0)
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
            // (the mixtures of experts are refused by the family's reader: lowered from their kinds alone, with no description of the reader's to hold them to)
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

    /// Nemotron-H lowered from Nemotron 3 Super's released config and from the mixture-of-experts toy's, whole and as a split's stages, is the executor's own
    /// description of it with its names left out, and names only the description's tensors.
    #[test]
    fn nemotron_lowered_is_the_executors_own_description() {
        for f in ["nemotron_h_config.json", "nemotron_h_moetoy_config.json"] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f);
            let text = std::fs::read_to_string(p).unwrap();
            let c = crate::nemotron::ConfigNemotronH::parse(&text, f, None).unwrap();
            let kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
            let half = c.layers / 2;
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let (own, lowered) = match super::super::nemotron::describe(&c, &st) {
                    // (a shape the executor does not run is refused alike)
                    Err(e) => {
                        assert!(
                            nemotron(&kinds, c.eos.clone(), c.max_pos, &st).is_err(),
                            "{f}: refused by the executor's own ({}), not by the lowering",
                            e.msg
                        );
                        continue;
                    }
                    Ok(own) => (own, nemotron(&kinds, c.eos.clone(), c.max_pos, &st).unwrap_or_else(|e| panic!("{f}: {}", e.msg))),
                };
                // (every tensor it names is the description's)
                let names = kinds.names();
                assert_eq!(super::super::unnamed(&lowered), None, "{f}");
                assert!(lowered.mats.iter().flat_map(|it| &it.members).chain(&lowered.small).all(|m| names.contains(m)), "{f}");
                let (a, b) = (format!("{own:#?}"), format!("{:#?}", nameless(&lowered)));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("{f}: the lowered description differs at line {first:?}");
                }
            }
        }
    }

    /// A config.json whose layer kinds and family reader would describe different models is not loaded: here the builder made to see another MLP width than the
    /// reader (as a config the two read differently would), and the difference is named.
    #[test]
    fn two_readings_that_differ_are_not_loaded() {
        let (text, _) = header("gemma-4-12B-it.json");
        let c = crate::gemma4::ConfigG4::parse(&text, None, false).unwrap();
        let own = super::super::gemma4::describe(&c);
        let mut kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
        for s in kinds.shapes.values_mut().filter(|s| s.len() == 2 && s[0] == 15360) {
            s[0] = 15360 * 2;
        }
        let lowered = gemma4(&kinds, &|n: &str| kinds.shapes.get(n).cloned(), c.eos.clone(), c.max_pos).unwrap();
        let e = agree(lowered, &own, "the test's config.json").unwrap_err().msg;
        assert!(e.contains("describe different models") && e.contains("not loaded"), "{e}");
    }

    /// GLM-5.3-Flash lowered from its released config and the tiny and toy ones, whole and as a split's stages, is the executor's own description of it with its
    /// names left out; every tensor it names is the description's, and the released config's first layer (delta attention, a dense MLP) names its seven and two
    /// matrices at the release's shapes.
    #[test]
    fn glm_lowered_is_the_executors_own_description() {
        // (and the released one with every layer's feed-forward its experts, as glmrandom --real makes it: no dense MLP, so neither description sizes its slots)
        let fixture = |f: &str| std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap();
        let all_experts = {
            let mut v: serde_json::Value = serde_json::from_str(&fixture("glm5_next_config.json")).unwrap();
            let t = if v.get("text_config").is_some() { &mut v["text_config"] } else { &mut v };
            let n = t["num_hidden_layers"].as_u64().unwrap() as usize;
            t["first_k_dense_replace"] = 0.into();
            t["mlp_layer_types"] = serde_json::Value::Array(vec!["sparse".into(); n]);
            v.to_string()
        };
        for f in ["glm5_next_config.json", "glm5_next_tiny_config.json", "glm5_next_toy_config.json", "all experts"] {
            let text = if f == "all experts" { all_experts.clone() } else { fixture(f) };
            let c = crate::glm::ConfigGlm::parse(&text, f, None).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
            let kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
            if f == "glm5_next_config.json" {
                let d = glm(&kinds, c.eos.clone(), c.max_pos, &super::super::Stage::whole(c.layers)).unwrap();
                let shapes: Vec<(usize, usize)> = d.mats[..9].iter().map(|it| (it.rows, it.cols)).collect();
                assert_eq!(shapes, [(24576, 4096), (4096, 8192), (128, 4096), (8192, 128), (128, 4096), (8192, 128), (64, 4096), (24576, 4096), (4096, 12288)]);
                assert!(d.mats[0].members.iter().zip(["q_proj", "k_proj", "v_proj"]).all(|(m, x)| m.ends_with(&format!("layers.0.self_attn.{x}.weight"))));
            }
            let half = c.layers / 2;
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let own = match super::super::glm::describe(&c, &st) {
                    Ok(x) => x,
                    // (a config the executor does not run: the lowering refuses it too)
                    Err(e) => {
                        assert!(glm(&kinds, c.eos.clone(), c.max_pos, &st).is_err(), "{f}: refused by the executor's own reading ({}) alone", e.msg);
                        continue;
                    }
                };
                let lowered = glm(&kinds, c.eos.clone(), c.max_pos, &st).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
                assert_eq!(super::super::unnamed(&lowered), None, "{f}");
                let names = kinds.names();
                assert!(lowered.mats.iter().flat_map(|it| &it.members).all(|m| names.contains(m)), "{f}");
                assert!(lowered.small.iter().flat_map(|s| s.split('|')).all(|s| names.contains(s)), "{f}");
                let (a, b) = (format!("{own:#?}"), format!("{:#?}", nameless(&lowered)));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("{f}: the lowered description differs at line {first:?}");
                }
            }
        }
    }

    /// GLM-5.3's layout lowered by its kinds alone (`latent_moe`), from its released config and the random case's, whole and as a split's stages: every tensor it
    /// names is the description's; each latent attention turns its shared key part and its indexer's first 64 channels by the one rotary table; a layer whose
    /// indexer is shared takes the picks of the one before it (`MlaOp::lent`, the same pick); a split between a layer that picks and one that takes its picks is
    /// refused, in words.
    #[test]
    fn glm53_lowers_by_its_kinds() {
        use super::super::mla::MlaOp;
        use super::super::Stage;
        let fixture = |f: &str| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap()).unwrap()
        };
        for f in ["glm_moe_dsa_config.json", "glm_moe_dsa_toy_config.json"] {
            let kinds = crate::runtime::describe::describe(&fixture(f)).unwrap();
            let layers = kinds.desc.blocks.len() / 2;
            let whole = latent_moe(&kinds, vec![1], 4096, &Stage::whole(layers)).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
            let names = kinds.names();
            assert!(whole.mats.iter().flat_map(|it| &it.members).all(|m| names.contains(m)), "{f}");
            assert!(whole.small.iter().all(|s| names.contains(s)), "{f}");
            assert_eq!((whole.ropes.len(), whole.ropes[0].pairs), (1, 32), "{f}");
            let mla: Vec<&MlaOp> = whole.layers.iter().flatten().filter_map(|o| if let Op::Mla(a) = o { Some(a) } else { None }).collect();
            assert_eq!(mla.len(), layers, "{f}");
            for (l, a) in mla.iter().enumerate() {
                // (a layer that takes another's picks turns no indexer channel: it has no indexer)
                assert_eq!(a.rot.map(|r| (r.table, r.index)), Some((0, if a.lent.is_some() { 0 } else { 64 })), "{f} layer {l}");
                assert_eq!(a.lent.is_some(), a.indexer.is_none(), "{f} layer {l}");
                if let (Some(p), Some(before)) = (&a.lent, l.checked_sub(1).map(|b| mla[b])) {
                    assert_eq!(Some(p), before.indexer.as_ref().map(|x| &x.pick).or(before.lent.as_ref()), "{f} layer {l}");
                }
            }
            // (each layer's rings: the latents', and its indexer's keys where it picks)
            assert_eq!(whole.rings.len(), layers + mla.iter().filter(|a| a.indexer.is_some()).count(), "{f}");
        }
        // (the random case: full, full, shared, shared; a stage of the last two takes picks the first stage made)
        let kinds = crate::runtime::describe::describe(&fixture("glm_moe_dsa_toy_config.json")).unwrap();
        assert!(latent_moe(&kinds, vec![1], 4096, &Stage { layers: (0, 2), first: true, last: false }).is_ok());
        let e = latent_moe(&kinds, vec![1], 4096, &Stage { layers: (2, 4), first: false, last: true }).unwrap_err().msg;
        assert!(e.contains("another stage"), "{e}");
    }

    /// GLM-5.3's draft head (its MTP layer, DeepSeek-V3's joined form) lowers by the model's own lowering (`latent_moe`) to a draft part: `Op::MtpIn`'s joined
    /// front (the two norms and eh_proj [hidden, 2 hidden] of the layer after the model's), then the head's layer as the model's (its first norm, the latent
    /// attention with an indexer of its own, the experts, the head's final norm), two rings of its own (its latents', its indexer's keys) that move after the
    /// model's, the model's embedding and LM head named; beside the model it fits. The release's head is the same operations at its shapes.
    #[cfg(feature = "dev")]
    #[test]
    fn the_glm_draft_head_lowers_to_a_draft_part() {
        use super::super::{head, MtpFront, Stage};
        let fixture = |f: &str| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap()).unwrap()
        };
        let mut toy = fixture("glm_moe_dsa_toy_config.json");
        toy["num_nextn_predict_layers"] = 1.into();
        for (f, c, at) in [("toy", toy, 4), ("release", fixture("glm_moe_dsa_config.json"), 78)] {
            let d = crate::runtime::describe::describe(&c).unwrap();
            let layers = d.desc.blocks.len() / 2;
            let big = latent_moe(&d, vec![1], 4096, &Stage::whole(layers)).unwrap();
            let whole = |d: &Described, eos: Vec<i64>, mp: usize| latent_moe(d, eos, mp, &Stage::whole(d.desc.blocks.len() / 2));
            let mut h = draft_head(&d, vec![1], 4096, whole).unwrap_or_else(|e| panic!("{f}: {}", e.msg));
            let Op::MtpIn { norm_embed, norm_hidden, front: MtpFront::Joined { eh } } = h.first else { panic!("{f}: {:?}", h.first) };
            let p = format!("model.layers.{at}");
            assert_eq!(
                (h.small[norm_embed].as_str(), h.small[norm_hidden].as_str(), h.mats[eh].members[0].as_str()),
                (format!("{p}.enorm.weight").as_str(), format!("{p}.hnorm.weight").as_str(), format!("{p}.eh_proj.weight").as_str())
            );
            assert_eq!((h.mats[eh].rows, h.mats[eh].cols, h.width(Slot::Hid), h.width(Slot::In)), (h.hidden, 2 * h.hidden, h.hidden, 2 * h.hidden), "{f}");
            let names: Vec<String> = h.layers[0].iter().map(super::super::rows::op_name).collect();
            assert_eq!(names, ["RmsNorm", "Mla", "Add", "RmsNorm", "Moe", "Add", "RmsNorm"], "{f}");
            let Some(Op::RmsNorm { w, .. }) = h.layers[0].last() else { panic!() };
            assert_eq!(h.small[*w], format!("{p}.shared_head.norm.weight"), "{f}");
            assert!(matches!(&h.layers[0][1], Op::Mla(a) if a.indexer.is_some() && a.lent.is_none()), "{f}");
            assert!(h
                .mats
                .iter()
                .flat_map(|it| &it.members)
                .all(|n| n.starts_with(&p) || n == &big.mats[big.embed].members[0] || n == &big.mats[big.lm].members[0]));
            assert_eq!(
                (h.mats[h.embed].members.clone(), h.mats[h.lm].members.clone()),
                (big.mats[big.embed].members.clone(), big.mats[big.lm].members.clone())
            );
            head::fits(&big, &h).unwrap();
            head::shift_rings(&mut h, big.rings.len()).unwrap();
            let rings: Vec<usize> = h.layers[0].iter().flat_map(|op| op.rings()).collect();
            assert_eq!(rings, [big.rings.len(), big.rings.len() + 1], "{f}");
            let Op::Mla(a) = &h.layers[0][1] else { unreachable!() };
            assert_eq!(a.indexer.as_ref().map(|x| x.pick.keys), Some(super::super::IdxKeys::Ring(big.rings.len() + 1)), "{f}");
        }
    }

    /// The executor's GLM head is the product's output in bf16, as the release's lm_head gives the logits: a description whose head rounds otherwise is
    /// refused, named.
    #[test]
    fn a_glm_head_that_rounds_otherwise_is_refused() {
        let text = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_next_toy_config.json")).unwrap();
        let c = crate::glm::ConfigGlm::parse(&text, "toy", None).unwrap();
        let mut kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
        let whole = super::super::Stage::whole(c.layers);
        assert!(glm(&kinds, c.eos.clone(), c.max_pos, &whole).is_ok());
        kinds.desc.head.rounding = crate::runtime::kinds::Rounding::F32;
        let e = glm(&kinds, c.eos.clone(), c.max_pos, &whole).unwrap_err().msg;
        assert!(e.contains("a head of") && e.contains("bf16"), "{e}");
    }

    /// The tensors the GLM family's lowering names are the ones its own loader read before the executor's loader took them over (src/glm.rs's `layer_packs`
    /// and `layer_kept`, in that loader's order), at the checkpoint's shapes, on the random models (bf16, and FP8 experts) whole and as a split's stages: each
    /// layer's two norms and its attention block's stream mix, then a delta layer's convolutions (the three as one), A_log, dt_bias and gated norm, or a latent
    /// layer's two norms, kv_b and its indexer's seven; then the feed-forward block's stream mix; the last stage's final norm. Matrices: a delta layer's q, k and
    /// v as one, o, f_a, f_b, g_a, g_b, b; a latent layer's q_a, q_b, kv_a, o; a dense MLP's gate and up as one, down; the embedding and the head.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_glms_tensors_are_named_as_the_loader_reads_them() {
        use crate::glm::GlmKind;
        for fp8 in [false, true] {
            let dir = std::env::temp_dir().join(format!("glyd-lower-glm-names-{}-{fp8}", std::process::id()));
            crate::glm::random(&dir, fp8, None).unwrap();
            let c = crate::glm::ConfigGlm::load(&dir).unwrap();
            let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
            let half = c.layers / 2;
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let d = super::super::glm::lowered(&dir, &c, &st).unwrap_or_else(|e| panic!("fp8 {fp8}: {}", e.msg));
                let (mut mats, mut small): (Vec<(Vec<String>, usize, usize)>, Vec<String>) = (Vec::new(), Vec::new());
                for l in st.layers.0..st.layers.1 {
                    let (packs, kept) = (c.layer_packs(l), c.layer_kept(l));
                    let name = |t: &str| kept.iter().find(|x| x.name.ends_with(t)).unwrap_or_else(|| panic!("layer {l}: {t}")).name.clone();
                    small.extend(kept[..5].iter().map(|t| t.name.clone()));
                    let (attn, mlp) = packs.split_at(packs.len() - 2);
                    let kda = c.kinds[l] == GlmKind::Kda;
                    for p in attn.iter().filter(|p| kda || !p.key.ends_with(".kv_b_proj")) {
                        mats.push((p.parts.iter().map(|x| x.0.clone()).collect(), p.rows(), p.cols));
                    }
                    if kda {
                        small.push(["q", "k", "v"].map(|x| name(&format!("self_attn.{x}_conv1d.weight"))).join("|"));
                        small.extend(["self_attn.A_log", "self_attn.dt_bias", "self_attn.o_norm.weight"].map(name));
                    } else {
                        small.extend(["q_a_layernorm.weight", "kv_a_layernorm.weight"].map(|x| name(&format!("self_attn.{x}"))));
                        small.push(format!("{}layers.{l}.self_attn.kv_b_proj.weight", c.prefix));
                        let ix = [
                            "wq_b.weight",
                            "wk.weight",
                            "k_norm.weight",
                            "k_norm.bias",
                            "weights_proj.weight",
                            "index_kpool_compress_ape",
                            "index_kpool_compress_gate",
                        ];
                        small.extend(ix.map(|x| name(&format!("self_attn.indexer.{x}"))));
                    }
                    small.extend(kept[5..8].iter().map(|t| t.name.clone()));
                    if !c.moe[l] {
                        for p in mlp {
                            mats.push((p.parts.iter().map(|x| x.0.clone()).collect(), p.rows(), p.cols));
                        }
                    }
                }
                if st.last {
                    small.push(format!("{}norm.weight", c.prefix));
                }
                mats.push((vec![format!("{}embed_tokens.weight", c.prefix)], c.vocab, c.hidden));
                if st.last && !c.tie {
                    mats.push((vec!["lm_head.weight".into()], c.vocab, c.hidden));
                }
                let got: Vec<(Vec<String>, usize, usize)> = d.mats.iter().map(|it| (it.members.clone(), it.rows, it.cols)).collect();
                assert_eq!(got, mats, "fp8 {fp8}, {st:?}");
                assert_eq!(d.small, small, "fp8 {fp8}, {st:?}");
                // (each named at the checkpoint's shape: a matrix's members' rows add up, their columns its own)
                for (members, rows, cols) in &got {
                    let shapes: Vec<Vec<u64>> = members.iter().map(|m| ck.get(m).unwrap_or_else(|| panic!("no tensor {m}")).1.shape.clone()).collect();
                    assert!(
                        shapes.iter().all(|s| s.len() == 2 && s[1] as usize == *cols) && shapes.iter().map(|s| s[0] as usize).sum::<usize>() == *rows,
                        "{members:?}"
                    );
                }
                assert!(d.small.iter().flat_map(|s| s.split('|')).all(|s| ck.get(s).is_some()));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The random GLM-5.3-Flash the logits and GPU checks run (bf16, and its experts in FP8), lowered whole and as a split's stages, is the executor's own description of it.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_glm_lowers_as_its_own_description() {
        for fp8 in [false, true] {
            let dir = std::env::temp_dir().join(format!("glyd-lower-glm-{}-{fp8}", std::process::id()));
            crate::glm::random(&dir, fp8, None).unwrap();
            let c = crate::glm::ConfigGlm::load(&dir).unwrap();
            let half = c.layers / 2;
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let (own, lowered) = (
                    super::super::glm::describe(&c, &st).unwrap(),
                    super::super::glm::lowered(&dir, &c, &st).unwrap_or_else(|e| panic!("fp8 {fp8}: {}", e.msg)),
                );
                assert_eq!(format!("{own:#?}"), format!("{:#?}", nameless(&lowered)));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Kimi-K3 lowered from its released config (the header fixture's), whole and as a split's stages, with the experts guessed ahead and not, is the executor's own
    /// description of it.
    #[test]
    fn kimi_lowered_is_the_executors_own_description() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/Kimi-K3.json");
        let h: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let text = h["config"].as_str().unwrap();
        let c = crate::kimi::ConfigKimi::parse(text, "Kimi-K3", None).unwrap_or_else(|e| panic!("{}", e.msg));
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
        // (the release's delta layers: convolutions and the gated norm fp32)
        let w32: Vec<(bool, bool)> = c.kinds.iter().map(|&k| if k == crate::glm::GlmKind::Kda { (true, true) } else { (false, false) }).collect();
        let half = c.layers / 2;
        for early in [false, true] {
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let own = super::super::kimi::describe(&c, &st, &w32, early).unwrap();
                let lowered = kimi(&kinds, c.eos.clone(), c.max_pos, &st, &w32, early).unwrap_or_else(|e| panic!("{}", e.msg));
                // (every tensor it names is the release's, at the shape the release has it)
                assert_eq!(super::super::unnamed(&lowered), None);
                let names = kinds.names();
                assert!(lowered.mats.iter().flat_map(|it| &it.members).all(|m| m.is_empty() || names.contains(m)));
                assert!(lowered.small.iter().flat_map(|s| s.split('|')).all(|s| names.contains(s)));
                let (a, b) = (format!("{own:#?}"), format!("{:#?}", super::super::kimi::nameless(&lowered)));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("the lowered description differs at line {first:?}");
                }
            }
        }
    }

    /// The Kimi family's second random shape (the newer config's sparse attention) lowers as the loader's own description of it: its two latent attention layers
    /// with the indexer of a key a token (RMS-normed keys at the model's epsilon, no pooling tensors, the best 16 tokens kept), a ring of its keys beside each
    /// layer's latents, the shared key part kept.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_kimi_with_sparse_attention_lowers_as_its_own_description() {
        let dir = std::env::temp_dir().join(format!("glyd-lower-kimi-dsa-{}", std::process::id()));
        crate::kimi_ref::random_dsa(&dir).unwrap();
        let c = crate::kimi::ConfigKimi::load(&dir).unwrap();
        assert_eq!(c.dsa, Some(crate::kimi::KimiDsa { layers: vec![3, 7], heads: 4, dim: 64, top_k: 16, layer_norm: false }));
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let w32 = super::super::kimi::dtypes(&ck, &c).unwrap();
        for st in [super::super::Stage::whole(c.layers), super::super::Stage { layers: (4, 8), first: false, last: true }] {
            let d = super::super::kimi::lowered(&dir, &c, &st, &w32, false).unwrap_or_else(|e| panic!("{}", e.msg));
            let ops: Vec<&super::super::MlaOp> = d.layers.iter().flatten().filter_map(|o| if let Op::Mla(a) = o { Some(a) } else { None }).collect();
            assert_eq!((ops.len(), d.rings.len()), if st.first { (2, 4) } else { (1, 2) });
            for a in ops {
                let x = a.indexer.as_ref().expect("the indexer");
                assert_eq!((x.key, x.pooled(), x.pick.ratio, x.pick.top_k, x.pick.heads, x.dim), (super::super::KeyNorm::Rms(1e-5), false, 1, 16, 4, 64));
                assert_eq!(
                    (a.small_count(), a.rope, d.rings[a.ring].d, d.rings[a.ring + 1].d, x.pick.keys),
                    (7, 64, 128 + 64, 128, super::super::IdxKeys::Ring(a.ring + 1))
                );
            }
        }
        // (the first shape has no indexer)
        crate::kimi_ref::random(&dir, 1, true, 8).unwrap();
        let c = crate::kimi::ConfigKimi::load(&dir).unwrap();
        let d = super::super::kimi::describe(&c, &super::super::Stage::whole(c.layers), &vec![(true, true); c.layers], false).unwrap();
        assert!(c.dsa.is_none() && d.layers.iter().flatten().all(|o| !matches!(o, Op::Mla(a) if a.indexer.is_some())));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The random Kimi-K3 the logits and GPU checks run, lowered whole and as a split's stages, is the executor's own description of it; its dtypes as written.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_kimi_lowers_as_its_own_description() {
        let dir = std::env::temp_dir().join(format!("glyd-lower-kimi-{}", std::process::id()));
        crate::kimi_ref::random(&dir, 1, true, 8).unwrap();
        let c = crate::kimi::ConfigKimi::load(&dir).unwrap();
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let w32 = super::super::kimi::dtypes(&ck, &c).unwrap();
        assert!(w32.iter().zip(&c.kinds).all(|(&w, &k)| (k == crate::glm::GlmKind::Kda) == (w == (true, true))), "{w32:?}");
        // (a type other than F32 or BF16 for a delta layer's convolutions or gated norm is refused: the kernels read them by width)
        let kl = c.kinds.iter().position(|&k| k == crate::glm::GlmKind::Kda).unwrap();
        let hdr = dir.join("model.safetensors");
        assert!(hdr.exists(), "the random model's checkpoint is one model.safetensors");
        {
            let bytes = std::fs::read(&hdr).unwrap();
            let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let head = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
            let name = format!("{}layers.{kl}.self_attn.o_norm.weight", c.prefix);
            let at = head.find(&format!("\"{name}\"")).unwrap();
            let tail = &head[at..];
            let f32_at = at + tail.find("\"F32\"").unwrap();
            // ("F32" becomes "I32": the same length and width, so the file still parses and only the loader's own check can refuse it)
            let mut edited = bytes.clone();
            edited[8 + f32_at + 1..8 + f32_at + 4].copy_from_slice(b"I32");
            let bad = std::env::temp_dir().join(format!("glyd-lower-kimi-i8-{}", std::process::id()));
            std::fs::create_dir_all(&bad).unwrap();
            for f in std::fs::read_dir(&dir).unwrap() {
                let f = f.unwrap().path();
                if f.file_name().is_some_and(|x| x != "model.safetensors") && f.is_file() {
                    std::fs::copy(&f, bad.join(f.file_name().unwrap())).unwrap();
                }
            }
            std::fs::write(bad.join("model.safetensors"), &edited).unwrap();
            let r =
                glyd_gpu::safetensors::Checkpoint::open(&bad).map_err(|e| e.to_string()).and_then(|ck| super::super::kimi::dtypes(&ck, &c).map_err(|e| e.msg));
            let _ = std::fs::remove_dir_all(&bad);
            assert!(r.as_ref().is_err_and(|e| e.contains("where F32 or BF16 is read")), "a gated norm of I32 read as {r:?}");
        }
        // (the fused products' widths as the loader stacks them from the checkpoint's own tensors: a delta layer's six, an experts layer's three, padded)
        let rows = |names: &[String]| -> usize { super::fused(names.iter().map(|n| ck.get(&format!("{n}.weight")).unwrap().1.shape[0] as usize).sum()) };
        let (kl, ml) = (c.kinds.iter().position(|&k| k == crate::glm::GlmKind::Kda).unwrap(), c.moe.iter().position(|&m| m).unwrap());
        let a = format!("{}layers.{kl}.self_attn", c.prefix);
        let f = format!("{}layers.{ml}.block_sparse_moe", c.prefix);
        let kda = rows(&["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"].map(|x| format!("{a}.{x}")));
        let moe = rows(&["routed_expert_down_proj", "shared_experts.gate_proj", "shared_experts.up_proj"].map(|x| format!("{f}.{x}")));
        let own = super::super::kimi::describe(&c, &super::super::Stage::whole(c.layers), &w32, false).unwrap();
        assert_eq!(own.widths[Slot::Mix as usize], kda.max(moe).max(3 * c.kda.heads * c.kda.dim).max(c.hidden), "the fused products' widths");
        let half = c.layers / 2;
        for st in [
            super::super::Stage::whole(c.layers),
            super::super::Stage { layers: (0, half), first: true, last: false },
            super::super::Stage { layers: (half, c.layers), first: false, last: true },
        ] {
            let (own, lowered) = (
                super::super::kimi::describe(&c, &st, &w32, false).unwrap(),
                super::super::kimi::lowered(&dir, &c, &st, &w32, false).unwrap_or_else(|e| panic!("{}", e.msg)),
            );
            assert_eq!(format!("{own:#?}"), format!("{:#?}", super::super::kimi::nameless(&lowered)));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tensors the Kimi family's lowering names are the ones its own loader read before the executor's loader took them over, in that loader's order and at
    /// the checkpoint's shapes (the random models of both shapes, whole and as a split's stages): each layer's six depth-mix and norm tensors, then a delta
    /// layer's convolutions (the three as one), A_log,
    /// dt_bias and gated norm, or a latent layer's two norms, kv_b and its indexer's five; the last stage's output mix and final norm. Matrices: a delta
    /// layer's six as one (zero rows to 512s), o, the forget gate's second; a latent layer's q_a, the gate and kv_a as one, q_b, o; a dense MLP's gate and up as
    /// one, down; the embedding and the head. Each operation's index names the tensor it reads.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_kimis_tensors_are_named_as_the_loader_reads_them() {
        use super::super::{DepthMixOp, Stage};
        let dir = std::env::temp_dir().join(format!("glyd-lower-kimi-names-{}", std::process::id()));
        for dsa in [false, true] {
            if dsa {
                crate::kimi_ref::random_dsa(&dir).unwrap();
            } else {
                crate::kimi_ref::random(&dir, 1, true, 8).unwrap();
            }
            let c = crate::kimi::ConfigKimi::load(&dir).unwrap();
            let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
            let w32 = super::super::kimi::dtypes(&ck, &c).unwrap();
            let p = c.prefix;
            for st in [Stage::whole(c.layers), Stage { layers: (0, 4), first: true, last: false }, Stage { layers: (4, 8), first: false, last: true }] {
                let d = super::super::kimi::lowered(&dir, &c, &st, &w32, false).unwrap_or_else(|e| panic!("{}", e.msg));
                // (the loader's reads, in its order)
                let (mut mats, mut small): (Vec<Vec<String>>, Vec<String>) = (Vec::new(), Vec::new());
                for l in st.layers.0..st.layers.1 {
                    let (q, a) = (format!("{p}layers.{l}"), format!("{p}layers.{l}.self_attn"));
                    small.extend(
                        ["input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "self_attention_res_proj", "mlp_res_norm", "mlp_res_proj"]
                            .map(|n| format!("{q}.{n}.weight")),
                    );
                    if c.kinds[l] == crate::glm::GlmKind::Kda {
                        mats.push(["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"].map(|x| format!("{a}.{x}.weight")).to_vec());
                        mats.extend([vec![format!("{a}.o_proj.weight")], vec![format!("{a}.f_b_proj.weight")]]);
                        small.push(["q", "k", "v"].map(|x| format!("{a}.{x}_conv1d.weight")).join("|"));
                        small.extend(["A_log", "dt_bias", "o_norm.weight"].map(|x| format!("{a}.{x}")));
                    } else {
                        mats.push(["q_a_proj", "g_proj", "kv_a_proj_with_mqa"].map(|x| format!("{a}.{x}.weight")).to_vec());
                        mats.extend([vec![format!("{a}.q_b_proj.weight")], vec![format!("{a}.o_proj.weight")]]);
                        small.extend(["q_a_layernorm", "kv_a_layernorm", "kv_b_proj"].map(|x| format!("{a}.{x}.weight")));
                        if c.dsa.as_ref().is_some_and(|x| x.layers.contains(&l)) {
                            small.extend(["wq_b", "wk", "k_norm", "weights_proj"].map(|x| format!("{a}.indexer.{x}.weight")));
                        }
                    }
                    if !c.moe[l] {
                        mats.extend([vec![format!("{q}.mlp.gate_proj.weight"), format!("{q}.mlp.up_proj.weight")], vec![format!("{q}.mlp.down_proj.weight")]]);
                    }
                }
                if st.last {
                    small.extend(["output_attn_res_norm", "output_attn_res_proj", "norm"].map(|x| format!("{p}{x}.weight")));
                }
                mats.extend([vec![format!("{p}embed_tokens.weight")], vec!["language_model.lm_head.weight".to_string()]]);
                let members: Vec<Vec<String>> = d.mats.iter().map(|it| it.members.iter().filter(|m| !m.is_empty()).cloned().collect()).collect();
                assert_eq!((members, &d.small), (mats, &small), "dsa {dsa}, {st:?}");
                // (each matrix at the checkpoint's shape: its members' columns, their rows, zero rows to 512s after several)
                for it in &d.mats {
                    let shapes: Vec<Vec<u64>> = it.members.iter().filter(|m| !m.is_empty()).map(|m| ck.get(m).unwrap().1.shape.clone()).collect();
                    let rows: usize = shapes.iter().map(|s| s[0] as usize).sum();
                    assert!(shapes.iter().all(|s| s.len() == 2 && s[1] as usize == it.cols), "{:?}", it.members);
                    let padded = it.members.last().is_some_and(String::is_empty);
                    assert_eq!(it.rows, if padded { super::fused(rows) } else { rows }, "{:?}", it.members);
                    assert_eq!(padded, shapes.len() > 1, "{:?}", it.members);
                }
                assert_eq!(d.forms.iter().map(|f| &d.small[f.0]).collect::<Vec<_>>(), small.iter().filter(|s| s.contains('|')).collect::<Vec<_>>());
                // (each operation's indices name what it reads)
                let ends = |pairs: &[(usize, &str)]| {
                    for &(i, s) in pairs {
                        assert!(d.small[i].ends_with(s), "{} where {s} is read", d.small[i]);
                    }
                };
                for op in d.layers.iter().flatten() {
                    match op {
                        Op::DepthMix(DepthMixOp { w_n, w_p, w, .. }) => ends(&[(*w_n, "_res_norm.weight"), (*w_p, "_res_proj.weight"), (*w, "norm.weight")]),
                        Op::KdaConv { w, .. } => ends(&[(*w, "v_conv1d.weight")]),
                        Op::KdaDelta { a_log, dt_bias, .. } => ends(&[(*a_log, "A_log"), (*dt_bias, "dt_bias")]),
                        Op::KdaOut { w, .. } => ends(&[(*w, "o_norm.weight")]),
                        Op::Mla(x) => {
                            ends(&[(x.q_norm, "q_a_layernorm.weight"), (x.kv_norm, "kv_a_layernorm.weight"), (x.kv_b, "kv_b_proj.weight")]);
                            if let Some(ix) = &x.indexer {
                                ends(&[(ix.q_b, "wq_b.weight"), (ix.k, "wk.weight"), (ix.k_norm, "k_norm.weight"), (ix.weights, "weights_proj.weight")]);
                            }
                        }
                        _ => {}
                    }
                }
                assert!(d.layers.iter().flatten().flat_map(|op| op.mats()).all(|k| !d.mats[k].members[0].is_empty()));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The streams' operations round as the kind says (`StreamsP::rounding`): DeepSeek-V4-Flash's release (`Once`) on every stream operation, and the same
    /// description with transformers' bf16 rounding (`Twice`) on every one.
    #[test]
    fn the_streams_round_as_their_kind() {
        use super::super::deepseek_v4::V4Opts;
        use crate::runtime::kinds::{Residual, Rounding};
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/DeepSeek-V4-Flash-0731.json");
        let h: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let mut kinds = crate::runtime::describe::describe(&serde_json::from_str(h["config"].as_str().unwrap()).unwrap()).unwrap();
        let (vo, st) = (V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 4096 }, super::super::Stage::whole(43));
        let rounds = |m: &ModelDesc| -> Vec<Rounding> {
            m.layers
                .iter()
                .flatten()
                .filter_map(|op| match *op {
                    Op::StreamsPre { round, .. } | Op::StreamsPost { round, .. } | Op::StreamsHead { round, .. } => Some(round),
                    _ => None,
                })
                .collect()
        };
        let once = rounds(&deepseek_v4(&kinds, vec![1], 1 << 20, &st, &vo).unwrap());
        assert!(once.len() == 4 * 43 + 1 && once.iter().all(|&r| r == Rounding::Once));
        for b in &mut kinds.desc.blocks {
            if let Residual::Streams(s) = &mut b.residual {
                s.rounding = Rounding::Twice;
            }
        }
        let twice = rounds(&deepseek_v4(&kinds, vec![1], 1 << 20, &st, &vo).unwrap());
        assert!(twice.len() == 4 * 43 + 1 && twice.iter().all(|&r| r == Rounding::Twice));
    }

    /// A plain dense decoder's config (the model type the description builder reads it by, the attention's settings) of three layers of hidden 256, 4 query heads over 2
    /// of 64, an MLP of 512, a vocabulary of 1,024: test shapes, no release's.
    fn dense_config(model_type: &str, extra: serde_json::Value) -> serde_json::Value {
        let mut c = serde_json::json!({"architectures": ["X"], "model_type": model_type, "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 1024, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 4096, "hidden_act": "silu", "tie_word_embeddings": false});
        c.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        c
    }

    /// Each form of the plain dense decoder the description builder reads (the random dense model of the logits checks; no q and k norms, a window on every
    /// layer; q and k norms with a tied head; neither) lowers to these operations, written out here by hand: the input norm (then its add of the residual),
    /// the q, k and v product, the attention kind, o, the norm with the residual, gate and up with SiLU, down, the final norm on the last rows; its matrices
    /// the checkpoint's tensors stacked in that order and sized by the description's shapes, which are the checkpoint's; its widths, heads, rotary and norms
    /// the config's as the planner's sizes read it (`weights::Config`). A scaled rope and biases in the attention are refused, named.
    #[test]
    fn dense_lowered_from_each_form_of_its_config() {
        use super::super::attn::{QkNorm, RopeAt};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("tests/fixtures/dense_tiny_manifest.json")).unwrap()).unwrap();
        let tiny_text = manifest["config"].as_str().unwrap().to_string();
        let tiny_shapes: std::collections::HashMap<String, Vec<usize>> = manifest["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_str().unwrap().to_string(), t[2].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()))
            .collect();
        let cases = [
            ("the random dense model of the logits checks", serde_json::from_str(&tiny_text).unwrap()),
            ("no q and k norms, a window on every layer", dense_config("mistral", serde_json::json!({"sliding_window": 128}))),
            ("q and k norms, a tied head", dense_config("qwen3", serde_json::json!({"tie_word_embeddings": true, "rms_norm_eps": 1e-6}))),
            ("no q and k norms, no window", dense_config("llama", serde_json::json!({}))),
        ];
        for (what, cfg) in cases {
            let kinds = crate::runtime::describe::describe(&cfg).unwrap_or_else(|e| panic!("{what}: {}", e.msg));
            assert_eq!(kinds.shapes.keys().cloned().collect::<std::collections::BTreeSet<_>>(), kinds.names(), "{what}: a shape for every tensor");
            let d = dense(&kinds, vec![2], 4096).unwrap_or_else(|e| panic!("{what}: {}", e.msg));
            let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
            let (hid, heads, kv, hd, inter, layers) =
                (n("hidden_size"), n("num_attention_heads"), n("num_key_value_heads"), n("head_dim"), n("intermediate_size"), n("num_hidden_layers"));
            let (qk, tied, window) = (cfg["model_type"] == "qwen3", cfg["tie_word_embeddings"] == true, cfg["sliding_window"].as_u64().unwrap_or(0) as usize);
            let eps = cfg["rms_norm_eps"].as_f64().unwrap() as f32;
            assert_eq!((d.family, d.hidden, d.vocab, d.eps, d.layers.len()), (super::super::dense::FAMILY, hid, n("vocab_size"), eps, layers), "{what}");
            let theta = cfg["rope_theta"].as_f64().unwrap();
            assert_eq!(format!("{:?}", d.ropes), format!("{:?}", [Rope { theta, dim: hd, pairs: hd / 2, trig64: false }]), "{what}");
            // (the small tensors: each layer's two norms, its q and k norms where it has them; then the final norm)
            let per = if qk { 4 } else { 2 };
            assert_eq!(d.small.len(), per * layers + 1, "{what}");
            assert_eq!(d.small.last().unwrap(), "model.norm.weight");
            assert!(matches!(d.first, Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, weight: NormW::Plain, .. }), "{what}: {:?}", d.first);
            for (l, ops) in d.layers.iter().enumerate() {
                let (p, s0, w) = (format!("model.layers.{l}."), per * l, 4 * l);
                assert_eq!(d.small[s0], format!("{p}input_layernorm.weight"));
                assert_eq!(d.small[s0 + 1], format!("{p}post_attention_layernorm.weight"));
                let ops: Vec<&Op> = ops.iter().collect();
                let ops = if l > 0 {
                    assert!(matches!(ops[0], Op::NormAdd { y: Slot::D, w, plus1: false } if *w == s0), "{what}: layer {l}: {:?}", ops[0]);
                    &ops[1..]
                } else {
                    &ops[..]
                };
                assert_eq!(ops.len(), 6, "{what}: layer {l}");
                assert!(matches!(ops[0], Op::Linear { w: x, x: Slot::H, y: Slot::Proj, .. } if *x == w));
                assert!(matches!(ops[2], Op::Linear { w: x, x: Slot::Att, y: Slot::O, .. } if *x == w + 1));
                assert!(matches!(ops[3], Op::NormAdd { y: Slot::O, w: x, plus1: false } if *x == s0 + 1));
                assert!(matches!(ops[4], Op::GateUpSilu { w: x, inter: i } if *x == w + 2 && *i == inter));
                assert!(matches!(ops[5], Op::Linear { w: x, x: Slot::M, y: Slot::D, .. } if *x == w + 3));
                let Op::Attn(a) = ops[1] else { panic!("{what}: layer {l}: {:?}", ops[1]) };
                let norm = if qk { QkNorm::Weight { q: s0 + 2, k: s0 + 3 } } else { QkNorm::None };
                assert_eq!(
                    (a.d, a.nq, a.nk, a.stride, a.q_step, a.k_off, a.v_off, a.norm, a.ring, a.window, a.writes),
                    (hd, heads, kv, (heads + 2 * kv) * hd, hd, heads * hd, Some((heads + kv) * hd), norm, l, window, true),
                    "{what}: layer {l}"
                );
                assert_eq!(
                    (a.rope, a.scale, a.q_scale, a.softcap, a.gate, a.round, a.sbf16),
                    (Some(RopeAt { table: 0, span: hd, pairs: hd / 2 }), (hd as f32).powf(-0.5), 1.0, 0.0, None, false, false)
                );
                assert_eq!((d.rings[l].d, d.rings[l].nk, d.rings[l].window), (hd, kv, window));
                let at = format!("{p}self_attn.");
                assert_eq!(d.mats[w].members, [format!("{at}q_proj.weight"), format!("{at}k_proj.weight"), format!("{at}v_proj.weight")]);
                assert_eq!((d.mats[w].rows, d.mats[w].cols), ((heads + 2 * kv) * hd, hid));
                assert_eq!((d.mats[w + 1].rows, d.mats[w + 1].cols), (hid, heads * hd));
                assert_eq!(d.mats[w + 2].members, [format!("{p}mlp.gate_proj.weight"), format!("{p}mlp.up_proj.weight")]);
                assert_eq!((d.mats[w + 2].rows, d.mats[w + 2].cols, d.mats[w + 3].rows, d.mats[w + 3].cols), (2 * inter, hid, hid, inter));
            }
            assert_eq!((d.embed, d.lm), (4 * layers, if tied { 4 * layers } else { 4 * layers + 1 }), "{what}");
            assert_eq!(d.mats.len(), d.lm + 1);
            assert_eq!(d.final_norm, Some((d.small.len() - 1, Slot::D, false)));
            assert!(d.decoded && !d.skinny && d.states.is_empty());
            for (s, w) in [(Slot::Proj, (heads + 2 * kv) * hd), (Slot::Att, heads * hd), (Slot::Gu, 2 * inter), (Slot::M, inter), (Slot::Hlast, hid)] {
                assert_eq!(d.width(s), w, "{what}: {s:?}");
            }
            // (and the config as the planner's sizes read it: every form of the kind, by its model type)
            let c = crate::weights::Config::parse(&cfg.to_string(), what, None).unwrap();
            assert_eq!(
                (c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter, c.layers, c.eps, c.rope_theta, c.tie),
                (hid, heads, kv, hd, inter, layers, eps, theta, tied)
            );
        }
        // (the random dense model's shapes are its checkpoint's, every one)
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(&tiny_text).unwrap()).unwrap();
        for (name, s) in &kinds.shapes {
            assert_eq!(Some(s), tiny_shapes.get(name), "{name}");
        }
        assert_eq!(kinds.shapes.len(), tiny_shapes.len());
        // (a kind the executor does not take: a scaled rope, biases in the attention)
        let scaled = crate::runtime::reference::testing::Fixture::load("llama_model").j["config"].clone();
        let e = dense(&crate::runtime::describe::describe(&scaled).unwrap(), vec![2], 4096).unwrap_err().msg;
        assert!(e.contains("rope") && e.contains("Llama3"), "{e}");
        let e = dense(&crate::runtime::describe::describe(&dense_config("qwen2", serde_json::json!({}))).unwrap(), vec![2], 4096).unwrap_err().msg;
        assert!(e.contains("a bias"), "{e}");
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

    /// The executor's Gemma 4 head rounds the logits to bf16 between operations (the product's output, g4_softcap's three): a description whose head
    /// says otherwise is refused, named.
    #[cfg(feature = "dev")]
    #[test]
    fn a_gemma4_head_that_rounds_otherwise_is_refused() {
        let mut d = crate::runtime::describe::describe(&serde_json::from_str(crate::logits::GEMMA4_RANDOM_CONFIG).unwrap()).unwrap();
        let shapes = d.shapes.clone();
        let shape = |n: &str| shapes.get(n).cloned();
        assert!(gemma4(&d, &shape, vec![1], 4096).is_ok());
        d.desc.head.rounding = crate::runtime::kinds::Rounding::F32;
        let e = gemma4(&d, &shape, vec![1], 4096).unwrap_err().msg;
        assert!(e.contains("a head of") && e.contains("rounds the logits to bf16"), "{e}");
    }

    /// KV sharing lowered (the gemma4-kv case's random model): a shared layer's attention reads its store layer's ring and writes none, its product the
    /// queries alone; only the eight layers' first four make rings; the shared layers' MLPs are twice as wide. The model is its lowering (`gemma4::lowered`),
    /// and the CPU reference runs it whole (every row of its logits finite).
    #[cfg(feature = "dev")]
    #[test]
    fn a_kv_sharing_gemma4_reads_its_store_layers_rings() {
        let dir = std::env::temp_dir().join(format!("glyd-lower-g4kv-{}", std::process::id()));
        crate::logits::gemma4_kv_random(&dir).unwrap();
        let c = crate::gemma4::ConfigG4::load(&dir).unwrap();
        let d = super::super::gemma4::lowered(&dir, &c).unwrap_or_else(|e| panic!("{}", e.msg));
        let kinds = crate::runtime::describe::describe_dir(&dir).unwrap().0;
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let mut m = crate::runtime::reference::model::RefModel::load(&kinds, &|n: &str| crate::runtime::reference::model::checkpoint_tensor(&ck, n)).unwrap();
        let logits = m.forward(&[2, 48, 85, 122, 159], 0).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(logits.iter().flatten().all(|v| v.is_finite()) && logits.len() == 5);
        assert_eq!(d.rings.len(), 4);
        let attn: Vec<(bool, usize, usize)> =
            d.layers.iter().map(|ops| ops.iter().find_map(|o| if let Op::Attn(a) = o { Some((a.writes, a.ring, a.stride)) } else { None }).unwrap()).collect();
        // (layers 4-6 sliding read layer 2's ring, layer 7 global layer 3's; a shared layer's product is its 4 query heads)
        assert_eq!(attn[4..], [(false, 2, 4 * 256), (false, 2, 4 * 256), (false, 2, 4 * 256), (false, 3, 4 * 512)]);
        assert!(attn[..4].iter().all(|a| a.0));
        assert_eq!((d.mats[2].rows, d.mats[4 * 4 + 2].rows), (2 * 512, 4 * 512));
    }

    /// Per-layer inputs (E2B, E4B) on the CPU reference: a random Gemma 4 with them describes as three blocks a layer (the branch's after the MLP, the layer
    /// scalar moved onto it), its inputs are the formula by hand for a token, the whole model runs, and the executor's lowering refuses it by name.
    #[cfg(feature = "dev")]
    #[test]
    fn gemma4_per_layer_inputs_on_the_reference() {
        use crate::runtime::kinds::{Body, Residual};
        use crate::runtime::reference::{matvec, model::RefModel, norm::norm};
        let text = r#"{"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "eos_token_id": [1, 106],
            "text_config": {"model_type": "gemma4_text", "hidden_size": 128, "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 256, "global_head_dim": 512, "intermediate_size": 128, "vocab_size": 512, "rms_norm_eps": 1e-6, "sliding_window": 16,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"], "hidden_size_per_layer_input": 32, "vocab_size_per_layer_input": 512,
            "final_logit_softcapping": 30.0, "max_position_embeddings": 4096, "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}, "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"}}}}"#;
        let dir = std::env::temp_dir().join(format!("glyd-lower-g4ple-{}", std::process::id()));
        crate::logits::gemma4_shaped_random(&dir, text, 7).unwrap();
        let d = crate::runtime::describe::describe_dir(&dir).unwrap().0;
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let src = |n: &str| crate::runtime::reference::model::checkpoint_tensor(&ck, n);
        let mut m = RefModel::load(&d, &src).unwrap();
        assert_eq!(d.desc.blocks.len(), 9);
        for l in 0..3 {
            let (f, g) = (&d.desc.blocks[3 * l + 1], &d.desc.blocks[3 * l + 2]);
            assert!(matches!(f.residual, Residual::Add) && matches!(g.residual, Residual::AddThenScale));
            assert!(matches!(g.body, Body::PerLayer(p) if p.layer == l));
        }
        // (token 9's inputs by hand: its row of the table times sqrt(32), plus the normed context, times 2^-0.5)
        let (e, pl) = (src(&d.embed).unwrap().1, d.desc.embed.per_layer.unwrap());
        let row: Vec<f32> = e[9 * 128..][..128].iter().map(|&x| x * d.desc.embed.scale).collect();
        let get = |r: &str| {
            src(match &d.extra.iter().find(|(n, _)| n == r).unwrap().1 {
                crate::runtime::describe::Source::Tensor(n) => n,
                s => panic!("{s:?}"),
            })
            .unwrap()
            .1
        };
        let ctx: Vec<f32> = matvec(&get("ple_proj"), 128, &row).iter().map(|v| v / 128f32.sqrt()).collect();
        let (t, w) = (get("ple_embed"), get("ple_norm"));
        let mut want = Vec::new();
        for l in 0..3 {
            let c = norm(&pl.norm, &ctx[l * 32..][..32], Some(&w), None);
            want.extend((0..32).map(|i| (c[i] + t[9 * 96 + l * 32 + i] * 32f32.sqrt()) * std::f32::consts::FRAC_1_SQRT_2));
        }
        let got = m.per_layer_input(9).unwrap();
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1e-5 * b.abs().max(1.0)), "{:?} {:?}", &got[..4], &want[..4]);
        let logits = m.forward(&[2, 9, 40, 77], 0).unwrap();
        assert!(logits.len() == 4 && logits.iter().flatten().all(|v| v.is_finite()));
        let e = gemma4(&d, &|n: &str| d.shapes.get(n).cloned(), vec![1], 4096).unwrap_err().msg;
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("per-layer inputs"), "{e}");
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

    /// Each family's executor description: the random models' of the logits checks, the fixtures' configs (the Mamba-2 layout's, with and without experts: the
    /// kind's only description in the tree), a plain dense config, Gemma 4's draft.
    #[cfg(feature = "dev")]
    fn every_family(tag: &str) -> Vec<ModelDesc> {
        use super::super::{deepseek_v4, gemma4, glm, kimi, muse, qwen35, qwen4x, Stage};
        // (a directory of each caller's own: the tests run at once)
        let dir = |x: &str| std::env::temp_dir().join(format!("glyd-by-op-{tag}-{x}-{}", std::process::id()));
        let mut v = Vec::new();
        let d = dir("q35");
        crate::logits::qwen35_random(&d).unwrap();
        let c = crate::qwen35::Config35::load(&d).unwrap();
        v.push(qwen35::describe(&c, &Stage::whole(c.layers)).unwrap());
        let d = dir("g4");
        crate::logits::gemma4_random(&d).unwrap();
        let c = crate::gemma4::ConfigG4::load(&d).unwrap();
        v.extend([gemma4::describe(&c), gemma4::describe_draft(&c, &c)]);
        let d = dir("muse");
        crate::muse_ref::random(&d).unwrap();
        let c = crate::muse::ConfigMuse::load(&d).unwrap();
        v.push(muse::lowered(&d, &crate::runtime::describe::describe_dir(&d).unwrap().0, &Stage::whole(c.layers)).unwrap());
        let d = dir("kimi");
        crate::kimi_ref::random(&d, 1, true, 8).unwrap();
        let c = crate::kimi::ConfigKimi::load(&d).unwrap();
        let w32 = kimi::dtypes(&glyd_gpu::safetensors::Checkpoint::open(&d).unwrap(), &c).unwrap();
        v.push(kimi::lowered(&d, &c, &Stage::whole(c.layers), &w32, false).unwrap());
        let d = dir("glm");
        crate::glm::random(&d, false, None).unwrap();
        let c = crate::glm::ConfigGlm::load(&d).unwrap();
        v.push(glm::lowered(&d, &c, &Stage::whole(c.layers)).unwrap());
        let d = dir("v4");
        crate::dsv4_ref::random(&d, None).unwrap();
        let (layers, eos, max_pos) = deepseek_v4::ends(&d).unwrap();
        let vo = deepseek_v4::V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 512 };
        let (kinds, _) = crate::runtime::describe::describe_dir(&d).unwrap();
        v.push(super::super::lower_comp::lower(&kinds, eos, max_pos, &Stage::whole(layers), &vo).unwrap().0);
        let d = dir("qx");
        crate::logits::qwen4x_random(&d).unwrap();
        v.push(qwen4x::lowered(&d, 1024).unwrap_or_else(|e| panic!("{}", e.msg)));
        for f in ["nemotron_h_config.json", "nemotron_h_moetoy_config.json"] {
            let text = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap();
            let c = crate::nemotron::ConfigNemotronH::parse(&text, f, None).unwrap();
            let kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
            v.extend(nemotron(&kinds, c.eos.clone(), c.max_pos, &Stage::whole(c.layers)));
        }
        v.push(dense(&crate::runtime::describe::describe(&dense_config("llama", serde_json::json!({}))).unwrap(), vec![2], 4096).unwrap());
        for x in ["q35", "g4", "muse", "kimi", "glm", "v4", "qx"] {
            let _ = std::fs::remove_dir_all(dir(x));
        }
        v
    }

    /// The decisions the executor made by a family's name, made by its description's operations now, give every family the answer the name lists gave, but
    /// where named: a second sequence for Qwen3.8-Flash-Next (`add_seqs`: every one of its operations keeps a sequence's data in its rings and states, as
    /// `batchable` said already); a verify pass for Gemma 4 and the plain dense models (attention rings and row-wise operations alone; the range worker,
    /// verify's caller with Qwen3.5), for Qwen3.8-Flash-Next (its draft head's rounds, src/dynamic/head.rs: its sparse attention, its plain experts held on
    /// the GPU, its n-gram embedding's states, which `rewind` goes back in) and for DeepSeek-V4 (its DSpark drafter's rounds: the compressed attention, whose
    /// compressors' open windows `rewind` pools again over the kept rows, its weighed experts held on the GPU).
    #[cfg(feature = "dev")]
    #[test]
    fn decisions_by_operation_give_the_family_lists_answers() {
        use super::super::{dense::FAMILY as DENSE, qwen4x::FAMILY as QX, rows, state_type, unverifiable_op};
        let all = every_family("decide");
        let families: std::collections::BTreeSet<&str> = all.iter().map(|d| d.family).collect();
        assert_eq!(families.len(), 10, "{families:?}");
        for d in &all {
            let f = d.family;
            let listed = matches!(f, "Qwen3.5" | "Gemma 4" | "Muse Spark" | "Kimi-K3" | "GLM-5.3-Flash" | "Nemotron-H" | "DeepSeek-V4" | DENSE);
            assert_eq!(rows::one_sequence_op(d).is_none(), listed || f == QX, "{f}: more sequences than one");
            assert_eq!(
                unverifiable_op(d).is_none(),
                matches!(f, "Qwen3.5" | "Gemma 4" | DENSE | QX | "DeepSeek-V4"),
                "{f}: a verify pass ({:?})",
                unverifiable_op(d).map(rows::op_name)
            );
            if !d.states.is_empty() {
                let gdn = "fp32 (the delta rule's), bf16 (the convolution's)";
                assert_eq!(state_type(d), if f == "Qwen3.5" { gdn } else { "not reported" }, "{f}: the states' types");
            }
        }
        assert_eq!(all.iter().find(|d| d.family == "Gemma 4 draft").and_then(rows::one_sequence_op).map(rows::op_name).as_deref(), Some("DraftIn"));
    }

    /// A cut between layers hands over what each family's stages handed by name (Qwen3.5 X and D, Kimi-K3 X and Blk, the rest X, H, D, O and S where they are),
    /// but where named: Nemotron-H and the plain dense models, whose layers begin by adding the last one's output (O, D) to X, now hand X and that alone
    /// (H, which that add writes, no more). Every layer after the first begins with an operation that reads those slots.
    #[cfg(feature = "dev")]
    #[test]
    fn a_cut_hands_over_what_the_next_layer_reads_first() {
        use super::super::{dense::FAMILY as DENSE, handed_over, Slot::*};
        for d in every_family("cut") {
            let named: &[Slot] = match d.family {
                "Qwen3.5" => &[X, D],
                "Kimi-K3" => &[X, Blk],
                _ => &[X, H, D, O, S],
            };
            let want: Vec<Slot> = match d.family {
                "Nemotron-H" => vec![X, O],
                DENSE => vec![X, D],
                _ => named.iter().copied().filter(|&s| d.widths[s as usize] > 0).collect(),
            };
            let got: Vec<Slot> = handed_over(&d).into_iter().map(|x| x.0).collect();
            assert_eq!(got, want, "{}", d.family);
            for (l, ops) in d.layers.iter().enumerate().skip(1).filter(|(_, o)| !o.is_empty()) {
                match ops[0] {
                    Op::NormAdd { y, .. } => assert_eq!(got, [X, y], "{} layer {l}", d.family),
                    Op::DepthMix(_) => assert_eq!(got, [X, Blk], "{} layer {l}", d.family),
                    ref op => assert_eq!(got.len(), named.iter().filter(|&&s| d.widths[s as usize] > 0).count(), "{} layer {l} begins with {op:?}", d.family),
                }
            }
        }
    }

    /// The executor's loader's rules, on the random Qwen3.5's tensors named in its own loader's order (src/kern35.rs), make that loader's choices: the same
    /// matrices dense (the b and a gates' parts under 64 rows), the same small tensors in fp32 (A_log; the gated norm's weight, its values bf16's), the same rotary
    /// tables; and its description names the tensors that loader read, in its order (what the executor's loader reads).
    #[cfg(feature = "dev")]
    #[test]
    fn the_executors_rules_on_qwen35_make_its_own_loaders_choices() {
        use super::super::{held_as_saved, read_f32, read_f32_as_bf16};
        let dir = std::env::temp_dir().join(format!("glyd-q35-loader-{}", std::process::id()));
        crate::logits::qwen35_random(&dir).unwrap();
        let c = crate::qwen35::Config35::load(&dir).unwrap();
        let d = super::super::qwen35::lowered(&dir, &c, &super::super::Stage::whole(c.layers)).unwrap();
        // (its tensors in the order the description names them)
        let packs: std::collections::HashMap<String, crate::qwen35::Pack35> = c.packs().into_iter().map(|p| (p.key.clone(), p)).collect();
        let item = |src: &crate::kern35::Src| match src {
            crate::kern35::Src::Pack(k) => Item { members: packs[k].parts.iter().map(|x| x.0.clone()).collect(), rows: packs[k].rows(), cols: packs[k].cols },
            crate::kern35::Src::Dense(p, r, cl) => Item { members: p.clone(), rows: *r, cols: *cl },
        };
        let mut mats: Vec<Item> = crate::kern35::layer_mats(&c).iter().flatten().map(item).collect();
        mats.push(item(&crate::kern35::Src::Pack("embed".into())));
        if !c.tie {
            mats.push(item(&crate::kern35::Src::Pack("lm_head".into())));
        }
        let mut small = Vec::new();
        for (l, k) in c.kinds.iter().enumerate() {
            let p = format!("{}layers.{l}", c.prefix);
            small.extend([format!("{p}.input_layernorm.weight"), format!("{p}.post_attention_layernorm.weight")]);
            small.extend(match k {
                crate::qwen35::Kind::Full => ["q_norm", "k_norm"].map(|x| format!("{p}.self_attn.{x}.weight")).to_vec(),
                crate::qwen35::Kind::Linear => ["conv1d.weight", "A_log", "dt_bias", "norm.weight"].map(|x| format!("{p}.linear_attn.{x}")).to_vec(),
            });
        }
        small.push(format!("{}norm.weight", c.prefix));
        assert_eq!(format!("{:?}", d.mats), format!("{mats:?}"));
        assert_eq!(d.small, small);
        // (and each stage of a split names its own, its reading and the family's alike)
        for st in [super::super::Stage { layers: (0, 2), first: true, last: false }, super::super::Stage { layers: (2, c.layers), first: false, last: true }] {
            let (own, low) = (super::super::qwen35::describe(&c, &st).unwrap(), super::super::qwen35::lowered(&dir, &c, &st).unwrap());
            assert_eq!(format!("{own:#?}"), format!("{low:#?}"));
            assert_eq!((super::super::unnamed(&low), low.small.len()), (None, low.final_norm.unwrap().0 + 1), "{st:?}");
            assert!(low.wanted(&st).iter().enumerate().all(|(k, &w)| !w || low.mats[k].rows > 0));
        }
        assert_eq!((super::super::unnamed(&d), d.small.len()), (None, d.final_norm.unwrap().0 + 1));
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let rows_of = |n: &str| ck.get(n).map(|(_, t)| t.shape[0] as usize);
        let dense: std::collections::HashSet<String> = crate::kern35::layer_mats(&c)
            .into_iter()
            .flatten()
            .filter_map(|s| if let crate::kern35::Src::Dense(p, ..) = s { Some(p) } else { None })
            .flatten()
            .collect();
        assert!(!dense.is_empty());
        for (k, it) in d.mats.iter().enumerate() {
            assert_eq!(held_as_saved(&d, k, &rows_of), it.members.iter().all(|m| dense.contains(m)), "{:?}", it.members);
        }
        let kept: std::collections::HashMap<String, &str> = c.kept().into_iter().map(|t| (t.name, t.dtype)).collect();
        let (f32s, as_bf16) = (read_f32(&d), read_f32_as_bf16(&d));
        for (i, n) in d.small.iter().enumerate() {
            assert_eq!(f32s.contains(&i) || as_bf16.contains(&i), kept[n] == "F32", "{n}");
            assert_eq!(as_bf16.contains(&i), n.ends_with("linear_attn.norm.weight"), "{n}");
        }
        for r in &d.ropes {
            assert_eq!(crate::gemma4::rotary(r.theta, r.dim, r.pairs, 64), crate::weights::rope_tables(c.rope_theta, c.rotary_dim, 64));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The Nemotron-H lowering names its tensors as its own loader reads them (src/kern_nh.rs: `layer_packs` and `layer_kept` in that loader's order), on the
    /// random model whole and as a split's stages, and the executor's loader's rules on them make that loader's choices: the same matrices dense (the parts
    /// the packer does not take), every small tensor as saved in bf16 (none read wider).
    #[cfg(feature = "dev")]
    #[test]
    fn the_executors_rules_on_nemotron_make_its_own_loaders_choices() {
        use super::super::{held_as_saved, read_f32, read_f32_as_bf16};
        let dir = std::env::temp_dir().join(format!("glyd-nh-loader-{}", std::process::id()));
        crate::kcheck_nh::random(&dir, 0, 16).unwrap();
        let c = crate::nemotron::ConfigNemotronH::load(&dir).unwrap();
        // (the loader's matrices and small tensors in its order, of the stage's layers; the final norm, the embedding and the head named on every stage)
        let half = c.layers / 2;
        for st in [
            super::super::Stage::whole(c.layers),
            super::super::Stage { layers: (0, half), first: true, last: false },
            super::super::Stage { layers: (half, c.layers), first: false, last: true },
        ] {
            let d = super::super::nemotron::lowered(&dir, &c, &st).unwrap();
            let it = |p: &crate::qwen35::Pack35| (p.parts.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), p.rows(), p.cols);
            let mut mats: Vec<_> =
                (st.layers.0..st.layers.1).filter(|&l| c.kinds[l] != crate::nemotron::NhKind::Moe).flat_map(|l| c.layer_packs(l)).map(|p| it(&p)).collect();
            mats.push((vec![format!("{}embeddings.weight", c.prefix)], c.vocab, c.hidden));
            if !c.tie {
                mats.push((vec!["lm_head.weight".into()], c.vocab, c.hidden));
            }
            let mut small = Vec::new();
            for l in st.layers.0..st.layers.1 {
                let kept = c.layer_kept(l);
                let n = if c.kinds[l] == crate::nemotron::NhKind::Mamba { kept.len() } else { 1 };
                small.extend(kept.into_iter().take(n).map(|t| t.name));
            }
            small.push(format!("{}norm_f.weight", c.prefix));
            let got: Vec<_> = d.mats.iter().map(|x| (x.members.clone(), x.rows, x.cols)).collect();
            assert_eq!((got, &d.small), (mats, &small), "{st:?}");
            assert_eq!((super::super::unnamed(&d), d.small.len()), (None, d.final_norm.unwrap().0 + 1), "{st:?}");
        }
        let d = super::super::nemotron::lowered(&dir, &c, &super::super::Stage::whole(c.layers)).unwrap();
        let packs: Vec<crate::qwen35::Pack35> = (0..c.layers).filter(|&l| c.kinds[l] != crate::nemotron::NhKind::Moe).flat_map(|l| c.layer_packs(l)).collect();
        let ck = glyd_gpu::safetensors::Checkpoint::open(&dir).unwrap();
        let rows_of = |n: &str| ck.get(n).map(|(_, t)| t.shape[0] as usize);
        let dense: std::collections::HashSet<String> = packs.iter().filter(|p| !p.packable()).flat_map(|p| p.parts.iter().map(|x| x.0.clone())).collect();
        for (k, it) in d.mats.iter().enumerate() {
            assert_eq!(held_as_saved(&d, k, &rows_of), it.members.iter().all(|m| dense.contains(m)), "{:?}", it.members);
        }
        let kept: std::collections::HashMap<String, &str> = (0..c.layers).flat_map(|l| c.layer_kept(l)).map(|t| (t.name, t.dtype)).collect();
        assert!(d.small.iter().filter(|n| kept.contains_key(*n)).count() + 1 == d.small.len());
        let (f32s, as_bf16) = (read_f32(&d), read_f32_as_bf16(&d));
        for (i, n) in d.small.iter().enumerate() {
            assert!(!f32s.contains(&i) && !as_bf16.contains(&i), "{n}");
            assert!(kept.get(n).is_none_or(|&t| t == "BF16"), "{n}");
        }
        assert!(d.layers.iter().flatten().any(|op| matches!(op, Op::MambaScan { .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The descriptions the executor's loader reads (`Dyn::new_placed`: Gemma 4, its draft, the plain dense models, Qwen3.5, Muse Spark, Kimi-K3, GLM-5.3-Flash,
    /// Nemotron-H, Qwen3.8-Flash-Next) name every matrix their operations use. DeepSeek-V4's, which its own loader fills by its lowering's plan
    /// (src/dynamic/deepseek_v4.rs), names none, and the executor's loader refuses it in words.
    #[cfg(feature = "dev")]
    #[test]
    fn the_executors_loader_reads_only_descriptions_that_name_their_tensors() {
        use super::super::unnamed;
        let all = every_family("named");
        assert!(all.iter().any(|d| d.family == "DeepSeek-V4") && all.iter().filter(|d| d.family == "Nemotron-H").count() == 2);
        for d in all {
            let named = d.family != "DeepSeek-V4";
            assert_eq!(unnamed(&d).is_none(), named, "{}: {:?} of {} matrices named", d.family, unnamed(&d), d.mats.len());
        }
    }
}
