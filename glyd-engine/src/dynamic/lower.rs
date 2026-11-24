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
        rings.push(Ring { d: dh, nk, window: a.window.unwrap_or(0), values: true });
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
        let attn = AttnOp::of(a, Slot::Proj, ring, Some(r), Some((q_norm, k_norm)))?;
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

/// The pre-norm dense decoder (the description builder's `llama` reading: Llama, Mistral, Qwen2, Qwen3): pairs of blocks, an attention of the kind's parameters
/// (its q and k norms where the kind has them, a window where it slides) and the gated SiLU MLP, each `x + block(rmsnorm(x) w)`; the final norm on the last rows,
/// the head the embedding's matrix where tied. Each matrix of a layer is its checkpoint tensors stacked (q, k and v; gate and up), sized by the shapes the
/// description gives. The products are the library's at every row count, a long chunk's decoded and multiplied dense where its route says so: the products,
/// norms, activation, embedding and head of the dense Qwen3 engine this replaces, kernel for kernel. A kind this lowering does not take (a bias, a rope with
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
                let t = Rope { theta: r.theta, dim: r.dim, pairs: r.freqs };
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
        rings.push(Ring { d: a.head_dim, nk: a.kv_heads, window: a.window.unwrap_or(0), values: true });
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
    })
}

/// Qwen3.8-Flash-Next (`qwen4_exp_text`, Condor's model) on the shared operations, from its layer kinds alone: every block inside the gated residuals (`Op::StreamsPre`
/// with `StreamsMix::Gated`, `Op::StreamsPost` gated) over the four streams the embedding row becomes (`Op::Streams`), the final mixer (`HeadMix::Gated`); Gated DeltaNet
/// with the sigmoid gate (`Op::GatedNorm` `sigmoid`); the sparse attention (`Op::IdxMean`, `Op::IdxPick` of `IdxKeys::Pools` unweighted, `Op::Attn` `picked`); the experts
/// (`Op::Moe`, block k the k-th layer's); the n-gram embedding before its layer (`Op::NgramRows`, its product, `Op::NgramAdd`). The matrices are the checkpoint's by
/// name (the table of n-grams held as saved); the small tensors as saved (A_log fp32, as the delta rule reads it: `dynamic::qwen4x`). `max_pos`: the positions
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
    if m.embed.streams != 4
        || m.embed.scale != 1.0
        || m.embed.norm.is_some()
        || m.head.collapse != Collapse::GatedMix
        || m.head.softcap.is_some()
        || m.head.scale != 1.0
    {
        return err(format!("{what}: an embedding of {:?} and a head of {:?}: the kernels take four gated streams and their mixer", m.embed, m.head));
    }
    let (mut mats, mut small, mut states, mut ropes) = (Vec::new(), Vec::<String>::new(), Vec::<usize>::new(), Vec::<Rope>::new());
    let mut ops: Vec<Vec<Op>> = Vec::new();
    let mut layer: Vec<Op> = Vec::new();
    let (mut mix_w, mut attn_w, mut rank_w, mut ple_w, mut n_moe) = (0, 0, 0, 0, 0);
    let (mut cd_w, mut vd_w, mut nv_w, mut ix_w) = (0, 0, 0, 0);
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
            if g.streams != 4 {
                return err(format!("{what}: gated residuals over {} streams: the kernels take 4", g.streams));
            }
            pn(&b.norm_in, "a gated residual")?;
            let norm = name(small, one(roles, "norm_in")?);
            let (down, up, inject) =
                (mat(mats, vec![one(roles, "mix_down")?])?, mat(mats, vec![one(roles, "mix_up")?])?, mat(mats, vec![one(roles, "inject")?])?);
            Ok((Op::StreamsPre { mix: StreamsMix::Gated { down, up, inject }, norm, round: crate::runtime::kinds::Rounding::Twice }, g.lowrank))
        };
    let post = Op::StreamsPost { round: crate::runtime::kinds::Rounding::Twice, gated: true };
    for (b, roles) in m.blocks.iter().zip(&d.blocks) {
        match &b.body {
            Body::Ngram(n) => {
                let Ngram { order, heads, dim, conv, dilation, streams, eos: end, .. } = *n;
                if (order, conv, dilation, streams) != (3, 4, 3, 4)
                    || n.form != crate::runtime::kinds::NgramForm::Condor
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
                let hash =
                    [name(&mut small, one(roles, "ngram_mult")?), name(&mut small, one(roles, "ngram_sizes")?), name(&mut small, one(roles, "ngram_offsets")?)];
                let kv = mat(&mut mats, vec![one(roles, "key")?, one(roles, "value")?])?;
                let norms = [name(&mut small, one(roles, "norm_in")?), name(&mut small, one(roles, "norm_key")?), name(&mut small, one(roles, "norm_conv")?)];
                let taps = name(&mut small, one(roles, "conv")?);
                let (hist, cv) = (state(&mut states, 8), state(&mut states, 9 * 4 * d_ * 2));
                layer.extend([
                    Op::NgramRows(super::ngram::NgramRows { table, hash, hist, per: heads, row: dim / (2 * heads), eos: end }),
                    Op::Linear { w: kv, x: Slot::In, y: Slot::Gu, name: "n-gram embedding: key and value product" },
                    Op::NgramAdd(super::ngram::NgramAdd { norms, taps, conv: cv }),
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
                        if r.scaling != RopeScaling::None
                            || r.freq_dim != r.dim
                            || ix.head_dim != 128
                            || ix.block != 4
                            || r.dim > 128
                            || ix.top_k % ix.block != 0
                        {
                            return err(format!(
                                "{what}: a rotary of {r:?} and an indexer of {ix:?}: the kernels take the plain rotary, index heads of 128 and pools of 4"
                            ));
                        }
                        let rt = Rope { theta: r.theta, dim: r.dim, pairs: r.freqs };
                        let table = match ropes.iter().position(|x| (x.theta, x.dim, x.pairs) == (rt.theta, rt.dim, rt.pairs)) {
                            Some(i) => i,
                            None => {
                                ropes.push(rt);
                                ropes.len() - 1
                            }
                        };
                        let qk = (name(&mut small, one(roles, "q_norm")?), name(&mut small, one(roles, "k_norm")?));
                        let kvb = max_pos * a.kv_heads * a.head_dim * 2;
                        let kv = [state(&mut states, kvb), state(&mut states, kvb)];
                        let att = AttnOp::picked(a, Slot::Mix, kv, Some(table), Some(qk))?;
                        let stride = att.stride;
                        let qkv = mat(&mut mats, vec![one(roles, "q")?, one(roles, "k")?, one(roles, "v")?])?;
                        if mats[qkv].rows != stride {
                            return err(format!("{what}: a q, k and v product of {} rows, where its attention reads {stride}", mats[qkv].rows));
                        }
                        let index = mat(&mut mats, vec![one(roles, "index_qk")?])?;
                        let o = mat(&mut mats, vec![one(roles, "o")?])?;
                        let (iqn, ikn) = (name(&mut small, one(roles, "index_q_norm")?), name(&mut small, one(roles, "index_k_norm")?));
                        let (keys, pools) = (state(&mut states, max_pos * 128 * 2), state(&mut states, (max_pos / 4 + 1) * 128 * 2));
                        let pick = super::IdxPick {
                            heads: ix.heads,
                            ratio: ix.block,
                            top_k: ix.top_k / ix.block,
                            expand: true,
                            tail: true,
                            wscale: 1.0,
                            score: super::IdxScore::F32 { dim: 128, scale: (128f32).powf(-0.5) },
                            keys: super::IdxKeys::Pools(pools),
                            weighted: false,
                        };
                        layer.extend([
                            Op::Linear { w: qkv, x: Slot::H, y: Slot::Mix, name: "attention: qkv product" },
                            Op::Linear { w: index, x: Slot::H, y: Slot::Qr, name: "indexer: product" },
                            Op::IdxMean { heads: ix.heads, q_norm: iqn, k_norm: ikn, rope: table, keys, pools },
                            Op::IdxPick(pick),
                            Op::Attn(att),
                            Op::Linear { w: o, x: Slot::Att, y: Slot::O, name: "attention: o product" },
                        ]);
                        (mix_w, attn_w, ix_w) = (mix_w.max(stride), attn_w.max(a.heads * a.head_dim), ix_w.max(ix.heads));
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
    let w = 4 * d_;
    let mut widths = [0usize; SLOTS];
    for (s, x) in [
        (Slot::X, d_),
        (Slot::H, d_),
        (Slot::S, w),
        (Slot::Hq, w),
        (Slot::Mid, rank_w),
        (Slot::Cp, w),
        (Slot::Kv, 4),
        (Slot::O, d_),
        (Slot::Mix, mix_w),
        (Slot::Ba, 2 * nv_w),
        (Slot::Xc, cd_w),
        (Slot::Core, vd_w),
        (Slot::Qo, attn_w),
        (Slot::Att, attn_w),
        (Slot::Qr, (ix_w + 1) * 128),
        (Slot::Iq, ix_w * 128),
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
        ropes,
        rings: Vec::new(),
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
    let eps = m.blocks.first().map_or(1e-6, |b| b.norm_in.eps);
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
                rings.push(Ring { d: 256, nk, window: 0, values: true });
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
        eps,
        max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: vec![Rope { theta: rope.theta, dim: rotary, pairs: rotary / 2 }],
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
    })
}

/// Nemotron-H (Nemotron 3 Super, Nemotron 3.5 Lightning): a block a layer, Mamba-2, attention without a positional encoding, or a mixture of experts (its block of
/// experts is moe_nh.rs's); every layer x + mixer(norm(x)), the final norm on the last row. The matrices, small tensors and blocks are kern_nh.rs's loader's
/// (`dynamic::nemotron::describe` indexes them in its order); stage `st` of a model split across GPUs.
pub fn nemotron(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::nemotron::per_layer;
    use crate::kern_nh::{NH_N, NH_P};
    use crate::nemotron::NhKind;
    let m = &d.desc;
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
    if mb.head_dim != NH_P || mb.state != NH_N || mb.conv != 4 || hd != 128 {
        return err(format!(
            "Mamba-2 heads of {} channels with a state of {}, a convolution of {}, attention heads of {hd}: this engine runs heads of {NH_P}, a state of {NH_N}, a convolution of 4 and attention heads of 128",
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
        match kind {
            NhKind::Mamba => {
                let (conv, ssm) = (states.len(), states.len() + 1);
                states.push(3 * cd * 2);
                states.push(mb.heads * NH_P * NH_N * 4);
                ops.extend([
                    Op::Linear { w: b, x: Slot::H, y: Slot::Mix, name: "Mamba-2: in product" },
                    Op::MambaConv { w: s0 + 1, bias: s0 + 2, state: conv, cd, row, x_off: di },
                    Op::MambaScan { dt_bias: s0 + 5, a_log: s0 + 3, d: s0 + 4, state: ssm, cd, row, dt_off: di + cd, nh: mb.heads, ng: mb.groups },
                    Op::MambaNorm { w: s0 + 6, row, di, ng: mb.groups },
                    Op::Linear { w: b + 1, x: Slot::Core, y: Slot::O, name: "Mamba-2: out product" },
                ]);
            }
            NhKind::Attention => {
                let ring = rings.len();
                rings.push(Ring { d: hd, nk, window: 0, values: true });
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
        mats: Vec::new(),
        small: Vec::new(),
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
    })
}

/// Muse Spark: pairs of blocks, the gated attention (q, k, v and the gate in one product, heads of 64, the rotary on the window's layers alone) with its input-scaled
/// norms and residual gate, then the experts in the latent with theirs; the embedding normed without a weight, the soft-capped head times the output multiplier.
/// The matrices, small tensors and blocks are dynamic/muse.rs's loader's (`dynamic::muse::describe` indexes them in its order).
pub fn muse(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("Muse Spark: {} blocks, where each layer is an attention block and an experts block", m.blocks.len()));
    }
    let layers = m.blocks.len() / 2;
    let mut atts = Vec::with_capacity(layers);
    let (mut lat, mut mult) = (None, 1.0);
    for l in 0..layers {
        match (&m.blocks[2 * l].body, &m.blocks[2 * l + 1].body) {
            (Body::Mixer(Mixer::Attention(a)), Body::Ffn(Ffn::Moe(e))) => {
                atts.push(a);
                lat = lat.or(e.latent.as_ref().map(|x| x.dim));
                mult = e.router.logit_scale;
            }
            (a, f) => return err(format!("Muse Spark: layer {l} is {a:?} and {f:?}")),
        }
    }
    let a0 = atts.first().ok_or_else(|| Error("Muse Spark: no layers".into()))?;
    let (nq, nk, hd, lat) = (a0.heads, a0.kv_heads, a0.head_dim, lat.ok_or_else(|| Error("Muse Spark: experts without a latent".into()))?);
    let (eps, post_eps) = (m.blocks[0].norm_in.eps, m.blocks[0].norm_out.map_or(1e-6, |n| n.eps));
    if hd != 64 || m.hidden > 8192 || lat > 4096 {
        return err(format!(
            "heads of {hd}, a width of {}, a latent of {lat}: this engine runs heads of 64, widths up to 8192 and latents up to 4096",
            m.hidden
        ));
    }
    let row = (2 * nq + 2 * nk) * hd;
    let ones = 0;
    let (mut n_small, mut n_mats, mut n_moe, mut rings, mut ops_all) = (1, 0, 0, Vec::new(), Vec::new());
    for (l, a) in atts.iter().enumerate() {
        if !(st.layers.0..st.layers.1).contains(&l) {
            ops_all.push(Vec::new());
            continue;
        }
        let s0 = n_small;
        n_small += 9;
        let m0 = n_mats;
        n_mats += 4;
        let block = n_moe;
        n_moe += 1;
        let ring = rings.len();
        rings.push(Ring { d: hd, nk, window: a.window.unwrap_or(0), values: true });
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
            Op::Moe(super::muse::latent_moe(block, s0 + 5, post_eps, mult)),
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
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
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
    })
}

/// GLM-5.3-Flash: the four streams mixed around each layer's two blocks, Kimi delta attention (its forget gate bounded) or the latent attention (its block's),
/// a dense MLP (its SwiGLU clamped) in the first layers and a mixture of experts in the rest (its block's), the mean of the streams and the final norm. The
/// matrices, small tensors and blocks are model_glm.rs's loader's (`dynamic::glm::describe` indexes them in its order).
pub fn glm(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage) -> Result<ModelDesc> {
    use super::glm::{dense_ops, glm_desc, kda_ops, per_layer};
    use crate::glm::GlmKind;
    use crate::model_glm::GM_D;
    use crate::runtime::kinds::{Decay, Residual};
    let m = &d.desc;
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
        let s = n_small;
        n_small += 5;
        let mut ops = vec![Op::StreamsPre { mix: StreamsMix::Hc { f: s + 2, base: s + 3, scale: s + 4 }, norm: s, round }];
        let m0 = n_mats;
        let nm = per_layer(kind, moe);
        n_mats += nm;
        if kind == GlmKind::Kda {
            let s0 = n_small;
            n_small += 4;
            let (conv, ssm) = (states.len(), states.len() + 1);
            states.push(3 * 3 * qd * 2);
            states.push(heads * GM_D * GM_D * 4);
            ops.extend(kda_ops(m0, s0, conv, ssm, qd, heads, bound));
        } else if let Some(x) = mla {
            // (q's and the key-value latent's products apart; the indexer's picks given as their tokens and the open pool's)
            let op = super::mla::MlaOp::of(x, super::mla::MlaIn::Apart { q_a: m0, kv_a: m0 + 2 }, m0 + 1, m0 + 3, n_small, rings.len(), true)?;
            n_small += op.small_count();
            rings.extend(op.rings());
            ops.push(Op::Mla(op));
        }
        let h = n_small;
        n_small += 3;
        ops.extend([Op::StreamsPost { round, gated: false }, Op::StreamsPre { mix: StreamsMix::Hc { f: h, base: h + 1, scale: h + 2 }, norm: s + 1, round }]);
        if moe {
            ops.push(Op::Moe(MoeOp::of(n_moe)));
            n_moe += 1;
        } else {
            ops.extend(dense_ops(m0 + nm - 2, inter, limit));
        }
        ops.push(Op::StreamsPost { round, gated: false });
        ops_all.push(ops);
    }
    if st.last {
        ops_all.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::StreamsHead { mix: HeadMix::Mean, norm: n_small, round });
    }
    let e = n_mats;
    let lm = if st.last && d.head.is_some() { e + 1 } else { e };
    let eps = m.blocks.first().map_or(1e-5, |b| b.norm_in.eps);
    Ok(glm_desc(m.hidden, qd, heads, inter, m.vocab, eps, max_pos, eos, hc, ops_all, (states, rings), e, lm))
}

/// Kimi-K3: Kimi delta attention (its forget gate bounded) or gated NoPE latent attention (its block's), a dense MLP in the first layers and latent experts with
/// shared ones in the rest (their block's), every block's input a depth mix over the groups of layers before it; SiTU. `w32` the delta layers' dtypes as the
/// checkpoint holds them, `early` whether the cached experts are guessed ahead. The matrices, small tensors and blocks are dynamic/kimi.rs's loader's
/// (`dynamic::kimi::describe` indexes them in its order).
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
    let (mut inter, mut situ, mut moe_rows, mut mla) = (0, (0.0, 0.0), 0, None);
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
                // (q_a, the gate and kv_a one product)
                mla = Some(super::mla::MlaOp::of(x, super::mla::MlaIn::Fused { w: 0 }, 0, 0, 0, 0, false)?);
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
        ls.push(KimiLayer { kda, moe, w32: w32.get(l).copied().unwrap_or_default() });
    }
    let eps = m.blocks.first().map_or(1e-6, |b| b.norm_in.eps);
    Ok(kimi_desc(&KimiDims { h: m.hidden, qd, nh, kd, bound, inter, situ, eps, moe_rows, mla }, &ls, &starts, st, early, m.vocab, max_pos, eos))
}

/// DeepSeek-V4: the compressed attention (the window, and on a compressing layer its compressor's entries; on a ratio-4 layer the overlapping windows and the indexer's
/// choice of them), the queries' and the output's low ranks in groups, then the experts (their block's; hash-routed in the first layers), around each block the four
/// streams' mix; the weighted hyper-head. `vo` what the level and the switches make of its products and caches. The matrices, small tensors and blocks are
/// dynamic/deepseek_v4.rs's loader's (`dynamic::deepseek_v4::describe` indexes them in its order).
pub fn deepseek_v4(d: &Described, eos: Vec<i64>, max_pos: usize, st: &super::Stage, vo: &super::deepseek_v4::V4Opts) -> Result<ModelDesc> {
    use super::deepseek_v4::{v4_desc, V4Shape};
    use crate::runtime::kinds::Residual;
    let m = &d.desc;
    if m.blocks.len() % 2 != 0 {
        return err(format!("DeepSeek-V4: {} blocks, where each layer is an attention block and an experts block", m.blocks.len()));
    }
    let layers = m.blocks.len() / 2;
    let (mut ratios, mut hash) = (Vec::with_capacity(layers), Vec::with_capacity(layers));
    let (mut first, mut idx, mut hc) = (None, (0, 0, 0), (0.0, 0, 0, crate::runtime::kinds::Rounding::Once));
    for l in 0..layers {
        let (ab, eb) = (&m.blocks[2 * l], &m.blocks[2 * l + 1]);
        let c = match &ab.body {
            Body::Mixer(Mixer::Compressed(c)) => c,
            b => return err(format!("DeepSeek-V4: layer {l}'s mixer is {b:?}")),
        };
        if !c.ape || !c.q_head_norm || c.share.is_some() {
            return err(format!(
                "DeepSeek-V4: layer {l} is V4.1's compressed sparse attention (no window bias, parts shared across layers): no operation of this executor yet"
            ));
        }
        if c.indexer.is_some() != (c.ratio == 4) {
            return err(format!(
                "DeepSeek-V4: layer {l} compresses by {} with{} an indexer: this engine runs the indexer on the layers of ratio 4 alone",
                c.ratio,
                if c.indexer.is_some() { "" } else { "out" }
            ));
        }
        if let Some(x) = &c.indexer {
            idx = (x.heads, x.head_dim, x.top_k);
        }
        let e = match &eb.body {
            Body::Ffn(Ffn::Moe(e)) => e,
            b => return err(format!("DeepSeek-V4: layer {l}'s feed-forward is {b:?}")),
        };
        for b in [ab, eb] {
            match &b.residual {
                Residual::Streams(s) if !s.lagged => hc = (s.eps, s.sinkhorn_iters, s.streams, s.rounding),
                r => return err(format!("DeepSeek-V4: layer {l}'s residual is {r:?}, not mHC's streams (lagged mHC has no operation yet)")),
            }
        }
        ratios.push(c.ratio);
        hash.push(e.router.hash);
        first = first.or(Some(c));
    }
    let c = first.ok_or_else(|| Error("DeepSeek-V4: no layers".into()))?;
    let s = V4Shape {
        d: m.hidden,
        vocab: m.vocab,
        eps: m.blocks[0].norm_in.eps,
        max_pos,
        eos,
        nh: c.heads,
        hd: c.head_dim,
        rope_dim: c.rope.dim,
        q_lora: c.q_lora,
        ol: c.o_lora,
        og: c.o_groups,
        wo_cols: c.heads * c.head_dim / c.o_groups.max(1),
        idx,
        window: c.window,
        ratios,
        hash,
        streams: hc.2,
        hc: (hc.0, hc.1),
        round: hc.3,
    };
    v4_desc(&s, st, vo)
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

    /// Nemotron-H lowered from Nemotron 3 Super's released config and from the mixture-of-experts toy's, whole and as a split's stages, is the executor's own
    /// description of it.
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
                let (a, b) = (format!("{own:#?}"), format!("{lowered:#?}"));
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

    /// Muse Spark lowered from the random model's config the logits checks run, whole and as a split's stages, is the executor's own description of it.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_muse_lowers_as_its_own_description() {
        let dir = std::env::temp_dir().join(format!("glyd-lower-muse-{}", std::process::id()));
        crate::muse_ref::random(&dir).unwrap();
        let c = crate::muse::ConfigMuse::load(&dir).unwrap();
        let half = c.layers / 2;
        for st in [
            super::super::Stage::whole(c.layers),
            super::super::Stage { layers: (0, half), first: true, last: false },
            super::super::Stage { layers: (half, c.layers), first: false, last: true },
        ] {
            let (own, lowered) =
                (super::super::muse::describe(&c, &st).unwrap(), super::super::muse::lowered(&dir, &c, &st).unwrap_or_else(|e| panic!("{}", e.msg)));
            assert_eq!(format!("{own:#?}"), format!("{lowered:#?}"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GLM-5.3-Flash lowered from its released config and the tiny and toy ones, whole and as a split's stages, is the executor's own description of it; each
    /// layer's matrices are as many as model_glm.rs's loader packs.
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
            for l in 0..c.layers {
                // (a latent attention layer's four more: the kind's operation's, mla_glm.rs's `MlaWeights`)
                let latent = if c.kinds[l] == crate::glm::GlmKind::Mla { 4 } else { 0 };
                assert_eq!(crate::model_glm::layer_matrices(&c, l).len() + latent, super::super::glm::per_layer(c.kinds[l], c.moe[l]), "{f}: layer {l}");
            }
            let kinds = crate::runtime::describe::describe(&serde_json::from_str(&text).unwrap()).unwrap();
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
                let (a, b) = (format!("{own:#?}"), format!("{lowered:#?}"));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("{f}: the lowered description differs at line {first:?}");
                }
            }
        }
    }

    /// The random GLM-5.3-Flash the logits and GPU checks run (bf16, and its experts in FP8), lowered whole and as a split's stages, is the executor's own description of it.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_glm_lowers_as_its_own_description() {
        for fp8 in [false, true] {
            let dir = std::env::temp_dir().join(format!("glyd-lower-glm-{}-{fp8}", std::process::id()));
            crate::kcheck_glm::random(&dir, fp8, None).unwrap();
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
                assert_eq!(format!("{own:#?}"), format!("{lowered:#?}"));
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
                let (a, b) = (format!("{own:#?}"), format!("{lowered:#?}"));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("the lowered description differs at line {first:?}");
                }
            }
        }
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
        let rows =
            |names: &[String]| -> usize { super::super::kimi::fused(names.iter().map(|n| ck.get(&format!("{n}.weight")).unwrap().1.shape[0] as usize).sum()) };
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
            assert_eq!(format!("{own:#?}"), format!("{lowered:#?}"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// DeepSeek-V4 lowered from its released config (the header fixture's), whole and as a split's stages, at the lossless level and another, the output groups
    /// stacked and not, every FP8 matrix held FP8 and the default's, is the executor's own description of it.
    #[test]
    fn deepseek_v4_lowered_is_the_executors_own_description() {
        use super::super::deepseek_v4::V4Opts;
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/DeepSeek-V4-Flash-0731.json");
        let h: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let text = h["config"].as_str().unwrap();
        let c = crate::deepseek_v4::ConfigV4::parse(text, "DeepSeek-V4-Flash", None).unwrap_or_else(|e| panic!("{}", e.msg));
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
        let half = c.layers / 2;
        let base = V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 4096 };
        for vo in [base, V4Opts { exact8: false, ..base }, V4Opts { ostack: false, ..base }, V4Opts { f8_min: 0, sim: false, ..base }] {
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let own = super::super::deepseek_v4::describe(&c, &st, &vo).unwrap();
                let lowered = deepseek_v4(&kinds, c.eos.clone(), c.max_pos, &st, &vo).unwrap_or_else(|e| panic!("{}", e.msg));
                let (a, b) = (format!("{own:#?}"), format!("{lowered:#?}"));
                if a != b {
                    let first = a.lines().zip(b.lines()).enumerate().find(|(_, (x, y))| x != y);
                    panic!("{vo:?}: the lowered description differs at line {first:?}");
                }
            }
        }
    }

    /// The streams' operations round as the kind says (`StreamsP::rounding`): DeepSeek-V4 described with transformers' bf16 rounding lowers to other operations
    /// than its loader's (the release's), and is not loaded; every stream operation of the release's description carries `Once`.
    #[test]
    fn the_streams_round_as_their_kind() {
        use super::super::deepseek_v4::V4Opts;
        use crate::runtime::kinds::{Residual, Rounding};
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/DeepSeek-V4-Flash-0731.json");
        let h: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let text = h["config"].as_str().unwrap();
        let c = crate::deepseek_v4::ConfigV4::parse(text, "DeepSeek-V4-Flash", None).unwrap_or_else(|e| panic!("{}", e.msg));
        let mut kinds = crate::runtime::describe::describe(&serde_json::from_str(text).unwrap()).unwrap();
        let (vo, st) = (V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 4096 }, super::super::Stage::whole(c.layers));
        let own = super::super::deepseek_v4::describe(&c, &st, &vo).unwrap();
        let rounds: Vec<Rounding> = own
            .layers
            .iter()
            .flatten()
            .filter_map(|op| match *op {
                Op::StreamsPre { round, .. } | Op::StreamsPost { round, .. } | Op::StreamsHead { round, .. } => Some(round),
                _ => None,
            })
            .collect();
        assert_eq!(rounds.len(), 4 * c.layers + 1);
        assert!(rounds.iter().all(|&r| r == Rounding::Once));
        for b in &mut kinds.desc.blocks {
            if let Residual::Streams(s) = &mut b.residual {
                s.rounding = Rounding::Twice;
            }
        }
        let lowered = deepseek_v4(&kinds, c.eos.clone(), c.max_pos, &st, &vo).unwrap();
        let e = agree(lowered, &own, "the test's config.json").unwrap_err().msg;
        assert!(e.contains("describe different models") && e.contains("Twice"), "{e}");
    }

    /// The random DeepSeek-V4 the logits and GPU checks run, lowered whole and as a split's stages, is the executor's own description of it.
    #[cfg(feature = "dev")]
    #[test]
    fn the_random_deepseek_v4_lowers_as_its_own_description() {
        use super::super::deepseek_v4::V4Opts;
        let dir = std::env::temp_dir().join(format!("glyd-lower-v4-{}", std::process::id()));
        crate::dsv4_ref::random(&dir, None).unwrap();
        let c = crate::deepseek_v4::ConfigV4::load(&dir).unwrap();
        let half = c.layers / 2;
        let base = V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 512 };
        for vo in [base, V4Opts { f8_min: 0, ..base }] {
            for st in [
                super::super::Stage::whole(c.layers),
                super::super::Stage { layers: (0, half), first: true, last: false },
                super::super::Stage { layers: (half, c.layers), first: false, last: true },
            ] {
                let (own, lowered) = (
                    super::super::deepseek_v4::describe(&c, &st, &vo).unwrap(),
                    super::super::deepseek_v4::lowered(&dir, &c, &st, &vo).unwrap_or_else(|e| panic!("{}", e.msg)),
                );
                assert_eq!(format!("{own:#?}"), format!("{lowered:#?}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A Llama-style config (model_type, the attention's settings) of `layers` layers of hidden 256, 4 query heads over 2 of 64, an MLP of 512, a vocabulary of 1,024:
    /// test shapes, no release's.
    fn llama_style(model_type: &str, extra: serde_json::Value) -> serde_json::Value {
        let mut c = serde_json::json!({"architectures": ["X"], "model_type": model_type, "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 1024, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 4096, "hidden_act": "silu", "tie_word_embeddings": false});
        c.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        c
    }

    /// Each Llama-style config of the fixtures (the random Qwen3 the logits checks run, a Mistral-layout one with a window, Qwen3's with its q and k norms
    /// and a tied head) lowers to the operations the dense Qwen3 engine ran (src/model.rs, removed): the input norm (then its add of the residual), the q, k
    /// and v product, the attention kind, o, the norm with the residual, gate and up with SiLU, down, the final norm on the last rows; its matrices the
    /// checkpoint's tensors stacked in that engine's order and sized by the description's shapes, which are the checkpoint's; its widths, heads, rotary and
    /// norms the config's as that engine read it (`weights::Config`, for Qwen3). Llama 3's scaled rope and Qwen2's biases are refused, named.
    #[test]
    fn dense_lowered_from_each_llama_style_config() {
        use super::super::attn::{QkNorm, RopeAt};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("tests/fixtures/qwen3_tiny_manifest.json")).unwrap()).unwrap();
        let tiny_text = manifest["config"].as_str().unwrap().to_string();
        let tiny_shapes: std::collections::HashMap<String, Vec<usize>> = manifest["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_str().unwrap().to_string(), t[2].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()))
            .collect();
        let cases = [
            ("the random Qwen3 of the logits checks", serde_json::from_str(&tiny_text).unwrap()),
            ("a Mistral layout, a window on every layer", llama_style("mistral", serde_json::json!({"sliding_window": 128}))),
            ("a Qwen3 layout, a tied head", llama_style("qwen3", serde_json::json!({"tie_word_embeddings": true, "rms_norm_eps": 1e-6}))),
            ("a Llama layout", llama_style("llama", serde_json::json!({}))),
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
            assert_eq!(format!("{:?}", d.ropes), format!("{:?}", [Rope { theta, dim: hd, pairs: hd / 2 }]), "{what}");
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
            if cfg["model_type"] == "qwen3" {
                // (and the config as the dense Qwen3 engine read it)
                let mut text = cfg.clone();
                text["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
                let c = crate::weights::Config::parse(&text.to_string(), what, None).unwrap();
                assert_eq!(
                    (c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter, c.layers, c.eps, c.rope_theta, c.tie),
                    (hid, heads, kv, hd, inter, layers, eps, theta, tied)
                );
            }
        }
        // (the random Qwen3's shapes are its checkpoint's, every one)
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(&tiny_text).unwrap()).unwrap();
        for (name, s) in &kinds.shapes {
            assert_eq!(Some(s), tiny_shapes.get(name), "{name}");
        }
        assert_eq!(kinds.shapes.len(), tiny_shapes.len());
        // (a kind the executor does not take: Llama 3's scaled rope, Qwen2's biases)
        let llama3 = crate::runtime::reference::testing::Fixture::load("llama_model").j["config"].clone();
        let e = dense(&crate::runtime::describe::describe(&llama3).unwrap(), vec![2], 4096).unwrap_err().msg;
        assert!(e.contains("rope") && e.contains("Llama3"), "{e}");
        let e = dense(&crate::runtime::describe::describe(&llama_style("qwen2", serde_json::json!({}))).unwrap(), vec![2], 4096).unwrap_err().msg;
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
