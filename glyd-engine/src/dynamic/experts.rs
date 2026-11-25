//! A mixture of experts' block from its layer kind (`runtime::kinds::Moe`) and its roles' tensors: the core's layer (src/moe.rs: the router, the experts, the
//! shared expert and its gate), which `Op::Moe` runs. For any family whose description names its experts: the kind says the router's scores, the experts'
//! activation, whether they work in a latent and whether each one's output is normed, and whether the shared expert is gated; the roles say where each matrix is,
//! a tensor (or its rows) an expert or one tensor of them all stacked. The experts and the shared expert are held at the level asked: a lossless level's stacked
//! and packed as one matrix each (the packer's check: it decodes to the weights bit for bit), a block level's (kestrel, swift, wren) an expert at a time. A
//! stage can hold them otherwise (`Held`): this GPU's slice of each expert's rows where every layer is across a group of GPUs, a hummingbird directory's codebook
//! rows, the expert cache's places.

use super::{MoeOp, Slot};
use crate::ffi::{err, Error, Result};
use crate::hb_pack::HbDir;
use crate::hbg_store::HbCaching;
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeWeights, Router, Scoring, Shared};
use crate::moe_rows::Down;
use crate::offload::ExpertStore;
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{self, Body, Ffn, NormKind, NormWeight, RouterScore};
use crate::weights::{item_words, ordered, pack_item_map, pack_words, upload_bytes, upload_packed, Item, Layout};
use glyd_gpu::pack::Tiered;
use glyd_gpu::safetensors::Checkpoint;
use std::rc::Rc;

/// The norm of each expert's output before it is weighed, where the kind has one (`Latent::expert_norm`).
fn normed(m: &kinds::Moe) -> Option<kinds::Norm> {
    m.latent.as_ref().and_then(|l| l.expert_norm)
}

/// The core layer's shape and arithmetic for the kind `m` over a model `hidden` wide, its experts `inter` wide: the router reads the model's width, the experts
/// the latent's where the kind has one. Experts whose outputs are normed take each pair's activation from a kernel of its own, rounded (never the first
/// product's), and their router's logits times the kind's scale. What the core's layer does not run is refused in words: hash routing and grouped choices are
/// other operations' (`MoeOp`'s variants, a family's own block), a logit scale the normed experts' alone.
pub fn cfg(m: &kinds::Moe, hidden: usize, inter: usize) -> Result<MoeCfg> {
    let r = &m.router;
    if r.hash || r.groups.is_some() {
        return err(format!("experts of {m:?}: hash routing and grouped choices are not the resident block's"));
    }
    if r.logit_scale != 1.0 && normed(m).is_none() {
        return err(format!("experts of {m:?}: a logit scale is the router's of experts whose outputs are normed"));
    }
    let scoring = match r.score {
        RouterScore::Softmax => Scoring::Softmax,
        RouterScore::Sigmoid => Scoring::Sigmoid,
        RouterScore::SqrtSoftplus => Scoring::SqrtSoftplus,
    };
    if r.bias && scoring == Scoring::Softmax {
        return err("a softmax router with a choice bias: the kernels add the bias to sigmoid and sqrt-softplus scores");
    }
    if m.expert.bias || m.shared.is_some_and(|s| s != m.expert) {
        return err(format!("experts of {:?} beside a shared expert of {:?}: the kernels take one activation and no biases", m.expert, m.shared));
    }
    let act = match (m.expert.act, m.expert.gated, normed(m)) {
        (kinds::Act::Silu, true, Some(n)) if n.kind == NormKind::Rms && n.weight == NormWeight::InputPlusOne && m.shared.is_none() => {
            Act::SwiGlu { limit: Some(m.expert.clamp.unwrap_or(f32::INFINITY)) }
        }
        (_, _, Some(n)) => {
            return err(format!(
                "experts of {:?} whose outputs take {n:?}: the kernels norm SwiGLU experts' outputs, input-scaled, with no shared expert",
                m.expert
            ))
        }
        (kinds::Act::Silu, true, None) => Act::SwiGlu { limit: m.expert.clamp },
        (kinds::Act::GeluTanh, true, None) if m.expert.clamp.is_none() => Act::GeGlu,
        (kinds::Act::Relu2, false, None) if m.expert.clamp.is_none() => Act::Relu2,
        (kinds::Act::Situ { beta, linear }, true, None) if m.expert.clamp.is_none() => Act::Situ { beta, linear },
        _ => return err(format!("experts of {:?}: the kernels take SwiGLU, GeGLU, relu squared and SiTU", m.expert)),
    };
    // (a softmax router is a bf16 product, its logits rounded before the scores; the others' scores are made in fp32)
    let router = Router { scoring, top_k: r.top_k, renorm: r.renorm, scale: r.scale, bf16_logits: scoring == Scoring::Softmax };
    Ok(MoeCfg { experts: m.experts, router_hidden: hidden, hidden: m.latent.as_ref().map_or(hidden, |l| l.dim), inter, router, act })
}

/// The operation of block `block` whose experts work in a latent and whose outputs are normed: the router reads H by its fp32 weights, its logits times `mult`
/// (the kind's `Router::logit_scale`), the experts the normed latent Xc, each expert's output normed (input-scaled by the small `post_w`, eps) before it is
/// weighed, their sum into M (`MoeLayer::forward_normed`).
pub(crate) fn latent_normed(block: usize, post_w: usize, eps: f32, mult: f32) -> MoeOp {
    MoeOp { x: Slot::Xc, y: Slot::M, logit_scale: mult, expert_norm: Some((post_w, eps)), ..MoeOp::of(block) }
}

/// The blocks of experts of the layers `layers` of a model of two blocks a layer (a mixer, then its feed-forward), in `Op::Moe`'s order: each one's layer, its
/// kind and its roles.
pub fn of_layers(d: &Described, layers: (usize, usize)) -> impl Iterator<Item = (usize, &kinds::Moe, &[(String, Source)])> {
    d.desc.blocks.iter().zip(&d.blocks).enumerate().filter_map(move |(b, (blk, roles))| match &blk.body {
        Body::Ffn(Ffn::Moe(m)) if (layers.0..layers.1).contains(&(b / 2)) => Some((b / 2, m, roles.as_slice())),
        _ => None,
    })
}

/// How a stage holds its blocks of experts beyond their level: a hummingbird directory's (src/hb_pack.rs; None: the release's), the expert cache (the stage's
/// caching and its store: src/hbg_store.rs, src/tiering.rs), on the GPU `ordinal`. A layer of a stage so held is across the thread's group of GPUs where it
/// has one (src/dynamic/tp.rs): this GPU's rows of each expert's.
#[derive(Clone, Copy)]
pub struct Held<'a> {
    pub hb: Option<&'a HbDir>,
    pub cache: Option<(&'a HbCaching, &'a Rc<ExpertStore>)>,
    pub ordinal: i32,
}

/// A role's experts as the checkpoint holds them: a tensor (or its rows stacked) an expert, or one tensor of them all [experts, rows, cols].
enum Stacked<'a> {
    Each(&'a [Vec<String>]),
    One(&'a str),
}

impl Stacked<'_> {
    fn of<'a>(src: Option<&'a Source>, experts: usize, role: &str) -> Result<Stacked<'a>> {
        match src {
            Some(Source::Experts(v)) if v.len() == experts && experts > 0 => Ok(Stacked::Each(v)),
            Some(Source::Tensor(n)) => Ok(Stacked::One(n)),
            other => {
                err(format!("the experts' role {role} is {other:?}, where a tensor (or its rows) an expert is, {experts} of them, or one tensor of them all"))
            }
        }
    }

    /// The name in an error.
    fn name(&self) -> String {
        match self {
            Stacked::Each(v) => v.iter().flatten().cloned().collect::<Vec<_>>().join(" + "),
            Stacked::One(n) => n.to_string(),
        }
    }

    /// Expert `e`'s name in an error.
    fn name_of(&self, e: usize) -> String {
        match self {
            Stacked::Each(v) => v[e].join(" + "),
            Stacked::One(n) => format!("{n}: expert {e}"),
        }
    }

    /// An expert's rows and its columns.
    fn dims(&self, ck: &Checkpoint, experts: usize) -> Result<(usize, usize)> {
        match self {
            Stacked::Each(v) => rows_cols(ck, &v[0]),
            Stacked::One(n) => {
                let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
                if t.dtype != "BF16" || t.shape.len() != 3 || t.shape[0] as usize != experts {
                    return err(format!("{n}: {} {:?}, where the bf16 matrices of {experts} experts are stacked", t.dtype, t.shape));
                }
                Ok((t.shape[1] as usize, t.shape[2] as usize))
            }
        }
    }

    /// Every expert's matrix, stacked [experts rows, cols]: bf16 words.
    fn words(&self, ck: &Checkpoint, experts: usize) -> Result<Vec<u16>> {
        let (rows, cols) = self.dims(ck, experts)?;
        match self {
            Stacked::Each(v) => item_words(ck, &Item { members: v.iter().flatten().cloned().collect(), rows: experts * rows, cols }),
            Stacked::One(n) => {
                let mut w = vec![0u16; experts * rows * cols];
                ck.read_bf16(n, &mut w).map_err(|e| Error(format!("{n}: {e}")))?;
                Ok(w)
            }
        }
    }

    /// Expert `e`'s matrix [rows, cols] (`all`: every expert's, where the checkpoint holds them as one tensor).
    fn expert(&self, ck: &Checkpoint, e: usize, all: &[u16], rows: usize, cols: usize) -> Result<Vec<u16>> {
        match self {
            Stacked::Each(v) => item_words(ck, &Item { members: v[e].clone(), rows, cols }),
            Stacked::One(_) => Ok(all[e * rows * cols..(e + 1) * rows * cols].to_vec()),
        }
    }
}

/// Matrices stacked along their rows: their rows' sum and their columns.
fn rows_cols(ck: &Checkpoint, members: &[String]) -> Result<(usize, usize)> {
    let (mut rows, mut cols) = (0, 0);
    for n in members {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        if t.shape.len() != 2 || (cols != 0 && cols != t.shape[1] as usize) {
            return err(format!("{n}: {:?}, where a matrix of {cols} columns is stacked", t.shape));
        }
        (rows, cols) = (rows + t.shape[0] as usize, t.shape[1] as usize);
    }
    Ok((rows, cols))
}

/// The block of the kind `m` whose tensors `roles` names, read from `ck`, its experts and shared expert at `level`, every expert on the GPU.
pub fn resident(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, level: Layout) -> Result<MoeLayer> {
    held(m, roles, hidden, ck, level, None)
}

/// `resident`, the experts of decoder layer `at.0` held as the stage holds them (`at.1`): a hummingbird directory's in place of the release's (in the expert
/// cache where the stage has one), the release's in the expert cache's places (each matrix's one lossless pack in the tiered layout, cut per expert), or on the
/// GPU, this GPU's rows of each expert's where the layer is across a group of GPUs.
pub fn held(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, level: Layout, at: Option<(usize, &Held)>) -> Result<MoeLayer> {
    let lossless = matches!(level, Layout::Tiered | Layout::Twelve);
    if !lossless && !matches!(level, Layout::Int6 | Layout::Int5 | Layout::Int4) {
        return err(format!("the experts are held at the penguin, kestrel, swift and wren levels: {} is asked", level.name()));
    }
    let find = |r: &str| roles.iter().find(|(n, _)| n == r).map(|(_, s)| s);
    let one = |r: &str| match find(r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        other => err(format!("the experts' role {r} is {other:?}, where one tensor is")),
    };
    let read = |n: &str| ck.read(n).map_err(|e| Error(format!("{n}: {e}")));
    let dtype = |n: &str| ck.get(n).map(|(_, t)| t.dtype.clone()).ok_or_else(|| Error(format!("no tensor {n}")));
    let bf16 = |n: &str| -> Result<Vec<u8>> {
        if dtype(n)? != "BF16" {
            return err(format!("{n} is {}, where the router and the shared expert's gate are read as BF16", dtype(n)?));
        }
        read(n)
    };
    let (e, gated, norm) = (m.experts, m.expert.gated, normed(m));
    let up_role = if gated { "experts_gate_up" } else { "experts_up" };
    let (up, down) = (Stacked::of(find(up_role), e, up_role)?, Stacked::of(find("experts_down"), e, "experts_down")?);
    // (the experts' width: a hummingbird directory's record's, which holds no matrix of theirs; else their first matrix's rows)
    let inter = match at.and_then(|(l, h)| h.hb.map(|d| (d, l))) {
        Some((d, l)) => d.layer(2 * l + 1).ok_or_else(|| Error(format!("{}: no experts for decoder layer {l}", d.dir.display())))?.inter,
        None => {
            let (rows, _) = up.dims(ck, e)?;
            if gated {
                rows / 2
            } else {
                rows
            }
        }
    };
    let c = cfg(m, hidden, inter)?;
    let packed = |members: Vec<String>| -> Result<crate::weights::PackBuf> {
        let (rows, cols) = rows_cols(ck, &members)?;
        upload_packed(pack_item_map(ck, &Item { members, rows, cols }, level, &|_| {})?)
    };
    // (the normed experts' router is read by its fp32 weights, `MoeLayer::forward_normed`; another's is a bf16 product)
    let router = {
        let n = one("router")?;
        match norm {
            Some(_) => upload_bytes(&crate::moe_nh::bias_f32(&dtype(&n)?, &read(&n)?)?)?,
            None => upload_bytes(&bf16(&n)?)?,
        }
    };
    let bias = if m.router.bias {
        let n = one("choice_bias")?;
        Some(upload_bytes(&crate::moe_nh::bias_f32(&dtype(&n)?, &read(&n)?)?)?)
    } else {
        None
    };
    let shared = match m.shared {
        None => None,
        Some(_) => {
            let s_up = one("shared_up")?;
            let (inter, _) = rows_cols(ck, std::slice::from_ref(&s_up))?;
            let first = if gated { vec![one("shared_gate")?, s_up] } else { vec![s_up] };
            let gate = if m.shared_gate { Some(upload_bytes(&bf16(&one("shared_router")?)?)?) } else { None };
            Some(Shared { inter, up: packed(first)?, down: packed(vec![one("shared_down")?])?, gate })
        }
    };
    // (a stage that holds its experts its own way has its layers across the thread's group of GPUs, where it is one's)
    let across = at.and_then(|_| super::tp::slicer());
    if let Some((l, h)) = at {
        if h.hb.is_some() {
            let Some(n) = norm else {
                return err("a hummingbird directory's experts of a kind whose outputs are not normed: its family's loader reads them");
            };
            // (the host's pairs of the cache normed as the sum norms each expert's output: the norm's input scale as the GPU holds it, bf16)
            let w: Vec<f32> = match h.cache {
                Some(_) => {
                    let b = bf16(&one("expert_norm")?)?;
                    let w: Vec<f32> = b.as_chunks::<2>().0.iter().map(|x| crate::hb_codec::f32_of(u16::from_le_bytes(*x))).collect();
                    crate::muse::plus_one(&w).iter().map(|&v| crate::hb_codec::f32_of(crate::hb_codec::bf16(v))).collect()
                }
                None => Vec::new(),
            };
            let finish = crate::cpu_hbg::Finish::Normed { signs: Vec::new(), rot: 0, w, eps: n.eps };
            if let Some(x) = crate::hbg_store::hb_layer(h.hb, l, across.is_some(), h.cache, c.act, finish, h.ordinal, c.router.top_k)? {
                return x.into_layer(c, router, bias, None, shared);
            }
        }
        // (R12: the release's experts in the expert cache, every one in the computer's memory: each stack's one lossless pack in the tiered layout, whatever the level of the rest, cut per expert)
        if let Some((cg, store)) = h.cache {
            if across.is_some() || cg.ram.is_some() {
                return err("the release's experts in the expert cache: a layer on one GPU, every expert in the computer's memory (the disk holds hummingbird directories' experts alone)");
            }
            let places = |s: &Stacked| -> Result<crate::tiering::TieredPlaces> {
                let (rows, cols) = s.dims(ck, e)?;
                let p = glyd_gpu::pack::pack_tiered_checked(&s.words(ck, e)?, e * rows, cols)
                    .ok_or_else(|| Error(format!("{}: decoded to other bits than its weights", s.name())))?;
                crate::tiering::TieredPlaces::of(&p, e)
            };
            let (pu, pd) = (places(&up)?, places(&down)?);
            let (up, down, cache) = crate::tiering::cached_tiered(&pu, &pd, cg.places, store, l, h.ordinal)?;
            return MoeLayer::new(c, MoeWeights { router, bias, escale: None, up, down: Down::Block(down) }, shared)?.with_cache(cache);
        }
    }
    let (up, down) = if lossless {
        let stack = |s: &Stacked| -> Result<ExpertStack> {
            let (rows, cols) = s.dims(ck, e)?;
            let w = s.words(ck, e)?;
            let Some(t) = across else {
                return ExpertStack::lossless(upload_packed(pack_words(&w, e * rows, cols, level, &s.name())?)?, e);
            };
            // (a group of GPUs: this GPU's rows of each expert's, stacked)
            if rows % (64 * t.n) != 0 {
                return err(format!("{}: experts of {rows} rows across {} GPUs: each GPU's rows in 64s are wanted", s.name(), t.n));
            }
            let (r0, rn) = t.rows(rows);
            let mine: Vec<u16> = (0..e).flat_map(|x| w[(x * rows + r0) * cols..(x * rows + r0 + rn) * cols].iter().copied()).collect();
            let mut st = ExpertStack::lossless(upload_packed(pack_words(&mine, e * rn, cols, level, &s.name())?)?, e)?;
            st.tp_rows = rows;
            Ok(st)
        };
        (stack(&up)?, Down::Block(stack(&down)?))
    } else {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        // (where the checkpoint holds a matrix of every expert as one tensor: read once, cut per expert)
        let all = |s: &Stacked| match s {
            Stacked::One(_) => s.words(ck, e),
            Stacked::Each(_) => Ok(Vec::new()),
        };
        let (all_up, all_down) = (all(&up)?, all(&down)?);
        let (du, dd) = (up.dims(ck, e)?, down.dims(ck, e)?);
        let (mut ups, mut downs): (Vec<Tiered>, Vec<Tiered>) = (Vec::new(), Vec::new());
        ordered(
            2 * e,
            threads,
            2 * threads,
            &|k| {
                let (s, all, (rows, cols)) = if k % 2 == 0 { (&up, &all_up, du) } else { (&down, &all_down, dd) };
                let w = s.expert(ck, k / 2, all, rows, cols)?;
                let p = match level {
                    Layout::Int6 => glyd_gpu::pack::pack_int6(&w, rows, cols),
                    Layout::Int5 => glyd_gpu::pack::pack_int5(&w, rows, cols),
                    _ => glyd_gpu::pack::pack_int4(&w, rows, cols),
                };
                p.ok_or_else(|| Error(format!("{}: not packable at {}", s.name_of(k / 2), level.name())))
            },
            |k, t| {
                (if k % 2 == 0 { &mut ups } else { &mut downs }).push(t);
                Ok(())
            },
        )?;
        let join = |v: &[Tiered]| Joined::of(&v.iter().collect::<Vec<_>>()).and_then(|j| ExpertStack::upload(&j));
        (join(&ups)?, Down::Block(join(&downs)?))
    };
    MoeLayer::new(c, MoeWeights { router, bias, escale: None, up, down }, shared)
}

#[cfg(test)]
mod tests {
    use crate::moe::{Act, Scoring};
    use crate::runtime::kinds::{self, Mlp, Moe, Router, RouterScore};

    fn qwen() -> Moe {
        let mlp = Mlp { act: kinds::Act::Silu, gated: true, bias: false, clamp: None };
        Moe {
            router: Router { score: RouterScore::Softmax, logit_scale: 1.0, bias: false, groups: None, top_k: 10, renorm: true, scale: 1.0, hash: false },
            experts: 512,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: true,
            latent: None,
        }
    }

    /// The kind's router and activation as the core's layer takes them: a softmax router's logits bf16, a sigmoid's scores fp32 with its scale, the SwiGLU clamp.
    #[test]
    fn a_kind_becomes_the_core_layers_shape() {
        let c = super::cfg(&qwen(), 8192, 2048).unwrap();
        assert_eq!((c.experts, c.router_hidden, c.hidden, c.inter, c.up_rows()), (512, 8192, 8192, 2048, 4096));
        assert_eq!((c.router.scoring, c.router.top_k, c.router.renorm, c.router.scale, c.router.bf16_logits), (Scoring::Softmax, 10, true, 1.0, true));
        assert_eq!(c.act, Act::SwiGlu { limit: None });
        c.check().unwrap();
        let mut g = qwen();
        g.router = Router { score: RouterScore::Sigmoid, bias: true, scale: 2.5, top_k: 8, ..g.router };
        (g.expert.clamp, g.shared, g.shared_gate) = (Some(10.0), None, false);
        let c = super::cfg(&g, 4096, 2048).unwrap();
        assert_eq!((c.router.scoring, c.router.scale, c.router.bf16_logits, c.act), (Scoring::Sigmoid, 2.5, false, Act::SwiGlu { limit: Some(10.0) }));
    }

    /// Experts that work in a latent and whose outputs are normed: the experts' width is the latent's, the router's the model's; each pair's activation is a
    /// kernel's of its own (an infinite clamp: never the first product's, so a stack's rows can be sliced across GPUs), and the router's logit scale is
    /// theirs, the operation's to apply. A latent whose outputs are not normed keeps the first product's activation.
    #[test]
    fn experts_in_a_latent_with_normed_outputs() {
        use crate::runtime::kinds::{Latent, Norm, NormWeight};
        let mut m = qwen();
        m.router = Router { score: RouterScore::Sigmoid, logit_scale: 0.5, bias: true, top_k: 4, scale: 1.0, ..m.router };
        (m.experts, m.shared, m.shared_gate) = (16, None, false);
        let normed = Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(1e-8) };
        m.latent = Some(Latent { dim: 512, norm_in: Some(Norm::rms_plus_one(1e-5)), expert_norm: Some(normed), norm_out: None });
        let c = super::cfg(&m, 1024, 512).unwrap();
        assert_eq!((c.experts, c.router_hidden, c.hidden, c.inter, c.fused()), (16, 1024, 512, 512, 0));
        assert_eq!((c.router.scoring, c.router.bf16_logits, c.act), (Scoring::Sigmoid, false, Act::SwiGlu { limit: Some(f32::INFINITY) }));
        c.check().unwrap();
        let op = super::latent_normed(3, 7, 1e-8, 0.5);
        assert_eq!((op.block, op.x, op.y, op.logit_scale, op.expert_norm), (3, super::Slot::Xc, super::Slot::M, 0.5, Some((7, 1e-8))));
        // (a plain weight on the outputs' norm, or a shared expert beside them, is not the kernels')
        let mut plain = m.clone();
        plain.latent.as_mut().unwrap().expert_norm = Some(Norm::rms(1e-8));
        assert!(super::cfg(&plain, 1024, 512).unwrap_err().msg.contains("input-scaled"));
        let mut unnormed = m.clone();
        unnormed.latent.as_mut().unwrap().expert_norm = None;
        unnormed.router.logit_scale = 1.0;
        let c = super::cfg(&unnormed, 1024, 512).unwrap();
        assert_eq!((c.router_hidden, c.hidden, c.fused(), c.act), (1024, 512, 1, Act::SwiGlu { limit: None }));
    }

    /// What the resident block does not run is refused in words.
    #[test]
    fn what_the_block_does_not_run_is_refused() {
        let refused = |edit: &dyn Fn(&mut Moe), word: &str| {
            let mut m = qwen();
            edit(&mut m);
            let e = super::cfg(&m, 1024, 256).unwrap_err().msg;
            assert!(e.contains(word), "{e}");
        };
        refused(&|m| m.router.hash = true, "hash routing");
        refused(&|m| m.router.groups = Some((8, 4)), "grouped choices");
        refused(&|m| m.router.logit_scale = 2.0, "logit scale");
        refused(&|m| m.router.bias = true, "choice bias");
        refused(&|m| m.expert.bias = true, "no biases");
        refused(
            &|m| {
                m.expert.act = kinds::Act::Gelu;
                m.shared = Some(m.expert);
            },
            "SwiGLU",
        );
    }
}
