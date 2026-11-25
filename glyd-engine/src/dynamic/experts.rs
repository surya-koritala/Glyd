//! A mixture of experts' block from its layer kind (`runtime::kinds::Moe`) and its roles' tensors, every expert on the GPU: the core's layer (src/moe.rs: the
//! router, the experts, the shared expert and its gate), which `Op::Moe` runs. For any family whose description names its experts a tensor each: the kind says
//! the router's scores, the experts' activation and whether the shared expert is gated; the roles say where each matrix is. The experts and the shared expert are
//! held at the level asked: a lossless level's stacked and packed as one matrix each (the packer's check: it decodes to the weights bit for bit), a block
//! level's (kestrel, swift, wren) an expert at a time.

use crate::ffi::{err, Error, Result};
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeWeights, Router, Scoring, Shared};
use crate::moe_rows::Down;
use crate::runtime::describe::Source;
use crate::runtime::kinds::{self, RouterScore};
use crate::weights::{item_words, ordered, pack_item_map, upload_bytes, upload_packed, Item, Layout};
use glyd_gpu::pack::Tiered;
use glyd_gpu::safetensors::Checkpoint;

/// The core layer's shape and arithmetic for the kind `m` over a model `hidden` wide, its experts `inter` wide. What the core's layer does not run is refused in
/// words: hash routing, grouped choices, a logit scale and latent experts are other operations' (`MoeOp`'s variants, a family's own block).
pub fn cfg(m: &kinds::Moe, hidden: usize, inter: usize) -> Result<MoeCfg> {
    let r = &m.router;
    if r.hash || r.groups.is_some() || r.logit_scale != 1.0 || m.latent.is_some() {
        return err(format!("experts of {m:?}: hash routing, grouped choices, a logit scale and latent experts are not the resident block's"));
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
    let act = match (m.expert.act, m.expert.gated) {
        (kinds::Act::Silu, true) => Act::SwiGlu { limit: m.expert.clamp },
        (kinds::Act::GeluTanh, true) if m.expert.clamp.is_none() => Act::GeGlu,
        (kinds::Act::Relu2, false) if m.expert.clamp.is_none() => Act::Relu2,
        (kinds::Act::Situ { beta, linear }, true) if m.expert.clamp.is_none() => Act::Situ { beta, linear },
        _ => return err(format!("experts of {:?}: the kernels take SwiGLU, GeGLU, relu squared and SiTU", m.expert)),
    };
    // (a softmax router is a bf16 product, its logits rounded before the scores; the others' scores are made in fp32)
    let router = Router { scoring, top_k: r.top_k, renorm: r.renorm, scale: r.scale, bf16_logits: scoring == Scoring::Softmax };
    Ok(MoeCfg { experts: m.experts, router_hidden: hidden, hidden, inter, router, act })
}

/// The block of the kind `m` whose tensors `roles` names, read from `ck`, its experts and shared expert at `level`.
pub fn resident(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, level: Layout) -> Result<MoeLayer> {
    let lossless = matches!(level, Layout::Tiered | Layout::Twelve);
    if !lossless && !matches!(level, Layout::Int6 | Layout::Int5 | Layout::Int4) {
        return err(format!("the experts are held at the penguin, kestrel, swift and wren levels: {} is asked", level.name()));
    }
    let find = |r: &str| roles.iter().find(|(n, _)| n == r).map(|(_, s)| s);
    let one = |r: &str| match find(r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        other => err(format!("the experts' role {r} is {other:?}, where one tensor is")),
    };
    let each = |r: &str| match find(r) {
        Some(Source::Experts(v)) if v.len() == m.experts => Ok(v),
        other => err(format!("the experts' role {r} is {other:?}, where a tensor (or its rows) an expert is, {} of them", m.experts)),
    };
    // (matrices stacked along their rows: their rows' sum and their columns)
    let dims = |members: &[String]| -> Result<(usize, usize)> {
        let (mut rows, mut cols) = (0, 0);
        for n in members {
            let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
            if t.shape.len() != 2 || (cols != 0 && cols != t.shape[1] as usize) {
                return err(format!("{n}: {:?}, where a matrix of {cols} columns is stacked", t.shape));
            }
            (rows, cols) = (rows + t.shape[0] as usize, t.shape[1] as usize);
        }
        Ok((rows, cols))
    };
    let bf16 = |n: &str| -> Result<Vec<u8>> {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        if t.dtype != "BF16" {
            return err(format!("{n} is {}, where the router and the shared expert's gate are read as BF16", t.dtype));
        }
        ck.read(n).map_err(|e| Error(format!("{n}: {e}")))
    };
    let gated = m.expert.gated;
    let (up, down) = (each(if gated { "experts_gate_up" } else { "experts_up" })?, each("experts_down")?);
    let (up_rows, _) = dims(&up[0])?;
    let c = cfg(m, hidden, if gated { up_rows / 2 } else { up_rows })?;
    let packed = |members: Vec<String>| -> Result<crate::weights::PackBuf> {
        let (rows, cols) = dims(&members)?;
        upload_packed(pack_item_map(ck, &Item { members, rows, cols }, level, &|_| {})?)
    };
    let (up, down) = if lossless {
        let stack = |v: &[Vec<String>]| ExpertStack::lossless(packed(v.iter().flatten().cloned().collect())?, m.experts);
        (stack(up)?, Down::Block(stack(down)?))
    } else {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let (mut ups, mut downs): (Vec<Tiered>, Vec<Tiered>) = (Vec::new(), Vec::new());
        ordered(
            2 * m.experts,
            threads,
            2 * threads,
            &|k| {
                let members = if k % 2 == 0 { &up[k / 2] } else { &down[k / 2] };
                let (rows, cols) = dims(members)?;
                let w = item_words(ck, &Item { members: members.clone(), rows, cols })?;
                let p = match level {
                    Layout::Int6 => glyd_gpu::pack::pack_int6(&w, rows, cols),
                    Layout::Int5 => glyd_gpu::pack::pack_int5(&w, rows, cols),
                    _ => glyd_gpu::pack::pack_int4(&w, rows, cols),
                };
                p.ok_or_else(|| Error(format!("{}: not packable at {}", members.join(" + "), level.name())))
            },
            |k, t| {
                (if k % 2 == 0 { &mut ups } else { &mut downs }).push(t);
                Ok(())
            },
        )?;
        let join = |v: &[Tiered]| Joined::of(&v.iter().collect::<Vec<_>>()).and_then(|j| ExpertStack::upload(&j));
        (join(&ups)?, Down::Block(join(&downs)?))
    };
    let bias = if m.router.bias {
        let n = one("choice_bias")?;
        let (_, t) = ck.get(&n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        Some(upload_bytes(&crate::moe_nh::bias_f32(&t.dtype, &ck.read(&n).map_err(|e| Error(format!("{n}: {e}")))?)?)?)
    } else {
        None
    };
    let shared = match m.shared {
        None => None,
        Some(_) => {
            let s_up = one("shared_up")?;
            let (inter, _) = dims(std::slice::from_ref(&s_up))?;
            let first = if gated { vec![one("shared_gate")?, s_up] } else { vec![s_up] };
            let gate = if m.shared_gate { Some(upload_bytes(&bf16(&one("shared_router")?)?)?) } else { None };
            Some(Shared { inter, up: packed(first)?, down: packed(vec![one("shared_down")?])?, gate })
        }
    };
    let w = MoeWeights { router: upload_bytes(&bf16(&one("router")?)?)?, bias, escale: None, up, down };
    MoeLayer::new(c, w, shared)
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
        refused(&|m| m.expert.act = kinds::Act::Gelu, "SwiGLU");
    }
}
