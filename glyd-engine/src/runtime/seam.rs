//! R9's spikes (#230): one kind, the dense MLP block, through the layer-kind seam on more than the reference: MLX (`runtime::mlx`, a
//! Mac) and CUDA (`runtime::cuda`). What a spike's `build` reads of the block and its weights, and the check that runs a backend's block
//! against the reference's on the same weights.
use super::backend::{HostTensor, LayerDesc, WeightSource};
use super::kinds::{Act, Body, Ffn, NormKind, NormWeight, Residual};
use crate::ffi::{err, Error, Result};

/// A dense MLP block's weights as a spike runs them: the input norm's [hidden], gate and up [inter, hidden], down [hidden, inter], and the
/// norm's eps. Refused: anything but a SwiGLU MLP (no bias, no limits) after a plain RMS norm, added to the stream.
pub struct MlpParts {
    pub hidden: usize,
    pub inter: usize,
    pub eps: f32,
    pub norm: HostTensor,
    pub gate: HostTensor,
    pub up: HostTensor,
    pub down: HostTensor,
}

pub fn mlp_parts(layer: &LayerDesc, w: &dyn WeightSource, who: &str) -> Result<MlpParts> {
    let b = layer.block;
    let Body::Ffn(Ffn::Mlp(m)) = &b.body else { return err(format!("layer {}: {who} runs the dense MLP alone", layer.index)) };
    let plain = b.norm_in.kind == NormKind::Rms && b.norm_in.weight == NormWeight::Plain && b.norm_out.is_none() && b.residual == Residual::Add;
    if !(m.gated && m.act == Act::Silu && !m.bias && m.clamp.is_none() && plain) {
        return err(format!("layer {}: {who} runs a SwiGLU MLP after a plain RMS norm, added to the stream", layer.index));
    }
    let d = layer.hidden;
    let get = |role: &str| w.f32(role).ok_or_else(|| Error(format!("layer {}: no {role}", layer.index)));
    let (norm, gate, up, down) = (get("norm_in")?, get("gate")?, get("up")?, get("down")?);
    let i = gate.shape.first().copied().unwrap_or(0);
    if norm.shape != [d] || gate.shape != [i, d] || up.shape != [i, d] || down.shape != [d, i] {
        return err(format!("layer {}: shapes {:?} {:?} {:?} {:?} for a hidden width of {d}", layer.index, norm.shape, gate.shape, up.shape, down.shape));
    }
    Ok(MlpParts { hidden: d, inter: i, eps: b.norm_in.eps, norm, gate, up, down })
}

#[cfg(any(test, feature = "dev"))]
pub use check::check_mlp;

#[cfg(any(test, feature = "dev"))]
mod check {
    use super::*;
    use crate::runtime::backend::{Arenas, Backend, Format, Placement, StepCx};
    use crate::runtime::kinds::{Block, Mlp, Norm};
    use crate::runtime::reference::testing::{fill, rel};
    use crate::runtime::reference::Ref;

    /// A dense MLP block's weights, by role, from the reference tests' generator.
    struct Weights {
        hidden: usize,
        inter: usize,
    }

    impl WeightSource for Weights {
        fn format(&self, _: &str) -> Option<Format> {
            Some(Format::F32)
        }
        fn f32(&self, role: &str) -> Option<HostTensor> {
            let (d, i) = (self.hidden, self.inter);
            let shape = match role {
                "norm_in" => vec![d],
                "gate" | "up" => vec![i, d],
                "down" => vec![d, i],
                _ => return None,
            };
            let (lo, hi) = if role == "norm_in" { (0.5, 1.5) } else { (-0.06, 0.06) };
            Some(HostTensor { data: fill(&format!("mlp.{role}"), shape.iter().product(), lo, hi), shape })
        }
    }

    /// The dense MLP block (SwiGLU after an RMS norm, added to the stream) of `hidden` and `inter` on `tokens` tokens, through backend `b` and
    /// through the reference, on the same weights and input: the largest difference over the reference's largest magnitude. The queue is made
    /// first (a GPU backend's context is current from then on, for its weights).
    pub fn check_mlp<B: Backend<Tensor = Vec<f32>>>(b: &B, hidden: usize, inter: usize, tokens: usize) -> Result<f32> {
        let block = Block {
            norm_in: Norm::rms(1e-6),
            body: Body::Ffn(Ffn::Mlp(Mlp { act: Act::Silu, gated: true, bias: false, clamp: None })),
            norm_out: None,
            residual: Residual::Add,
        };
        let (w, layer) = (Weights { hidden, inter }, LayerDesc { index: 0, block: &block, hidden });
        let x0 = fill("input", tokens * hidden, -1.0, 1.0);
        let (pos, seqs, toks): (Vec<u32>, Vec<u32>, Vec<u32>) = ((0..tokens as u32).collect(), vec![0; tokens], vec![1; tokens]);
        let mut want = x0.clone();
        Ref::default()
            .build(&layer, &w, &Placement::default(), &mut Arenas::default())?
            .run(&mut StepCx { queue: &mut (), positions: &pos, seqs: &seqs, tokens: &toks }, &mut want)?;
        let mut q = b.queue(0)?;
        let mut got = x0.clone();
        b.build(&layer, &w, &Placement::default(), &mut Arenas::default())?
            .run(&mut StepCx { queue: &mut q, positions: &pos, seqs: &seqs, tokens: &toks }, &mut got)?;
        if got == x0 {
            return err("the block left the stream as it was");
        }
        if got.iter().any(|v| !v.is_finite()) {
            return err("a value that is not finite");
        }
        Ok(rel(&got, &want))
    }
}
