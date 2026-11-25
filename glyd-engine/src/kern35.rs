//! The Qwen3.5 layout's GPU pieces that the description executor (src/dynamic) and Condor (src/model_qwen4x.rs) run (its kernel table, `Kern35`, is src/kern/qwen35.rs's):
//! a packed matrix (`Mat`) and where each of a layer's matrices comes from (`layer_mats`: a pack the library makes, or tensors kept as the checkpoint has them). The
//! executor's loader (src/dynamic, `Dyn::new_stage`) reads the tensors the description names; this file loads none.
//!
//! A layer: x += mixer(norm1(x)); x += mlp(norm2(x)), the norms zero-centered. Full attention (1 in 4): the qkv product (each query head with its gate), the q/k norms and
//! the rotary on 64 of 256 dims, the causal attention times sigmoid(gate) (the attention kind's kernels, src/dynamic/attn.rs), the o product. Linear attention: the qkvz and ba products, the causal convolution with SiLU,
//! the gated delta rule (a recurrence, or in chunks for a prompt), the gated norm, the out product.

use crate::dev::Buf;
use crate::qwen35::{Config35, Kind};
use crate::weights::PackBuf;

pub use crate::kern::qwen35::Kern35;

/// A matrix of the model: packed, or bf16 as the checkpoint has it (rows, cols) where the library does not pack it.
pub enum Mat {
    Packed(PackBuf),
    Dense(Buf, usize, usize),
}

impl Mat {
    pub fn bytes(&self) -> usize {
        match self {
            Mat::Packed(p) => p.bytes,
            Mat::Dense(b, _, _) => b.bytes,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Mat::Packed(p) => p.pack.rows as usize,
            Mat::Dense(_, r, _) => *r,
        }
    }

    /// Its product's rows: the whole matrix's where this GPU holds a slice of them.
    pub fn whole_rows(&self) -> usize {
        match self {
            Mat::Packed(p) => p.whole_rows(),
            Mat::Dense(_, r, _) => *r,
        }
    }

    pub(crate) fn packed_opt(&self) -> Option<&PackBuf> {
        match self {
            Mat::Packed(p) => Some(p),
            Mat::Dense(..) => None,
        }
    }
}

/// Where a matrix of a layer comes from: a pack of `Config35::packs` (by its key), or the tensors kept as the checkpoint has them, stacked by rows (rows, cols).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Src {
    Pack(String),
    Dense(Vec<String>, usize, usize),
}

/// Each layer's matrices in the forward pass's order: full attention qkv, o; linear attention qkvz, ba, out; then gate_up, down. A matrix the packer does not pack (b and a
/// of each layer, 16 to 48 rows a part on the 2026 dense models) is the parts it keeps, stacked.
pub fn layer_mats(cfg: &Config35) -> Vec<Vec<Src>> {
    let keys: std::collections::HashSet<String> = cfg.packs().into_iter().map(|p| p.key).collect();
    let h = cfg.hidden;
    let one = |key: String, parts: &[(&str, usize)], module: &str| {
        if keys.contains(&key) {
            Src::Pack(key)
        } else {
            Src::Dense(parts.iter().map(|(n, _)| format!("{module}.{n}.weight")).collect(), parts.iter().map(|p| p.1).sum(), h)
        }
    };
    cfg.kinds
        .iter()
        .enumerate()
        .map(|(l, kind)| {
            let p = format!("{}layers.{l}", cfg.prefix);
            let mut v = match kind {
                Kind::Full => vec![Src::Pack(format!("{p}.self_attn.q_proj")), Src::Pack(format!("{p}.self_attn.o_proj"))],
                Kind::Linear => {
                    let m = format!("{p}.linear_attn");
                    vec![
                        Src::Pack(format!("{m}.in_proj_qkv")),
                        one(format!("{m}.in_proj_b"), &[("in_proj_b", cfg.lin_v_heads), ("in_proj_a", cfg.lin_v_heads)], &m),
                        Src::Pack(format!("{m}.out_proj")),
                    ]
                }
            };
            v.push(Src::Pack(format!("{p}.mlp.gate_proj")));
            v.push(Src::Pack(format!("{p}.mlp.down_proj")));
            v
        })
        .collect()
}

/// `x` rounded to bf16 (to nearest, ties to even), as an f32.
pub(crate) fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every layer's matrices are packs the packer makes or tensors it keeps, each pack taken once (the embedding and the LM head apart): the 2026 dense configs.
    #[test]
    fn the_layers_take_every_pack_once() {
        let all = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap();
        for name in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"] {
            let c = Config35::parse(&glyd_gpu::json::to_compact(all.get(name).unwrap()), name, None).unwrap();
            let plan = layer_mats(&c);
            assert_eq!(plan.len(), c.layers);
            let packs = c.packs();
            let kept: std::collections::HashSet<String> = c.kept().into_iter().map(|t| t.name).collect();
            let mut used = std::collections::HashSet::new();
            for (l, (srcs, (kind, _))) in plan.iter().zip(c.schedule()).enumerate() {
                assert_eq!(srcs.len(), if kind == Kind::Full { 4 } else { 5 }, "{name} layer {l}");
                for s in srcs {
                    match s {
                        Src::Pack(k) => {
                            assert!(packs.iter().any(|p| &p.key == k), "{name}: {k} is not a pack");
                            assert!(used.insert(k.clone()), "{name}: {k} twice");
                        }
                        Src::Dense(parts, rows, cols) => {
                            assert!(parts.iter().all(|p| kept.contains(p)), "{name}: {parts:?} not kept");
                            assert_eq!((*rows, *cols), (2 * c.lin_v_heads, c.hidden));
                        }
                    }
                }
            }
            let left: Vec<&str> = packs.iter().map(|p| p.key.as_str()).filter(|k| !used.contains(*k)).collect();
            assert_eq!(left, if c.tie { vec!["embed"] } else { vec!["embed", "lm_head"] }, "{name}");
        }
    }
}
