//! A whole model on the reference backend: a description (`describe::Described`) loaded from a tensor source, one sequence's tokens to
//! logits. The per-model check of the design (§3): tiny models built from a family's real config, logits against transformers.

use super::{norm::norm, Params, RefLayer};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::ModelDesc;

/// A checkpoint's tensors by name: (shape, values as f32).
pub type Tensors<'a> = &'a dyn Fn(&str) -> Option<(Vec<usize>, Vec<f32>)>;

fn get(src: Tensors, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    src(name).ok_or_else(|| Error(format!("the checkpoint has no tensor {name}")))
}

/// Matrices stacked along their rows.
fn rows(src: Tensors, names: &[String]) -> Result<(Vec<usize>, Vec<f32>)> {
    let (mut shape, mut data) = (Vec::new(), Vec::new());
    for n in names {
        let (s, v) = get(src, n)?;
        if shape.is_empty() {
            shape = s;
        } else if s.len() != shape.len() || s[1..] != shape[1..] {
            return err(format!("{n}: {s:?} does not stack under {shape:?}"));
        } else {
            shape[0] += s[0];
        }
        data.extend(v);
    }
    Ok((shape, data))
}

pub fn load(src: Tensors, role: &str, s: &Source) -> Result<(Vec<usize>, Vec<f32>)> {
    let (mut shape, data) = match s {
        Source::Tensor(n) => get(src, n)?,
        Source::Rows(v) => rows(src, v)?,
        Source::Experts(v) => {
            let (mut shape, mut data) = (Vec::new(), Vec::new());
            for e in v {
                let (s, d) = rows(src, e)?;
                shape = s;
                data.extend(d);
            }
            shape.insert(0, v.len());
            (shape, data)
        }
    };
    // (a conv1d weight [channels, 1, taps] is [channels, taps])
    if role == "conv" {
        shape.retain(|&d| d != 1);
    }
    Ok((shape, data))
}

pub struct RefModel {
    pub desc: ModelDesc,
    pub embed: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub head: Vec<f32>,
    pub layers: Vec<RefLayer>,
}

impl RefModel {
    pub fn load(d: &Described, src: Tensors) -> Result<RefModel> {
        let desc = d.desc.clone();
        let (es, embed) = get(src, &d.embed)?;
        if es != [desc.vocab, desc.hidden] {
            return err(format!("{}: {es:?}, not [{}, {}]", d.embed, desc.vocab, desc.hidden));
        }
        let head = match &d.head {
            Some(h) => get(src, h)?.1,
            None => embed.clone(),
        };
        let mut layers = Vec::new();
        for (b, roles) in desc.blocks.iter().zip(&d.blocks) {
            let mut p = Params::new();
            for (role, s) in roles {
                let (shape, v) = load(src, role, s)?;
                p.put(role, &shape, v);
            }
            layers.push(RefLayer::new(b.clone(), desc.hidden, p));
        }
        Ok(RefModel { final_norm: get(src, &d.final_norm)?.1, embed, head, layers, desc })
    }

    /// The tokens of one sequence (slot 0) from position `start`: each token's logits.
    pub fn forward(&mut self, tokens: &[u32], start: u32) -> Result<Vec<Vec<f32>>> {
        let (d, v) = (self.desc.hidden, self.desc.vocab);
        let mut x: Vec<f32> = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            if t as usize >= v {
                return err(format!("token {t} past the vocabulary of {v}"));
            }
            x.extend(self.embed[t as usize * d..][..d].iter().map(|&e| e * self.desc.embed.scale));
        }
        let pos: Vec<u32> = (0..tokens.len() as u32).map(|i| start + i).collect();
        let seqs = vec![0; tokens.len()];
        for l in &mut self.layers {
            l.forward(&pos, &seqs, tokens, &mut x)?;
        }
        let h = &self.desc.head;
        Ok(x.chunks(d)
            .map(|xt| {
                let y = norm(&self.desc.final_norm, xt, Some(&self.final_norm), None);
                super::matvec(&self.head, d, &y)
                    .into_iter()
                    .map(|z| {
                        let z = z * h.scale;
                        h.softcap.map_or(z, |c| c * (z / c).tanh())
                    })
                    .collect()
            })
            .collect())
    }
}
