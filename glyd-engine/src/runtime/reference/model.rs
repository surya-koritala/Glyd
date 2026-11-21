//! A whole model on the reference backend: a description (`describe::Described`) loaded from a tensor source, one sequence's tokens to
//! logits. The per-model check of the design (§3): tiny models built from a family's real config, logits against transformers.

use super::{depth_mix, norm::norm, Params, RefLayer};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{Collapse, ModelDesc, Residual};

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
    /// The model's own roles (`Described::extra`).
    pub p: Params,
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
        let mut p = Params::new();
        for (role, s) in &d.extra {
            let (shape, v) = load(src, role, s)?;
            p.put(role, &shape, v);
        }
        Ok(RefModel { final_norm: get(src, &d.final_norm)?.1, embed, head, layers, desc, p })
    }

    /// The tokens of one sequence (slot 0) from position `start`: each token's logits.
    pub fn forward(&mut self, tokens: &[u32], start: u32) -> Result<Vec<Vec<f32>>> {
        let (d, v, n) = (self.desc.hidden, self.desc.vocab, self.desc.embed.streams.max(1));
        let e = &self.desc.embed;
        let mut x: Vec<f32> = Vec::with_capacity(tokens.len() * n * d);
        for &t in tokens {
            if t as usize >= v {
                return err(format!("token {t} past the vocabulary of {v}"));
            }
            let mut row: Vec<f32> = self.embed[t as usize * d..][..d].iter().map(|&x| x * e.scale).collect();
            if let Some(nm) = e.norm {
                row = norm(&nm, &row, None, None);
            }
            for _ in 0..n {
                x.extend_from_slice(&row);
            }
        }
        let pos: Vec<u32> = (0..tokens.len() as u32).map(|i| start + i).collect();
        let seqs = vec![0; tokens.len()];
        // the depth mix's sources per token, and the open group's partial sum (the embedding to begin with)
        let mut sources: Vec<Vec<Vec<f32>>> = vec![Vec::new(); tokens.len()];
        let mut partial: Vec<Option<Vec<f32>>> = x.chunks(n * d).map(|r| Some(r[..d].to_vec())).collect();
        for l in &mut self.layers {
            let Residual::DepthMix(dm) = l.block.residual else {
                l.forward(&pos, &seqs, tokens, &mut x)?;
                continue;
            };
            let (wn, wp) = (l.p.vec("mix_norm")?.to_vec(), l.p.vec("mix_proj")?.to_vec());
            for t in 0..tokens.len() {
                let pv = partial[t].take();
                let input = match (&pv, sources[t].is_empty()) {
                    (Some(p), true) => p.clone(),
                    _ => {
                        let mut all: Vec<&[f32]> = sources[t].iter().map(|s| &s[..]).collect();
                        all.extend(pv.as_deref());
                        depth_mix(&all, &wn, &wp)
                    }
                };
                let pv = if dm.starts {
                    sources[t].extend(pv);
                    None
                } else {
                    pv
                };
                let o = l.output(seqs[t], pos[t], tokens[t], &input)?;
                partial[t] = Some(match pv {
                    Some(p) => p.iter().zip(&o).map(|(a, b)| a + b).collect(),
                    None => o,
                });
            }
        }
        let h = &self.desc.head;
        let mut out = Vec::with_capacity(tokens.len());
        for (t, xt) in x.chunks(n * d).enumerate() {
            let r: Vec<f32> = match h.collapse {
                Collapse::None if n == 1 => xt.to_vec(),
                Collapse::None => return err(format!("{n} residual streams and no collapse into the head")),
                Collapse::Mean => (0..d).map(|i| (0..n).map(|j| xt[j * d + i]).sum::<f32>() / n as f32).collect(),
                Collapse::Weighted { eps, norm_eps } => {
                    let (f, base, sc) = (self.p.vec("out_hc_fn")?, self.p.vec("out_hc_base")?, self.p.vec("out_hc_scale")?[0]);
                    let r = 1.0 / (xt.iter().map(|v| v * v).sum::<f32>() / xt.len() as f32 + norm_eps).sqrt();
                    let flat: Vec<f32> = xt.iter().map(|v| v * r).collect();
                    let mixes = super::matvec(f, flat.len(), &flat);
                    let pre: Vec<f32> = (0..n).map(|j| super::sigmoid(mixes[j] * sc + base[j]) + eps).collect();
                    (0..d).map(|i| (0..n).map(|j| pre[j] * xt[j * d + i]).sum()).collect()
                }
                Collapse::DepthMix => {
                    let mut all: Vec<&[f32]> = sources[t].iter().map(|s| &s[..]).collect();
                    all.extend(partial[t].as_deref());
                    depth_mix(&all, self.p.vec("out_mix_norm")?, self.p.vec("out_mix_proj")?)
                }
            };
            let y = norm(&self.desc.final_norm, &r, Some(&self.final_norm), None);
            out.push(
                super::matvec(&self.head, d, &y)
                    .into_iter()
                    .map(|z| {
                        let z = z * h.scale;
                        h.softcap.map_or(z, |c| c * (z / c).tanh())
                    })
                    .collect(),
            );
        }
        Ok(out)
    }
}
