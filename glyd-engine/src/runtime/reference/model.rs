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

/// A checkpoint's tensor as f32: bf16 and f32 as stored; a hash table's expert ids (int64); and in a development build DeepSeek-V4's FP8
/// matrices and FP4 experts decoded with their scales, and an MXFP4 matrix (`.weight_packed`, `.weight_scale`: Kimi-K3's experts) decoded
/// where its `.weight` is asked for. Another type: none.
pub fn checkpoint_tensor(ck: &glyd_gpu::safetensors::Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let Some((_, t)) = ck.get(name) else { return mxfp4(ck, name) };
    let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
    let v = match t.dtype.as_str() {
        "BF16" => ck.read(name).ok()?.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F32" => ck.read(name).ok()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "I64" => ck.read(name).ok()?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()) as f32).collect(),
        #[cfg(feature = "dev")]
        "F8_E4M3" | "I8" => {
            let (sh, v) = crate::dsv4_ref::tensor_of(ck, name).ok()?;
            return Some((sh.iter().map(|&d| d as usize).collect(), v));
        }
        _ => return None,
    };
    Some((shape, v))
}

#[cfg(feature = "dev")]
fn mxfp4(ck: &glyd_gpu::safetensors::Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let base = name.strip_suffix(".weight")?;
    let (_, t) = ck.get(&format!("{base}.weight_packed"))?;
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize * 2);
    let w = crate::kimi_ref::mxfp4(&ck.read(&format!("{base}.weight_packed")).ok()?, &ck.read(&format!("{base}.weight_scale")).ok()?, rows, cols).ok()?;
    Some((vec![rows, cols], w))
}

#[cfg(not(feature = "dev"))]
fn mxfp4(_: &glyd_gpu::safetensors::Checkpoint, _: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    None
}

/// Kimi-K3's depth mix for one token: the completed groups' sums (the sources) and the open group's partial sum.
#[derive(Clone, Debug)]
pub struct Depth {
    pub sources: Vec<Vec<f32>>,
    pub partial: Option<Vec<f32>>,
}

impl Depth {
    /// The mix of the sources and the partial sum by the query `wn` * `wp`.
    pub fn mix(&self, wn: &[f32], wp: &[f32]) -> Vec<f32> {
        let mut all: Vec<&[f32]> = self.sources.iter().map(|s| &s[..]).collect();
        all.extend(self.partial.as_deref());
        depth_mix(&all, wn, wp)
    }

    /// A block's input: the partial sum itself while there are no sources, else the mix; a block that `starts` a group then closes the
    /// partial sum into a source.
    pub fn input(&mut self, starts: bool, wn: &[f32], wp: &[f32]) -> Vec<f32> {
        let input = match (&self.partial, self.sources.is_empty()) {
            (Some(p), true) => p.clone(),
            _ => self.mix(wn, wp),
        };
        if starts {
            let p = self.partial.take();
            self.sources.extend(p);
        }
        input
    }

    /// The block's output added to the partial sum (the output itself where a group just started).
    pub fn add(&mut self, o: &[f32]) {
        self.partial = Some(match self.partial.take() {
            Some(p) => p.iter().zip(o).map(|(a, b)| a + b).collect(),
            None => o.to_vec(),
        });
    }
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
        let mut depth: Vec<Depth> = x.chunks(n * d).map(|r| Depth { sources: Vec::new(), partial: Some(r[..d].to_vec()) }).collect();
        for l in &mut self.layers {
            let Residual::DepthMix(dm) = l.block.residual else {
                l.forward(&pos, &seqs, tokens, &mut x)?;
                continue;
            };
            let (wn, wp) = (l.p.vec("mix_norm")?.to_vec(), l.p.vec("mix_proj")?.to_vec());
            for (t, dt) in depth.iter_mut().enumerate() {
                let input = dt.input(dm.starts, &wn, &wp);
                let o = l.output(seqs[t], pos[t], tokens[t], &input)?;
                dt.add(&o);
            }
        }
        let h = &self.desc.head;
        let mut out = Vec::with_capacity(tokens.len());
        for (t, xt) in x.chunks(n * d).enumerate() {
            let y = self.final_h(xt, Some(&depth[t]))?;
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

    /// The head's input for one token: its residual `xt` (its streams) collapsed into one (`Head::collapse`; Kimi-K3's from the depth mix's
    /// state), then the final norm.
    pub fn final_h(&self, xt: &[f32], depth: Option<&Depth>) -> Result<Vec<f32>> {
        let (d, n) = (self.desc.hidden, self.desc.embed.streams.max(1));
        let r: Vec<f32> = match self.desc.head.collapse {
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
                let dp = depth.ok_or_else(|| Error("the depth mix's collapse without its sources".into()))?;
                dp.mix(self.p.vec("out_mix_norm")?, self.p.vec("out_mix_proj")?)
            }
            // (the streams normed by the final norm before the mix, none after it)
            Collapse::GatedMix => {
                let (down, up) = (self.p.vec("out_mix_down")?, self.p.vec("out_mix_up")?);
                return Ok(super::gated_mix(n, &self.desc.final_norm, &self.final_norm, down, up, None, xt)?.0);
            }
        };
        Ok(norm(&self.desc.final_norm, &r, Some(&self.final_norm), None))
    }
}
