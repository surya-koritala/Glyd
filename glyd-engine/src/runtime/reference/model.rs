//! A whole model on the reference backend: a description (`describe::Described`) loaded from a tensor source, one sequence's tokens to
//! logits. The per-model check of the design (§3): tiny models built from a family's real config, logits against transformers.

use super::{depth_mix, norm::norm, Params, RefLayer, SeqState};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{Body, Collapse, Comp, Mixer, Mla, ModelDesc, Residual, StreamsP};

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
    /// Where Some: `forward` adds the residual (every token's streams) after each block that ends a layer (a feed-forward), a check's view of each layer.
    pub record: Option<Vec<Vec<f32>>>,
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
        Ok(RefModel { final_norm: get(src, &d.final_norm)?.1, embed, head, layers, desc, p, record: None })
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
        // (lagged mHC: each token's input weights from the last lagged block; one-hot on stream 0 before the first)
        let mut lag: Vec<Vec<f32>> = vec![(0..n).map(|j| (j == 0) as u32 as f32).collect(); tokens.len()];
        for b in 0..self.layers.len() {
            self.share_into(b, 0)?;
            let l = &mut self.layers[b];
            if let Residual::Streams(s @ StreamsP { lagged: true, .. }) = l.block.residual {
                for (t, xt) in x.chunks_mut(n * d).enumerate() {
                    let (pre, post, comb) = super::hc_weights(&s, &l.p, xt)?;
                    let input: Vec<f32> = (0..d).map(|i| (0..n).map(|j| lag[t][j] * xt[j * d + i]).sum()).collect();
                    let o = l.output(seqs[t], pos[t], tokens[t], &input)?;
                    l.post(xt, &o, Some((post, comb)))?;
                    lag[t] = pre;
                }
                continue;
            }
            let Residual::DepthMix(dm) = l.block.residual else {
                l.forward(&pos, &seqs, tokens, &mut x)?;
                if matches!(l.block.body, Body::Ffn(_)) {
                    if let Some(r) = self.record.as_mut() {
                        r.push(x.clone());
                    }
                }
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
            let y = self.final_h_with(xt, Some(&depth[t]), &lag[t])?;
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

    /// V4.1's compressed attention of block `b` reading other blocks' parts (`kinds::CompShare`): sequence `seq`'s entries (and their indexer
    /// keys), picks and candidate blocks copied from the blocks that make them, which ran before it. A query reads only what its position sees.
    fn share_into(&mut self, b: usize, seq: u32) -> Result<()> {
        // (a latent attention whose indexer is `shared`: the picks the latent attention of the layer before it made for this token, as transformers hands
        // the next layer `prev_topk_indices`; a layer before it that picks nothing lends none)
        if let Body::Mixer(Mixer::Mla(Mla { indexer: Some(ix), .. })) = &self.layers[b].block.body {
            if ix.shared {
                let before = (0..b).rev().find(|&s| matches!(self.layers[s].block.body, Body::Mixer(_)));
                let lent = match before.and_then(|s| self.layers[s].seqs.get(&seq)) {
                    Some(SeqState::Mla(c)) if matches!(&self.layers[before.unwrap_or(0)].block.body, Body::Mixer(Mixer::Mla(Mla { indexer: Some(_), .. }))) => {
                        c.picks.clone()
                    }
                    _ => return err(format!("block {b}: a shared indexer after a layer whose attention picks nothing")),
                };
                let st = self.layers[b].seqs.entry(seq).or_insert_with(|| SeqState::Mla(super::mla::LatentCache::default()));
                let SeqState::Mla(st) = st else { return err(format!("block {b}: a sequence's state is not a latent cache")) };
                st.lent = Some(lent);
            }
            return Ok(());
        }
        let Body::Mixer(Mixer::Compressed(Comp { share: Some(sh), .. })) = &self.layers[b].block.body else { return Ok(()) };
        let sh = *sh;
        let entries = sh.entries.map(|e| self.comp_state(b, e, seq).map(|c| c.map(|c| (c.comp.clone(), c.ikeys.clone())))).transpose()?.flatten();
        let picks = sh.picks.map(|e| self.comp_state(b, e, seq).map(|c| c.map(|c| c.picks.clone()))).transpose()?.flatten();
        let cands = sh.candidates.map(|e| self.comp_state(b, e, seq).map(|c| c.map(|c| c.cands.clone()))).transpose()?.flatten();
        let st = self.layers[b].seqs.entry(seq).or_insert_with(|| SeqState::Comp(super::comp::CompState::default()));
        let SeqState::Comp(st) = st else { return err(format!("block {b}: a sequence's state is not a compressed attention's")) };
        if let Some((pool, keys)) = entries {
            (st.comp, st.ikeys) = (pool, keys);
        }
        if let Some(p) = picks {
            st.picks = p;
        }
        if let Some(c) = cands {
            st.cands = c;
        }
        Ok(())
    }

    /// Block `src`'s compressed attention state of sequence `seq`, which block `b` reads (None before the sequence reached it).
    fn comp_state(&self, b: usize, src: usize, seq: u32) -> Result<Option<&super::comp::CompState>> {
        match self.layers.get(src).map(|l| l.seqs.get(&seq)) {
            None => err(format!("block {b} reads block {src}'s compressed attention, which the model has not")),
            Some(None) => Ok(None),
            Some(Some(SeqState::Comp(c))) => Ok(Some(c)),
            Some(Some(_)) => err(format!("block {b} reads block {src}, which is no compressed attention")),
        }
    }

    /// The head's input for one token: its residual `xt` (its streams) collapsed into one (`Head::collapse`; Kimi-K3's from the depth mix's
    /// state), then the final norm.
    pub fn final_h(&self, xt: &[f32], depth: Option<&Depth>) -> Result<Vec<f32>> {
        self.final_h_with(xt, depth, &[])
    }

    /// `final_h` with lagged mHC's last input weights `pre` (`Collapse::LastPre`).
    pub fn final_h_with(&self, xt: &[f32], depth: Option<&Depth>, pre: &[f32]) -> Result<Vec<f32>> {
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
            Collapse::LastPre if pre.len() == n => (0..d).map(|i| (0..n).map(|j| pre[j] * xt[j * d + i]).sum()).collect(),
            Collapse::LastPre => return err(format!("lagged mHC's collapse with {} weights for {n} streams", pre.len())),
            // (the streams normed by the final norm before the mix, none after it)
            Collapse::GatedMix => {
                let (down, up) = (self.p.vec("out_mix_down")?, self.p.vec("out_mix_up")?);
                return Ok(super::gated_mix(n, &self.desc.final_norm, &self.final_norm, down, up, None, xt)?.0);
            }
        };
        Ok(norm(&self.desc.final_norm, &r, Some(&self.final_norm), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::{Act, Block, Body, Embed, Ffn, Head, Mlp, Norm, Rounding};
    use crate::runtime::reference::{ffn, hc_weights, matvec, testing::fill};

    const D: usize = 8;
    const N: usize = 4;

    fn streams(lagged: bool) -> StreamsP {
        StreamsP { streams: N, sinkhorn_iters: 20, eps: 1e-6, norm_eps: 1e-6, rounding: Rounding::F32, lagged }
    }

    fn mlp() -> Mlp {
        Mlp { act: Act::Silu, gated: true, bias: false, clamp: None }
    }

    /// A block of a gated MLP over `N` streams, its weights by `k`.
    fn block(k: usize, s: StreamsP) -> RefLayer {
        let mut p = Params::new();
        for (role, shape) in [("gate", [12, D]), ("up", [12, D]), ("down", [D, 12]), ("hc_fn", [(2 + N) * N, N * D])] {
            p.put(role, &shape, fill(&format!("lag{k}.{role}"), shape[0] * shape[1], -0.5, 0.5));
        }
        p.put("hc_base", &[(2 + N) * N], fill(&format!("lag{k}.hc_base"), (2 + N) * N, -0.5, 0.5));
        p.put("hc_scale", &[3], vec![0.7, 0.9, 1.1]);
        p.put("norm_in", &[D], fill(&format!("lag{k}.norm_in"), D, 0.5, 1.5));
        RefLayer::new(Block { norm_in: Norm::rms(1e-6), body: Body::Ffn(Ffn::Mlp(mlp())), norm_out: None, residual: Residual::Streams(s) }, D, p)
    }

    fn model(s: StreamsP, collapse: Collapse) -> RefModel {
        let layers = vec![block(0, s), block(1, s)];
        let desc = ModelDesc {
            hidden: D,
            vocab: 16,
            embed: Embed { scale: 1.0, norm: None, streams: N },
            blocks: layers.iter().map(|l| l.block.clone()).collect(),
            final_norm: Norm::rms(1e-6),
            head: Head { tied: false, softcap: None, scale: 1.0, collapse },
            mtp: Vec::new(),
            vision: None,
            rope: None,
        };
        RefModel {
            desc,
            embed: fill("lag.embed", 16 * D, -1.0, 1.0),
            final_norm: fill("lag.fn", D, 0.5, 1.5),
            head: fill("lag.head", 16 * D, -0.5, 0.5),
            layers,
            p: Params::new(),
            record: None,
        }
    }

    /// Lagged mHC (DeepSeek-V4.1): a block's input is the streams weighed by the previous block's own input weights, the first block's by
    /// stream 0 alone, the head's by the last block's; written out by hand for two blocks it is the model's logits to the bit, and other
    /// logits than mHC's own-weights wiring.
    #[test]
    fn lagged_mhc_hands_each_blocks_weights_to_the_next() {
        let (s, tokens) = (streams(true), [3u32, 7, 1]);
        let mut m = model(s, Collapse::LastPre);
        let got = m.forward(&tokens, 0).unwrap();
        let n = |x: &[f32], w: Option<&[f32]>| norm(&Norm::rms(1e-6), x, w, None);
        let weigh = |w: &[f32], x: &[f32]| -> Vec<f32> { (0..D).map(|i| (0..N).map(|j| w[j] * x[j * D + i]).sum()).collect() };
        for (t, &id) in tokens.iter().enumerate() {
            let mut x: Vec<f32> = (0..N).flat_map(|_| m.embed[id as usize * D..][..D].to_vec()).collect();
            let mut prev = vec![1.0, 0.0, 0.0, 0.0];
            for l in &m.layers {
                let (pre, post, comb) = hc_weights(&s, &l.p, &x).unwrap();
                let o = ffn::mlp(&mlp(), &l.p, "", &n(&weigh(&prev, &x), l.p.opt("norm_in"))).unwrap();
                let old = x.clone();
                for k in 0..N {
                    for i in 0..D {
                        x[k * D + i] = post[k] * o[i] + (0..N).map(|j| comb[j][k] * old[j * D + i]).sum::<f32>();
                    }
                }
                prev = pre;
            }
            assert_eq!(got[t], matvec(&m.head, D, &n(&weigh(&prev, &x), Some(&m.final_norm))), "token {t}");
        }
        let own = model(streams(false), Collapse::Mean).forward(&tokens, 0).unwrap();
        assert!(own.iter().zip(&got).all(|(a, b)| a != b));
        // (a lagged block alone has no input weights to take)
        assert!(block(0, s).pre(&vec![0.5; N * D]).is_err());
    }

    /// Where asked, the forward pass keeps the residual after each layer's last block (its feed-forward): every token's streams, the last the head's input.
    #[test]
    fn a_recorded_pass_keeps_each_layers_residual() {
        let tokens = [3u32, 7, 1];
        let mut m = model(streams(false), Collapse::Mean);
        m.record = Some(Vec::new());
        let logits = m.forward(&tokens, 0).unwrap();
        let r = m.record.take().unwrap();
        assert_eq!((r.len(), r[0].len()), (2, tokens.len() * N * D));
        assert_ne!(r[0], r[1]);
        let mut again = model(streams(false), Collapse::Mean);
        assert_eq!(again.forward(&tokens, 0).unwrap(), logits, "recording leaves the logits as they are");
        assert!(again.record.is_none());
    }
}
