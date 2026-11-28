//! A whole model on the reference backend: a description (`describe::Described`) loaded from a tensor source, one sequence's tokens to
//! logits. The per-model check of the design (§3): tiny models built from a family's real config, logits against transformers.

use super::Libm;
use super::{depth_mix, norm::norm, Held, Params, RefLayer, SeqState};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{Body, Collapse, Comp, Head, Mixer, Mla, ModelDesc, Residual, Rounding, StreamsP};

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
        Source::RowsOf(n, ranges) => {
            let (shape, v) = get(src, n)?;
            let [r, c] = shape[..] else { return err(format!("{n}: {shape:?}, where rows are cut from a matrix")) };
            if let Some((a, b)) = ranges.iter().find(|(a, b)| a > b || *b > r) {
                return err(format!("{n}: rows {a}..{b} of a matrix of {r}"));
            }
            let data: Vec<f32> = ranges.iter().flat_map(|&(a, b)| v[a * c..b * c].iter().copied()).collect();
            (vec![data.len() / c.max(1), c], data)
        }
        Source::Transposed(n) => {
            let (shape, v) = get(src, n)?;
            let [c, r] = shape[..] else { return err(format!("{n}: {shape:?}, where a matrix is stored transposed")) };
            (vec![r, c], (0..r).flat_map(|i| (0..c).map(move |j| (i, j))).map(|(i, j)| v[j * r + i]).collect())
        }
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

/// A checkpoint's tensor as f32, by the type it is stored in (`formats::tensor`): bf16 and f32 as stored, a hash table's expert ids (int64), and a
/// quantized matrix decoded with the scales stored beside it (FP8 a block a power-of-two scale, FP4 a run of 32 one, as a release names them or as
/// compressed-tensors packs them), whatever the model is called. Another type: none.
pub fn checkpoint_tensor(ck: &glyd_gpu::safetensors::Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    crate::formats::tensor(ck, name)
}

/// One logit from the head's product `z`: times the head's scale, through its soft-cap, rounded where the head rounds (`Head::rounding`).
pub fn logit(h: &Head, z: f32) -> f32 {
    let (product, ops) = match h.rounding {
        Rounding::F32 => (false, false),
        Rounding::Once => (true, false),
        Rounding::Twice => (true, true),
    };
    let r = |v: f32, at: bool| if at { super::bf16(v) } else { v };
    let z = r(r(z, product) * h.scale, ops);
    h.softcap.map_or(z, |c| r(c * r(r(z / c, ops).ltanh(), ops), ops))
}

/// Kimi-K3's depth mix for one token: the completed groups' sums (the sources) and the open group's partial sum.
#[derive(Clone, Debug)]
pub struct Depth {
    pub sources: Vec<Vec<f32>>,
    pub partial: Option<Vec<f32>>,
}

impl Depth {
    /// The mix of the sources and the partial sum by the query `wn` * `wp`, each normed with `eps`.
    pub fn mix(&self, wn: &[f32], wp: &[f32], eps: f32) -> Vec<f32> {
        let mut all: Vec<&[f32]> = self.sources.iter().map(|s| &s[..]).collect();
        all.extend(self.partial.as_deref());
        depth_mix(&all, wn, wp, eps)
    }

    /// A block's input: the partial sum itself while there are no sources, else the mix; a block that `starts` a group then closes the
    /// partial sum into a source.
    pub fn input(&mut self, starts: bool, wn: &[f32], wp: &[f32], eps: f32) -> Vec<f32> {
        let input = match (&self.partial, self.sources.is_empty()) {
            (Some(p), true) => p.clone(),
            _ => self.mix(wn, wp, eps),
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
    /// The embedding [vocab, hidden], held as `Held` holds a tensor (bf16 where its values are).
    pub embed: Held,
    pub final_norm: Vec<f32>,
    /// The LM head's matrix; None where the head is the embedding (a tied head, held once).
    pub head: Option<Held>,
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
        let embed = Held::new(es, embed);
        let head = match &d.head {
            Some(h) => {
                let (s, v) = get(src, h)?;
                Some(Held::new(s, v))
            }
            None => None,
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

    /// Token `t`'s row of the embedding, as f32.
    pub fn embed_row(&self, t: usize) -> Vec<f32> {
        let d = self.desc.hidden;
        self.embed.w().row(t * d, d)
    }

    /// The LM head's matrix as a product reads it: its own, or the embedding where the head is tied.
    pub fn head_w(&self) -> super::W<'_> {
        self.head.as_ref().unwrap_or(&self.embed).w()
    }

    /// Every weight held in f32, the tied head still the embedding (a check's: what the model held as stored is held to).
    #[cfg(test)]
    pub fn widen(&mut self) {
        self.layers.iter_mut().for_each(|l| l.p.widen());
        self.p.widen();
        self.embed = self.embed.widened();
        self.head = self.head.as_ref().map(Held::widened);
    }

    /// The bytes the model's weights hold (`Held::bytes`): every block's, the model's own roles, the embedding and the head once.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.p.bytes()).sum::<usize>()
            + self.p.bytes()
            + self.embed.bytes()
            + self.head.as_ref().map_or(0, Held::bytes)
            + 4 * self.final_norm.len()
    }

    /// The model of `d` from here on with its release's simulated quantizations of its activations in the form of FP8 blocks of `block` (src/cpu_fp4.rs `Sim`;
    /// the kinds' reference simulates the form of 32s, src/dsv4_ref.rs `RefV4` that of 128s): every product whose matrix the checkpoint stores quantized (the
    /// description's `scales`) takes its input rounded as the release's act_quant rounds it, and the compressed attention's caches, indexer and output and the
    /// experts' activations are rounded as that release rounds them (`Params::sim`).
    pub fn simulate(&mut self, d: &Described, block: usize) -> Result<()> {
        if block != 32 {
            return err(format!("the kinds' reference simulates a release's quantizations of FP8 blocks of 32, not of {block}"));
        }
        for (l, roles) in self.layers.iter_mut().zip(&d.blocks) {
            l.p.sim = block;
            l.p.quantized = roles.iter().filter(|(_, s)| s.names().iter().any(|n| d.scales.contains_key(n))).map(|(r, _)| r.clone()).collect();
        }
        Ok(())
    }

    /// The tokens of one sequence (slot 0) from position `start`: each token's logits.
    pub fn forward(&mut self, tokens: &[u32], start: u32) -> Result<Vec<Vec<f32>>> {
        self.forward_seen(tokens, start, &mut |_| {})
    }

    /// `forward`, with the tokens' residual (each token's streams, one token after another) shown to `seen` before each block and after the last: what a
    /// check holds another backend's residual to, block by block, or rounds as another backend stores it. (A depth mix's residual is its sources and
    /// partial sums, which this does not show.)
    pub fn forward_seen(&mut self, tokens: &[u32], start: u32, seen: &mut dyn FnMut(&mut [f32])) -> Result<Vec<Vec<f32>>> {
        let (d, v, n) = (self.desc.hidden, self.desc.vocab, self.desc.embed.streams.max(1));
        let e = &self.desc.embed;
        let mut x: Vec<f32> = Vec::with_capacity(tokens.len() * n * d);
        for &t in tokens {
            if t as usize >= v {
                return err(format!("token {t} past the vocabulary of {v}"));
            }
            let mut row: Vec<f32> = self.embed_row(t as usize).into_iter().map(|x| x * e.scale).collect();
            if let Some(nm) = e.norm {
                row = norm(&nm, &row, None, None);
            }
            for _ in 0..n {
                x.extend_from_slice(&row);
            }
        }
        let pos: Vec<u32> = (0..tokens.len() as u32).map(|i| start + i).collect();
        let seqs = vec![0; tokens.len()];
        let ple: Vec<Vec<f32>> = tokens.iter().map(|&t| self.per_layer_input(t)).collect::<Result<_>>()?;
        let ple_dim = e.per_layer.map_or(0, |p| p.dim);
        // the depth mix's sources per token, and the open group's partial sum (the embedding to begin with)
        let mut depth: Vec<Depth> = x.chunks(n * d).map(|r| Depth { sources: Vec::new(), partial: Some(r[..d].to_vec()) }).collect();
        // (lagged mHC: each token's input weights from the last lagged block; one-hot on stream 0 before the first)
        let mut lag: Vec<Vec<f32>> = vec![(0..n).map(|j| (j == 0) as u32 as f32).collect(); tokens.len()];
        for b in 0..self.layers.len() {
            seen(&mut x);
            if let Some(SeqState::Mla(c)) = self.layers[b].seqs.get_mut(&0) {
                c.made.clear();
            }
            self.share_into(b, 0)?;
            let (before, rest) = self.layers.split_at_mut(b);
            let l = &mut rest[0];
            // (an attention that shares an earlier block's keys and values reads that block's cache)
            let store = match l.kv_from() {
                Some(j) if j < b => Some(&before[j]),
                Some(j) => return err(format!("block {b} shares the keys and values of block {j}, not an earlier one")),
                None => None,
            };
            // (a per-layer input's branch: each token's residual and its layer's input)
            if let Body::PerLayer(pl) = l.block.body {
                for (t, xt) in x.chunks_mut(n * d).enumerate() {
                    let input = &ple[t][pl.layer * ple_dim..][..ple_dim];
                    let o = l.norm_out(super::ffn::per_layer(&pl, &l.p, xt, input)?);
                    l.post(xt, &o, None)?;
                }
                continue;
            }
            if let Residual::Streams(s @ StreamsP { lagged: true, .. }) = l.block.residual {
                for (t, xt) in x.chunks_mut(n * d).enumerate() {
                    let (pre, post, comb) = super::hc_weights(&s, &l.p, xt)?;
                    let input = super::collapse(&s, &lag[t], xt, d);
                    let o = l.output_with(store, seqs[t], pos[t], tokens[t], &input)?;
                    l.post(xt, &o, Some((post, comb)))?;
                    lag[t] = pre;
                }
                continue;
            }
            let Residual::DepthMix(dm) = l.block.residual else {
                l.forward_with(store, &pos, &seqs, tokens, &mut x)?;
                if matches!(l.block.body, Body::Ffn(_)) {
                    if let Some(r) = self.record.as_mut() {
                        r.push(x.clone());
                    }
                }
                continue;
            };
            let (wn, wp) = (l.p.vec("mix_norm")?.to_vec(), l.p.vec("mix_proj")?.to_vec());
            for (t, dt) in depth.iter_mut().enumerate() {
                let input = dt.input(dm.starts, &wn, &wp, l.block.norm_in.eps);
                let o = l.output(seqs[t], pos[t], tokens[t], &input)?;
                dt.add(&o);
            }
        }
        seen(&mut x);
        let h = self.desc.head;
        let mut out = Vec::with_capacity(tokens.len());
        for (t, xt) in x.chunks(n * d).enumerate() {
            let y = self.final_h_with(xt, Some(&depth[t]), &lag[t])?;
            out.push(super::matvec_w(self.head_w(), d, &y).into_iter().map(|z| logit(&h, z)).collect());
        }
        Ok(out)
    }

    /// The token's per-layer inputs (`kinds::PerLayerIn`), every layer's in order (none where the model has none): its row of `ple_embed` times sqrt(dim),
    /// plus its scaled embedding's product by `ple_proj` times hidden^-0.5 normed a layer at a time by `ple_norm`, the sum times 2^-0.5.
    pub fn per_layer_input(&self, token: u32) -> Result<Vec<f32>> {
        let (d, e) = (self.desc.hidden, &self.desc.embed);
        let Some(pl) = e.per_layer else { return Ok(Vec::new()) };
        let n = pl.layers * pl.dim;
        if token as usize >= pl.vocab {
            return err(format!("token {token} past the per-layer inputs' vocabulary of {}", pl.vocab));
        }
        let row: Vec<f32> = self.embed_row(token as usize).into_iter().map(|x| x * e.scale).collect();
        let s = (d as f32).lpowf(-0.5);
        let ctx: Vec<f32> = super::matvec(self.p.vec("ple_proj")?, d, &row).into_iter().map(|v| v * s).collect();
        let w = self.p.vec("ple_norm")?;
        let own = &self.p.vec("ple_embed")?[token as usize * n..][..n];
        let (r, k) = ((pl.dim as f32).sqrt(), std::f32::consts::FRAC_1_SQRT_2);
        Ok(ctx
            .chunks(pl.dim)
            .zip(own.chunks(pl.dim))
            .flat_map(|(c, o)| norm(&pl.norm, c, Some(w), None).into_iter().zip(o).map(move |(c, &o)| (c + o * r) * k))
            .collect())
    }

    /// Block `b`'s body on one token's normed input `h`, with the block whose keys and values it shares where it does (`Attn::kv_from`), a compressed
    /// attention with the parts it reads of other blocks (`CompShare`), and a per-layer input's branch with its token's input.
    pub fn body(&mut self, b: usize, seq: u32, pos: u32, token: u32, h: &[f32]) -> Result<Vec<f32>> {
        // (V4.1's compressed attention reading other blocks' parts: theirs as they stand, which ran before it)
        if matches!(self.layers.get(b).map(|l| &l.block.body), Some(Body::Mixer(Mixer::Compressed(Comp { share: Some(_), .. })))) {
            self.share_into(b, seq)?;
        }
        if let Some(Body::PerLayer(pl)) = self.layers.get(b).map(|l| l.block.body.clone()) {
            let dim = self.desc.embed.per_layer.map_or(0, |p| p.dim);
            let input = self.per_layer_input(token)?;
            return super::ffn::per_layer(&pl, &self.layers[b].p, h, &input[pl.layer * dim..][..dim]);
        }
        let (before, rest) = self.layers.split_at_mut(b);
        let l = rest.first_mut().ok_or_else(|| Error(format!("no block {b}")))?;
        let store = match l.kv_from() {
            Some(j) if j < b => Some(&before[j]),
            Some(j) => return err(format!("block {b} shares the keys and values of block {j}, not an earlier one")),
            None => None,
        };
        l.body_with(store, seq, pos, token, h)
    }

    /// V4.1's compressed attention of block `b` reading other blocks' parts (`kinds::CompShare`): sequence `seq`'s entries (and their indexer
    /// keys), picks and candidate blocks copied from the blocks that make them, which ran before it. A query reads only what its position sees.
    fn share_into(&mut self, b: usize, seq: u32) -> Result<()> {
        // (a latent attention whose indexer is `shared`: the picks the latent attention of the layer before it made for each token of this call, in order,
        // as transformers hands the next layer `prev_topk_indices` (a token's own each: a call of a prompt's tokens runs a layer for them all before the
        // next layer); a layer before it that picks nothing lends none. Each latent attention's picks of a call start empty: `forward_seen`)
        if let Body::Mixer(Mixer::Mla(Mla { indexer: Some(ix), .. })) = &self.layers[b].block.body {
            if ix.shared {
                let before = (0..b).rev().find(|&s| matches!(self.layers[s].block.body, Body::Mixer(_)));
                let lent = match before.and_then(|s| self.layers[s].seqs.get(&seq)) {
                    Some(SeqState::Mla(c)) if matches!(&self.layers[before.unwrap_or(0)].block.body, Body::Mixer(Mixer::Mla(Mla { indexer: Some(_), .. }))) => {
                        c.made.clone()
                    }
                    _ => return err(format!("block {b}: a shared indexer after a layer whose attention picks nothing")),
                };
                let st = self.layers[b].seqs.entry(seq).or_insert_with(|| SeqState::Mla(super::mla::LatentCache::default()));
                let SeqState::Mla(st) = st else { return err(format!("block {b}: a sequence's state is not a latent cache")) };
                st.lent = lent.into();
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

    /// The model's streams' kind (its blocks' first; the final collapse rounds as they do), or none's (F32) where it has no streams.
    fn streams_p(&self) -> StreamsP {
        self.desc.blocks.iter().find_map(|b| if let Residual::Streams(s) = b.residual { Some(s) } else { None }).unwrap_or(StreamsP {
            streams: 1,
            sinkhorn_iters: 0,
            eps: 0.0,
            norm_eps: 0.0,
            rounding: crate::runtime::kinds::Rounding::F32,
            lagged: false,
            eps_outside: false,
        })
    }

    /// `final_h` with lagged mHC's last input weights `pre` (`Collapse::LastPre`).
    pub fn final_h_with(&self, xt: &[f32], depth: Option<&Depth>, pre: &[f32]) -> Result<Vec<f32>> {
        let (d, n) = (self.desc.hidden, self.desc.embed.streams.max(1));
        let r: Vec<f32> = match self.desc.head.collapse {
            Collapse::None if n == 1 => xt.to_vec(),
            Collapse::None => return err(format!("{n} residual streams and no collapse into the head")),
            Collapse::Mean => {
                // (streams that round `Twice`, transformers' bf16 tensors: the mean of the bf16 streams, summed in fp32, rounded once, as torch's mean of a bf16 tensor)
                let twice =
                    self.desc.blocks.iter().any(|b| matches!(b.residual, Residual::Streams(StreamsP { rounding: crate::runtime::kinds::Rounding::Twice, .. })));
                (0..d).map(|i| (0..n).map(|j| xt[j * d + i]).sum::<f32>() / n as f32).map(|v| if twice { super::bf16(v) } else { v }).collect()
            }
            Collapse::Weighted { eps, norm_eps } => {
                let (f, base, sc) = (self.p.vec("out_hc_fn")?, self.p.vec("out_hc_base")?, self.p.vec("out_hc_scale")?[0]);
                let sp = self.streams_p();
                let r = StreamsP { norm_eps, ..sp }.norm_factor(xt);
                let flat: Vec<f32> = xt.iter().map(|v| v * r).collect();
                let mixes = super::matvec(f, flat.len(), &flat);
                let pre: Vec<f32> = (0..n).map(|j| super::sigmoid(mixes[j] * sc + base[j]) + eps).collect();
                super::collapse(&sp, &pre, xt, d)
            }
            Collapse::DepthMix => {
                let dp = depth.ok_or_else(|| Error("the depth mix's collapse without its sources".into()))?;
                dp.mix(self.p.vec("out_mix_norm")?, self.p.vec("out_mix_proj")?, self.desc.final_norm.eps)
            }
            Collapse::LastPre if pre.len() == n => super::collapse(&self.streams_p(), pre, xt, d),
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
    use crate::runtime::kinds::{Act, Block, Body, Embed, Ffn, Head, Mlp, Norm, Rounding, TableAt};
    use crate::runtime::reference::{ffn, hc_weights, matvec_w, testing::fill};

    const D: usize = 8;
    const N: usize = 4;

    fn streams(lagged: bool) -> StreamsP {
        StreamsP { streams: N, sinkhorn_iters: 20, eps: 1e-6, norm_eps: 1e-6, rounding: Rounding::F32, lagged, eps_outside: false }
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
            embed: Embed { scale: 1.0, norm: None, streams: N, per_layer: None, table: TableAt::Gpu },
            blocks: layers.iter().map(|l| l.block.clone()).collect(),
            final_norm: Norm::rms(1e-6),
            head: Head { tied: false, softcap: None, scale: 1.0, collapse, rounding: Rounding::F32 },
            mtp: Vec::new(),
            vision: None,
            rope: None,
        };
        RefModel {
            desc,
            embed: Held::new(vec![16, D], fill("lag.embed", 16 * D, -1.0, 1.0)),
            final_norm: fill("lag.fn", D, 0.5, 1.5),
            head: Some(Held::new(vec![16, D], fill("lag.head", 16 * D, -0.5, 0.5))),
            layers,
            p: Params::new(),
            record: None,
        }
    }

    /// Streams that round `Twice` (GLM-5.3-Flash, as transformers' bf16 modeling rounds): the collapse of the bf16 streams in fp32 rounded once; the
    /// update bf16(bf16(bf16(post) bf16(out)) + bf16(sum of bf16(comb) x)), the streams staying bf16; the final mean rounded once before the norm. By
    /// hand against the block and the model; in fp32 (`F32`) none of it rounds.
    #[test]
    fn streams_that_round_twice_round_as_transformers_in_bf16() {
        use crate::runtime::reference::bf16;
        let s = StreamsP { rounding: Rounding::Twice, ..streams(false) };
        let l = block(0, s);
        let x: Vec<f32> = fill("twice.x", N * D, -2.0, 2.0).into_iter().map(bf16).collect();
        let o = fill("twice.o", D, -1.0, 1.0);
        let (pre, post, comb) = hc_weights(&s, &l.p, &x).unwrap();
        let (input, hc) = l.pre(&x).unwrap();
        let want: Vec<f32> = (0..D).map(|i| bf16((0..N).map(|j| pre[j] * x[j * D + i]).sum::<f32>())).collect();
        assert_eq!(input, want, "the collapse rounded once");
        let mut got = x.clone();
        l.post(&mut got, &o, hc).unwrap();
        for k in 0..N {
            for i in 0..D {
                let m: f32 = (0..N).map(|j| bf16(comb[j][k]) * x[j * D + i]).sum();
                assert_eq!(got[k * D + i], bf16(bf16(bf16(post[k]) * bf16(o[i])) + bf16(m)), "stream {k}, column {i}");
            }
        }
        assert!(got.iter().all(|&v| v == bf16(v)), "the streams stay bf16");
        let f = block(0, streams(false));
        let (fi, fh) = f.pre(&x).unwrap();
        let mut fx = x.clone();
        f.post(&mut fx, &o, fh).unwrap();
        assert!(fi != input && fx != got, "in fp32 nothing rounds");
        // (the final mean, rounded once before the norm)
        let m = model(s, Collapse::Mean);
        let mean: Vec<f32> = (0..D).map(|i| bf16((0..N).map(|j| x[j * D + i]).sum::<f32>() / N as f32)).collect();
        assert_eq!(m.final_h_with(&x, None, &[]).unwrap(), norm(&m.desc.final_norm, &mean, Some(&m.final_norm), None));
    }

    /// Streams that round `Once` (DeepSeek-V4's and V4.1's release: fp32 mixing of bf16 streams, `.to(x.dtype)` and `.type_as(x)`): the collapse summed in fp32 and
    /// rounded once, each stream's update post o + sum of comb x in fp32 rounded once, the streams staying bf16; lagged mHC's collapse and the head's last one
    /// (`Collapse::LastPre`) so too. By hand against the block and the model.
    #[test]
    fn streams_that_round_once_round_as_the_release() {
        use crate::runtime::reference::{bf16, collapse};
        let s = StreamsP { rounding: Rounding::Once, ..streams(false) };
        let l = block(0, s);
        let x: Vec<f32> = fill("once.x", N * D, -2.0, 2.0).into_iter().map(bf16).collect();
        let o = fill("once.o", D, -1.0, 1.0);
        let (pre, post, comb) = hc_weights(&s, &l.p, &x).unwrap();
        let (input, hc) = l.pre(&x).unwrap();
        let want: Vec<f32> = (0..D).map(|i| bf16((0..N).map(|j| pre[j] * x[j * D + i]).sum::<f32>())).collect();
        assert_eq!((input, collapse(&s, &pre, &x, D)), (want.clone(), want), "the collapse rounded once");
        let mut got = x.clone();
        l.post(&mut got, &o, hc).unwrap();
        for k in 0..N {
            for i in 0..D {
                assert_eq!(got[k * D + i], bf16(post[k] * o[i] + (0..N).map(|j| comb[j][k] * x[j * D + i]).sum::<f32>()), "stream {k}, column {i}");
            }
        }
        // (lagged: the head's collapse by the last weights, rounded once before the final norm)
        let m = model(StreamsP { lagged: true, ..s }, Collapse::LastPre);
        let w = [0.4f32, 0.3, 0.2, 0.1];
        let folded: Vec<f32> = (0..D).map(|i| bf16((0..N).map(|j| w[j] * x[j * D + i]).sum::<f32>())).collect();
        assert_eq!(m.final_h_with(&x, None, &w).unwrap(), norm(&m.desc.final_norm, &folded, Some(&m.final_norm), None));
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
            let mut x: Vec<f32> = (0..N).flat_map(|_| m.embed_row(id as usize)).collect();
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
            assert_eq!(got[t], matvec_w(m.head_w(), D, &n(&weigh(&prev, &x), Some(&m.final_norm))), "token {t}");
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

    /// The streams' norm with its epsilon outside the root (`StreamsP::eps_outside`): the mixing weights are of the streams times 1 / (rms + eps), by hand
    /// from the definition, in a block and in the hyper-head (which norms as the model's streams do); the other form is 1 / sqrt(mean + eps), another
    /// number wherever the epsilon is not nothing beside the streams, and the same one as the epsilon goes to 0.
    #[test]
    fn the_streams_norm_has_its_epsilon_where_the_kind_says() {
        use crate::runtime::reference::{matvec, sigmoid};
        let eps = 0.25f32;
        let inside = StreamsP { norm_eps: eps, ..streams(false) };
        let outside = StreamsP { eps_outside: true, ..inside };
        let x = fill("outside.x", N * D, -2.0, 2.0);
        let mean = x.iter().map(|v| v * v).sum::<f32>() / (N * D) as f32;
        assert_eq!((inside.norm_factor(&x), outside.norm_factor(&x)), (1.0 / (mean + eps).sqrt(), 1.0 / (mean.sqrt() + eps)));
        // (a block's weights: the input's by hand, pre_j = sigmoid((hc_fn x / (rms + eps))_j scale_0 + base_j) + hc_eps)
        let l = block(0, outside);
        let (pre, post, comb) = hc_weights(&outside, &l.p, &x).unwrap();
        let r = 1.0 / (mean.sqrt() + eps);
        let w = matvec(l.p.vec("hc_fn").unwrap(), N * D, &x.iter().map(|v| v * r).collect::<Vec<_>>());
        let base = l.p.vec("hc_base").unwrap();
        for j in 0..N {
            assert_eq!(pre[j], sigmoid(w[j] * 0.7 + base[j]) + 1e-6, "pre {j}");
            assert_eq!(post[j], 2.0 * sigmoid(w[N + j] * 0.9 + base[N + j]), "post {j}");
        }
        let (pre_in, post_in, comb_in) = hc_weights(&inside, &l.p, &x).unwrap();
        assert!(pre != pre_in && post != post_in && comb != comb_in, "the two forms are two functions");
        // (the same function as the epsilon goes: at 0 the two are the same bits or a rounding apart)
        let (a, b) = (StreamsP { norm_eps: 0.0, ..inside }.norm_factor(&x), StreamsP { norm_eps: 0.0, ..outside }.norm_factor(&x));
        assert!((a - b).abs() <= a * 2.0 * f32::EPSILON, "{a} {b}");
        // (the hyper-head: the model's streams' form)
        for s in [inside, outside] {
            let mut m = model(s, Collapse::Weighted { eps: 1e-6, norm_eps: eps });
            m.p.put("out_hc_fn", &[N, N * D], fill("outside.fn", N * N * D, -0.5, 0.5));
            m.p.put("out_hc_base", &[N], fill("outside.base", N, -0.5, 0.5));
            m.p.put("out_hc_scale", &[1], vec![0.8]);
            let r = s.norm_factor(&x);
            let mixes = matvec(m.p.vec("out_hc_fn").unwrap(), N * D, &x.iter().map(|v| v * r).collect::<Vec<_>>());
            let w: Vec<f32> = (0..N).map(|j| sigmoid(mixes[j] * 0.8 + m.p.vec("out_hc_base").unwrap()[j]) + 1e-6).collect();
            let folded: Vec<f32> = (0..D).map(|i| (0..N).map(|j| w[j] * x[j * D + i]).sum::<f32>()).collect();
            assert_eq!(m.final_h(&x, None).unwrap(), norm(&m.desc.final_norm, &folded, Some(&m.final_norm), None), "{s:?}");
        }
        // (said in a description's text only where it is set)
        assert!(!format!("{inside:?}").contains("eps_outside") && format!("{outside:?}").ends_with("eps_outside: true }"));
    }

    /// A matrix the checkpoint stores transposed loads turned, so that a product by the role is the stored matrix's by columns; three stored scalars load as
    /// the one vector of their role, in the order named; a tensor that is no matrix is refused where it is to be turned.
    #[test]
    fn a_transposed_matrix_and_scalars_apart_load_as_their_roles() {
        let stored = fill("turn.w", 5 * 3, -1.0, 1.0);
        let src = |n: &str| match n {
            "w" => Some((vec![5, 3], stored.clone())),
            "a" => Some((vec![1], vec![0.25])),
            "b" => Some((vec![1], vec![0.5])),
            "c" => Some((vec![1], vec![0.75])),
            "v" => Some((vec![7], vec![0.0; 7])),
            _ => None,
        };
        let (shape, w) = load(&src, "hc_fn", &Source::Transposed("w".into())).unwrap();
        assert_eq!(shape, [3, 5]);
        for i in 0..3 {
            for j in 0..5 {
                assert_eq!(w[i * 5 + j], stored[j * 3 + i], "row {i}, column {j}");
            }
        }
        // (the role's product of x is the stored matrix's columns weighed by x: x^T W_stored)
        let x = fill("turn.x", 5, -1.0, 1.0);
        let by_columns: Vec<f32> = (0..3).map(|i| (0..5).map(|j| stored[j * 3 + i] as f64 * x[j] as f64).sum::<f64>() as f32).collect();
        assert_eq!(crate::runtime::reference::matvec(&w, 5, &x), by_columns);
        let names = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(load(&src, "hc_scale", &Source::Rows(names)).unwrap(), (vec![3], vec![0.25, 0.5, 0.75]));
        assert!(load(&src, "hc_fn", &Source::Transposed("v".into())).unwrap_err().msg.contains("stored transposed"));
        assert_eq!(Source::Transposed("w".into()).names(), ["w"]);
    }

    /// The head's rounding points (`Head::rounding`): none in f32; the product's output alone; or a bf16 tensor between operations, as transformers
    /// soft-caps a bf16 model's logits (`/ cap`, `tanh`, `* cap`, each rounded), which the engine's g4_softcap does too. By hand.
    #[test]
    fn the_head_rounds_where_it_says() {
        use crate::runtime::reference::bf16;
        let h = |rounding, scale| Head { tied: true, softcap: Some(30.0), scale, collapse: Collapse::None, rounding };
        let z = 23.456_79f32;
        assert_eq!(logit(&h(Rounding::F32, 1.0), z), 30.0 * (z / 30.0).ltanh());
        assert_eq!(logit(&h(Rounding::Once, 1.0), z), 30.0 * (bf16(z) / 30.0).ltanh());
        let release = bf16(bf16(bf16(bf16(z) / 30.0).ltanh()) * 30.0);
        assert_eq!(logit(&h(Rounding::Twice, 1.0), z), release);
        assert_ne!(release, 30.0 * (z / 30.0).ltanh());
        // (the scale's product rounded too, before the cap)
        assert_eq!(logit(&h(Rounding::Twice, 0.5), z), bf16(bf16(bf16(bf16(bf16(z) * 0.5) / 30.0).ltanh()) * 30.0));
        // (no cap: the product, rounded where the head rounds)
        let bare = Head { softcap: None, ..h(Rounding::Twice, 1.0) };
        assert_eq!((logit(&bare, z), logit(&Head { rounding: Rounding::F32, ..bare }, z)), (bf16(z), z));
    }
}
