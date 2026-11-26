//! A multi-token-prediction draft head (`kinds::MtpHead`), DeepSeek-V3's form that GLM-5.3, DeepSeek-V4 and Qwen3.8 share: for the
//! model's final normed hidden state `h` at position j and the token `t` after it, x = eh_proj([norm_embed(embedding(t)),
//! norm_hidden(h)]) (the embedding first), then the head's blocks on a plain residual stream with a cache of their own, then
//! final_norm; the model's LM head turns the result into the draft, and the result is the next draft's `h`.
//!
//! Roles: `eh_proj` [hidden, 2 hidden], `norm_embed`, `norm_hidden`, `final_norm` [hidden]; each block's roles as a block of the model.
//!
//! The head of the model's streams (`MtpInput::Streams`, Qwen3.8-Flash-Next's): for the model's last streams S (n of hidden, before its collapse) at
//! position j and the token t after it, each stream x_s = `fc_hidden` (norm_hidden(S) as one vector of n hidden)_s + `fc_embedding`
//! norm_embed(embedding(t)); the head's blocks on those streams by the model's residual rule, with caches of their own; the streams after them are
//! the next draft's S; the head's input to the model's head is its own final mixer of them (`gated_mix` with `final_norm` on each stream,
//! `out_mix_down`, `out_mix_up`). Roles `norm_embed` [hidden], `norm_hidden` [n hidden], `fc_embedding`, `fc_hidden` [hidden, hidden],
//! `final_norm` [n hidden], `out_mix_down` [rank, n hidden], `out_mix_up` [n hidden, rank].

use super::{comp, dot, gated_mix, hc_weights, matvec, norm::norm, sigmoid, Params, RefLayer};
use crate::ffi::{err, Result};
use crate::runtime::describe::Described;
use crate::runtime::kinds::{Body, Collapse, Mixer, MtpHead, MtpInput, Residual};

pub struct RefMtp {
    pub head: MtpHead,
    pub hidden: usize,
    pub p: Params,
    pub blocks: Vec<RefLayer>,
}

impl RefMtp {
    /// `blocks`: the head's blocks with their weights, in order.
    pub fn new(head: MtpHead, hidden: usize, p: Params, blocks: Vec<RefLayer>) -> Result<RefMtp> {
        if blocks.len() != head.blocks.len() || blocks.iter().zip(&head.blocks).any(|(l, b)| l.block != *b) {
            return err("an MTP head's layers are not its blocks");
        }
        Ok(RefMtp { head, hidden, p, blocks })
    }

    /// Draft head `i` of the description `d` with its weights from `src` (its blocks' roles and its own, `Described::mtp`).
    pub fn load(d: &Described, i: usize, src: super::model::Tensors) -> Result<RefMtp> {
        let (Some(head), Some((blocks, own))) = (d.desc.mtp.get(i), d.mtp.get(i)) else {
            return err(format!("no draft head {i} named in the description ({} heads, {} named)", d.desc.mtp.len(), d.mtp.len()));
        };
        let params = |roles: &[(String, crate::runtime::describe::Source)]| -> Result<Params> {
            let mut p = Params::new();
            for (role, s) in roles {
                let (shape, v) = super::model::load(src, role, s)?;
                p.put(role, &shape, v);
            }
            Ok(p)
        };
        let layers = head.blocks.iter().zip(blocks).map(|(b, r)| Ok(RefLayer::new(b.clone(), d.desc.hidden, params(r)?))).collect::<Result<Vec<_>>>()?;
        RefMtp::new(head.clone(), d.desc.hidden, params(own)?, layers)
    }

    /// One row of a head of the model's streams (`MtpInput::Streams`): the next token `token` and its embedding row, the streams the row continues
    /// ([n hidden]: the model's last streams at the position, or the head's own after its draft before), at `pos` of sequence `seq`. The head's streams
    /// after its blocks (the next draft's input) and its input to the model's head (its final mixer of them). A row at a position the head's caches
    /// hold already takes its place and the later ones' (a draft's rows are run again on the model's own streams once the model has checked them).
    pub fn step_streams(&mut self, seq: u32, pos: u32, token: u32, embedding: &[f32], streams: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.hidden;
        let n = streams.len() / d.max(1);
        if self.head.input != MtpInput::Streams || embedding.len() != d || n == 0 || n * d != streams.len() {
            return err(format!(
                "a draft head of {:?} on an embedding of {} and streams of {} values for width {d}",
                self.head.input,
                embedding.len(),
                streams.len()
            ));
        }
        let o = self.p.prod("fc_embedding", &norm(&self.head.norm_embed, embedding, self.p.opt("norm_embed"), None))?;
        let hn = norm(&self.head.norm_hidden, streams, self.p.opt("norm_hidden"), None);
        let mut x = Vec::with_capacity(n * d);
        for s in hn.chunks(d) {
            x.extend(self.p.prod("fc_hidden", s)?.iter().zip(&o).map(|(a, b)| a + b));
        }
        for b in &mut self.blocks {
            b.forget_from(seq, pos);
            b.forward(&[pos], &[seq], &[token], &mut x)?;
        }
        let h = gated_mix(n, &self.head.final_norm, self.p.vec("final_norm")?, self.p.vec("out_mix_down")?, self.p.vec("out_mix_up")?, None, &x)?.0;
        Ok((x, h))
    }

    /// One row: the next token `token` and its embedding row, the model's hidden state before it (its final normed one, or the head's output after
    /// its draft before), at `pos` of sequence `seq`; the head's normed hidden state out (the model's head's input and the next draft's state). A row
    /// at a position the head's caches hold already takes its place and the later ones' (as `step_streams`).
    pub fn step(&mut self, seq: u32, pos: u32, token: u32, embedding: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        let d = self.hidden;
        if self.head.input != MtpInput::Joined || embedding.len() != d || hidden.len() != d {
            return err(format!("a draft head of {:?} on an MTP row of {} and {} values for width {d}", self.head.input, embedding.len(), hidden.len()));
        }
        let mut cat = norm(&self.head.norm_embed, embedding, self.p.opt("norm_embed"), None);
        cat.extend(norm(&self.head.norm_hidden, hidden, self.p.opt("norm_hidden"), None));
        let mut x = self.p.prod("eh_proj", &cat)?;
        for b in &mut self.blocks {
            b.forget_from(seq, pos);
            b.forward(&[pos], &[seq], &[token], &mut x)?;
        }
        Ok(norm(&self.head.final_norm, &x, self.p.opt("final_norm"), None))
    }
}

/// A DSpark draft head (`kinds::DSpark`) on the reference: its blocks (window-only compressed attention and the model's feed-forward on its residual
/// streams), the head's roles in `p`: `main_proj`, `main_norm` (its kind the head's `norm_hidden`), `markov_embed`, `markov_head`, `confidence`,
/// `final_norm`, and a hyper-head's `out_hc_fn`, `out_hc_base`, `out_hc_scale` where the head's collapse is `Collapse::Weighted`.
pub struct RefDSpark {
    pub head: MtpHead,
    pub hidden: usize,
    pub p: Params,
    pub blocks: Vec<RefLayer>,
    pub collapse: Collapse,
}

/// One drafted block: each position's base logits (the blocks' and the model's head), its logits with the Markov bias, the chosen token and, with
/// a confidence head, the estimate of its acceptance.
pub struct Drafted {
    pub base: Vec<Vec<f32>>,
    pub logits: Vec<Vec<f32>>,
    pub tokens: Vec<u32>,
    pub confidence: Vec<f32>,
}

impl RefDSpark {
    /// The description `d`'s DSpark drafter (its first draft head, `kinds::DSpark`) with its weights from `src` (its blocks' roles and its own,
    /// `Described::mtp`), its collapse the model's (`Head::collapse`: a hyper-head of the head's own roles, or lagged mHC's last weights).
    pub fn load(d: &Described, src: super::model::Tensors) -> Result<RefDSpark> {
        let (Some(head), Some((blocks, own))) = (d.desc.mtp.first(), d.mtp.first()) else {
            return err(format!("no draft head named in the description ({} heads, {} named)", d.desc.mtp.len(), d.mtp.len()));
        };
        if head.dspark.is_none() {
            return err("a draft head of no DSpark form");
        }
        let params = |roles: &[(String, crate::runtime::describe::Source)]| -> Result<Params> {
            let mut p = Params::new();
            for (role, s) in roles {
                let (shape, v) = super::model::load(src, role, s)?;
                p.put(role, &shape, v);
            }
            Ok(p)
        };
        if blocks.len() != head.blocks.len() {
            return err("a DSpark head's layers are not its blocks");
        }
        let layers = head.blocks.iter().zip(blocks).map(|(b, r)| Ok(RefLayer::new(b.clone(), d.desc.hidden, params(r)?))).collect::<Result<Vec<_>>>()?;
        Ok(RefDSpark { head: head.clone(), hidden: d.desc.hidden, p: params(own)?, blocks: layers, collapse: d.desc.head.collapse })
    }

    /// The block after the context `ctx` (for each position before the anchor, the target's attention inputs at the head's target layers,
    /// concatenated), the anchor `anchor` at position ctx.len(); `embed` and `head` the model's [vocab, hidden]. Greedy: each position's token the
    /// largest of its logits (the first of equals).
    pub fn draft(&mut self, ctx: &[Vec<f32>], anchor: u32, embed: &[f32], head: &[f32]) -> Result<Drafted> {
        self.draft_taking(ctx, anchor, embed, head, &[])
    }

    /// `draft`, the tokens of the block's first positions `take` in the place of its own choices (a check's, which holds an engine to the reference on the
    /// engine's own choices: each later position's Markov bias is then of the engine's token before it); a MoE block's experts as its `take` gives them.
    pub fn draft_taking(&mut self, ctx: &[Vec<f32>], anchor: u32, embed: &[f32], head: &[f32], take: &[u32]) -> Result<Drafted> {
        let Some(ds) = self.head.dspark.clone() else { return err("a draft head without DSpark's form") };
        let (d, p0) = (self.hidden, ctx.len() as u32);
        let main: Vec<Vec<f32>> =
            ctx.iter().map(|c| Ok(norm(&self.head.norm_hidden, &self.p.prod("main_proj", c)?, self.p.opt("main_norm"), None))).collect::<Result<_>>()?;
        let n = match self.blocks.first().map(|b| &b.block.residual) {
            Some(Residual::Streams(s)) => s.streams,
            _ => 1,
        };
        let toks: Vec<u32> = std::iter::once(anchor).chain(std::iter::repeat_n(ds.noise, ds.block - 1)).collect();
        let row = |t: u32| {
            embed.get(t as usize * d..(t as usize + 1) * d).map(|r| r.repeat(n)).ok_or_else(|| crate::ffi::Error(format!("token {t} past the embedding")))
        };
        let mut xs: Vec<Vec<f32>> = toks.iter().map(|&t| row(t)).collect::<Result<_>>()?;
        let mut lag: Vec<Vec<f32>> = vec![(0..n).map(|j| (j == 0) as u32 as f32).collect(); toks.len()];
        for l in &mut self.blocks {
            // (each position's input: mHC's weighted streams, lagged where the block says, or the residual itself)
            let mut post = Vec::new();
            let mut hs = Vec::new();
            for (j, x) in xs.iter().enumerate() {
                let input = match l.block.residual {
                    Residual::Streams(s) => {
                        let (pre, po, comb) = hc_weights(&s, &l.p, x)?;
                        let w = if s.lagged { std::mem::replace(&mut lag[j], pre) } else { pre };
                        post.push(Some((po, comb)));
                        super::collapse(&s, &w, x, d)
                    }
                    _ => {
                        post.push(None);
                        x.clone()
                    }
                };
                hs.push(l.norm_in(&input));
            }
            let outs: Vec<Vec<f32>> = match &l.block.body {
                // (the window's keys of the context from main_x, and every token of the block: not causal inside it)
                Body::Mixer(Mixer::Compressed(c)) if c.ratio == 0 => {
                    let ctx_keys: Vec<Vec<f32>> = main.iter().enumerate().map(|(t, m)| comp::key(c, &l.p, m, t as u32)).collect::<Result<_>>()?;
                    let blk: Vec<Vec<f32>> = hs.iter().enumerate().map(|(j, h)| comp::key(c, &l.p, h, p0 + j as u32)).collect::<Result<_>>()?;
                    (0..hs.len())
                        .map(|j| {
                            let pos = p0 + j as u32;
                            let (_, q) = comp::queries(c, &l.p, &hs[j], pos)?;
                            let from = (pos + 1).saturating_sub(c.window as u32) as usize;
                            let keys: Vec<&[f32]> = ctx_keys[from.min(ctx_keys.len())..].iter().chain(&blk).map(|k| &k[..]).collect();
                            comp::attend(c, &l.p, &q, &keys, pos)
                        })
                        .collect::<Result<_>>()?
                }
                Body::Ffn(_) => hs.iter().enumerate().map(|(j, h)| l.body(0, p0 + j as u32, toks[j], h)).collect::<Result<_>>()?,
                b => return err(format!("a DSpark block of {b:?}: its blocks are the window's attention and the feed-forward")),
            };
            for ((x, o), po) in xs.iter_mut().zip(outs).zip(post) {
                l.post(x, &l.norm_out(o), po)?;
            }
        }
        let (rank, mut prev) = (ds.markov_rank, anchor);
        let (w1, w2) = (self.p.vec("markov_embed")?, self.p.vec("markov_head")?);
        let mut out = Drafted { base: Vec::new(), logits: Vec::new(), tokens: Vec::new(), confidence: Vec::new() };
        // (the head's collapse rounds as its blocks' streams do)
        let sp = self.blocks.iter().find_map(|b| if let Residual::Streams(s) = b.block.residual { Some(s) } else { None });
        let fold = |w: &[f32], x: &[f32]| match &sp {
            Some(s) => super::collapse(s, w, x, d),
            None => (0..d).map(|i| (0..n).map(|k| w[k] * x[k * d + i]).sum()).collect(),
        };
        for (j, x) in xs.iter().enumerate() {
            let h: Vec<f32> = match self.collapse {
                Collapse::LastPre => fold(&lag[j], x),
                Collapse::Weighted { eps, norm_eps } => {
                    let (f, base, sc) = (self.p.vec("out_hc_fn")?, self.p.vec("out_hc_base")?, self.p.vec("out_hc_scale")?[0]);
                    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + norm_eps).sqrt();
                    let mixes = matvec(f, x.len(), &x.iter().map(|v| v * r).collect::<Vec<_>>());
                    let w: Vec<f32> = (0..n).map(|k| sigmoid(mixes[k] * sc + base[k]) + eps).collect();
                    fold(&w, x)
                }
                Collapse::None if n == 1 => x.clone(),
                c => return err(format!("a DSpark head's collapse {c:?}")),
            };
            let base = matvec(head, d, &norm(&self.head.final_norm, &h, self.p.opt("final_norm"), None));
            let e =
                w1.get(prev as usize * rank..(prev as usize + 1) * rank).ok_or_else(|| crate::ffi::Error(format!("token {prev} past the Markov embedding")))?;
            let logits: Vec<f32> = base.iter().zip(matvec(w2, rank, e)).map(|(a, b)| a + b).collect();
            let tok = take.get(j).copied().unwrap_or_else(|| (0..logits.len()).fold(0, |best, t| if logits[t] > logits[best] { t } else { best }) as u32);
            if ds.confidence {
                let cat: Vec<f32> = h.iter().chain(e).copied().collect();
                out.confidence.push(sigmoid(dot(self.p.vec("confidence")?, &cat)));
            }
            prev = tok;
            out.base.push(base);
            out.logits.push(logits);
            out.tokens.push(tok);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::{Act, Attn, Block, Comp, DSpark, Ffn, Mlp, Norm, Rope, RopeScaling, RopeStyle, Rounding, StreamsP};
    use crate::runtime::reference::testing::fill;

    const D: usize = 16;
    const V: usize = 24;

    /// A DSpark head of one draft layer (the window's attention of 2 heads of 8 over a window of 4, a gated MLP) on 2 lagged mHC streams, drafting
    /// `block` tokens from main_x of 2 target layers; a Markov rank of 4, and a confidence head.
    fn head(block: usize) -> RefDSpark {
        let s = StreamsP { streams: 2, sinkhorn_iters: 20, eps: 1e-6, norm_eps: 1e-6, rounding: Rounding::F32, lagged: true };
        let rope = Rope { theta: 10000.0, dim: 4, freqs: 2, freq_dim: 4, style: RopeStyle::Interleaved, scaling: RopeScaling::None, mrope: None };
        let c = Comp {
            heads: 2,
            head_dim: 8,
            q_lora: 12,
            o_groups: 2,
            o_lora: 4,
            window: 4,
            ratio: 0,
            overlap: false,
            rope,
            indexer: None,
            norm_eps: 1e-6,
            ape: false,
            q_head_norm: false,
            share: None,
        };
        let mlp = Mlp { act: Act::Silu, gated: true, bias: false, clamp: Some(10.0) };
        let blk = |body: Body| Block { norm_in: Norm::rms(1e-6), body, norm_out: None, residual: Residual::Streams(s) };
        let blocks = vec![blk(Body::Mixer(Mixer::Compressed(c))), blk(Body::Ffn(Ffn::Mlp(mlp)))];
        let mut layers = Vec::new();
        for (k, b) in blocks.iter().enumerate() {
            let mut p = Params::new();
            let mut put = |role: &str, shape: &[usize], lo: f32, hi: f32| p.put(role, shape, fill(&format!("ds{k}.{role}"), shape.iter().product(), lo, hi));
            put("norm_in", &[D], 0.5, 1.5);
            put("hc_fn", &[8, 2 * D], -0.5, 0.5);
            put("hc_base", &[8], -0.5, 0.5);
            put("hc_scale", &[3], 0.5, 1.5);
            if k == 0 {
                for (role, shape) in [("q_a", [12, D]), ("q_b", [16, 12]), ("kv", [8, D]), ("o_a", [8, 8]), ("o_b", [D, 8])] {
                    put(role, &shape, -0.4, 0.4);
                }
                put("q_a_norm", &[12], 0.5, 1.5);
                put("kv_norm", &[8], 0.5, 1.5);
                put("sinks", &[2], -0.5, 0.5);
            } else {
                put("gate", &[6, D], -0.4, 0.4);
                put("up", &[6, D], -0.4, 0.4);
                put("down", &[D, 6], -0.4, 0.4);
            }
            layers.push(RefLayer::new(b.clone(), D, p));
        }
        let mut p = Params::new();
        for (role, shape) in [("main_proj", [D, 2 * D]), ("markov_embed", [V, 4]), ("markov_head", [V, 4]), ("confidence", [1, D + 4])] {
            p.put(role, &shape, fill(&format!("ds.{role}"), shape[0] * shape[1], -0.5, 0.5));
        }
        p.put("main_norm", &[D], fill("ds.main_norm", D, 0.5, 1.5));
        p.put("final_norm", &[D], fill("ds.final_norm", D, 0.5, 1.5));
        let dspark = DSpark { block, noise: 23, targets: vec![5, 6], markov_rank: 4, confidence: true };
        let h = MtpHead {
            blocks,
            norm_embed: Norm::rms(1e-6),
            norm_hidden: Norm::rms(1e-6),
            final_norm: Norm::rms(1e-6),
            dspark: Some(dspark),
            input: MtpInput::Joined,
        };
        RefDSpark { head: h, hidden: D, p, blocks: layers, collapse: Collapse::LastPre }
    }

    /// A head of `n` streams of `D` (`MtpInput::Streams`) of `blocks` (each a gated-residual attention of 2 heads of 8 over 1 key head, no rotary),
    /// its own front and final mixer of rank 3.
    fn streams_head(n: usize, blocks: usize) -> RefMtp {
        use crate::runtime::kinds::{AttnRounding, GatedStreamsP};
        let w = n * D;
        let a = Attn {
            heads: 2,
            kv_heads: 1,
            head_dim: 8,
            v_head_dim: 8,
            window: None,
            sinks: false,
            rope: None,
            q_norm: None,
            k_norm: None,
            v_norm: None,
            gate: None,
            softcap: None,
            bias: false,
            scale: 8f32.powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::F32,
            kv_from: None,
        };
        let block = Block {
            norm_in: Norm::rms_plus_one(1e-6),
            body: Body::Mixer(Mixer::Attention(a)),
            norm_out: None,
            residual: Residual::GatedStreams(GatedStreamsP { streams: n, lowrank: 3 }),
        };
        let layers: Vec<RefLayer> = (0..blocks)
            .map(|k| {
                let mut p = Params::new();
                for (role, shape) in
                    [("q", [16, D]), ("k", [8, D]), ("v", [8, D]), ("o", [D, 16]), ("mix_down", [3, w]), ("mix_up", [w, 3]), ("inject", [n, w])]
                {
                    p.put(role, &shape, fill(&format!("st{k}.{role}"), shape[0] * shape[1], -0.4, 0.4));
                }
                p.put("norm_in", &[w], fill(&format!("st{k}.norm_in"), w, -0.5, 0.5));
                RefLayer::new(block.clone(), D, p)
            })
            .collect();
        let mut p = Params::new();
        for (role, shape) in [("fc_embedding", [D, D]), ("fc_hidden", [D, D]), ("out_mix_down", [3, w]), ("out_mix_up", [w, 3])] {
            p.put(role, &shape, fill(&format!("st.{role}"), shape[0] * shape[1], -0.4, 0.4));
        }
        for (role, k) in [("norm_embed", D), ("norm_hidden", w), ("final_norm", w)] {
            p.put(role, &[k], fill(&format!("st.{role}"), k, -0.5, 0.5));
        }
        let pn = Norm::rms_plus_one(1e-6);
        let head = MtpHead { blocks: vec![block; blocks], norm_embed: pn, norm_hidden: pn, final_norm: pn, dspark: None, input: MtpInput::Streams };
        RefMtp::new(head, D, p, layers).unwrap()
    }

    /// The head of the model's streams by its parts: each stream fc_hidden of its part of the streams normed as one vector, plus fc_embedding of the
    /// normed embedding (with no blocks, that is the head's streams), and its input to the model's head its own final mixer of them.
    #[test]
    fn a_streams_head_projects_each_stream_and_adds_the_embedding() {
        let n = 3;
        let (e, st) = (fill("st.e", D, -1.0, 1.0), fill("st.s", n * D, -1.0, 1.0));
        let mut h = streams_head(n, 0);
        let (x, y) = h.step_streams(0, 0, 5, &e, &st).unwrap();
        let p = &h.p;
        let pn = Norm::rms_plus_one(1e-6);
        let o = matvec(p.vec("fc_embedding").unwrap(), D, &norm(&pn, &e, p.opt("norm_embed"), None));
        let hn = norm(&pn, &st, p.opt("norm_hidden"), None);
        let want: Vec<f32> =
            (0..n).flat_map(|s| matvec(p.vec("fc_hidden").unwrap(), D, &hn[s * D..][..D]).into_iter().zip(o.clone()).map(|(a, b)| a + b)).collect();
        assert_eq!(x, want);
        let mixed = gated_mix(n, &pn, p.vec("final_norm").unwrap(), p.vec("out_mix_down").unwrap(), p.vec("out_mix_up").unwrap(), None, &x).unwrap().0;
        assert_eq!((y.len(), y), (D, mixed));
        // (the hidden norm is over all the streams at once: scaling one stream moves the others' share)
        let mut st2 = st.clone();
        st2[..D].iter_mut().for_each(|v| *v *= 3.0);
        assert_ne!(h.step_streams(0, 1, 5, &e, &st2).unwrap().0[D..], x[D..]);
    }

    /// A row run at a position the head's cache holds takes its place and drops the later ones: after rows at 0, 1 and 2, the row at 1 again (other
    /// streams) and then 2 give what a fresh head gives for 0, the new 1 and 2.
    #[test]
    fn a_row_run_again_takes_its_place_in_the_heads_cache() {
        let n = 2;
        let e: Vec<Vec<f32>> = (0..4).map(|i| fill(&format!("st.e{i}"), D, -1.0, 1.0)).collect();
        let st: Vec<Vec<f32>> = (0..4).map(|i| fill(&format!("st.s{i}"), n * D, -1.0, 1.0)).collect();
        let mut a = streams_head(n, 1);
        for i in 0..3 {
            a.step_streams(0, i, 1, &e[i as usize], &st[i as usize]).unwrap();
        }
        a.step_streams(0, 1, 1, &e[3], &st[3]).unwrap();
        let again = a.step_streams(0, 2, 1, &e[2], &st[2]).unwrap();
        let mut b = streams_head(n, 1);
        b.step_streams(0, 0, 1, &e[0], &st[0]).unwrap();
        b.step_streams(0, 1, 1, &e[3], &st[3]).unwrap();
        assert_eq!(again, b.step_streams(0, 2, 1, &e[2], &st[2]).unwrap());
        // (and the row at 1 matters to the row at 2)
        let mut c = streams_head(n, 1);
        for i in 0..3 {
            c.step_streams(0, i, 1, &e[i as usize], &st[i as usize]).unwrap();
        }
        assert_ne!(again, c.step_streams(0, 2, 1, &e[2], &st[2]).unwrap());
    }

    fn ctx(t: usize) -> Vec<Vec<f32>> {
        (0..t).map(|i| fill(&format!("ds.ctx{i}"), 2 * D, -1.0, 1.0)).collect()
    }

    /// DSpark's block by its parts: the anchor and the noise tokens in one pass, every position a prediction; the Markov stage left to right (each
    /// position's logits its base plus markov_head of the Markov embedding of the token before it, the anchor first), the confidence sigmoid of
    /// [hidden, Markov embedding]; the attention not causal inside the block (the anchor's logits change with the block's length) and reading the
    /// context in its window alone (main_x of a position out of the window changes nothing, one in it does).
    #[test]
    fn dspark_drafts_a_block_in_one_pass() {
        let (embed, out) = (fill("ds.embed", V * D, -1.0, 1.0), fill("ds.head", V * D, -0.5, 0.5));
        let c = ctx(8);
        let d = head(5).draft(&c, 3, &embed, &out).unwrap();
        assert_eq!((d.tokens.len(), d.confidence.len()), (5, 5));
        assert!(d.confidence.iter().all(|&x| x > 0.0 && x < 1.0));
        let p = &head(5).p;
        let (w1, w2) = (p.vec("markov_embed").unwrap(), p.vec("markov_head").unwrap());
        let mut prev = 3u32;
        for j in 0..5 {
            let e = &w1[prev as usize * 4..][..4];
            let want: Vec<f32> = d.base[j].iter().zip(matvec(w2, 4, e)).map(|(a, b)| a + b).collect();
            assert_eq!(d.logits[j], want, "position {j}");
            prev = d.tokens[j];
        }
        // (the block's later positions are keys of its first)
        let shorter = head(3).draft(&c, 3, &embed, &out).unwrap();
        assert_ne!(shorter.base[0], d.base[0]);
        // (the window of 4 at positions 8 to 12 reads the context from position 5 on)
        let mut far = c.clone();
        far[2] = vec![0.3; 2 * D];
        assert_eq!(head(5).draft(&far, 3, &embed, &out).unwrap().base, d.base);
        let mut near = c.clone();
        near[7] = vec![0.3; 2 * D];
        assert_ne!(head(5).draft(&near, 3, &embed, &out).unwrap().base, d.base);
    }
}
