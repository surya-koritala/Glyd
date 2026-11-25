//! The latent attention layer kind (`runtime::kinds::Mla`) as one operation of the executor (`Op::Mla`), for every family with one: Kimi-K3's gated NoPE latent
//! attention and GLM-5.3-Flash's latent attention over the keys its indexer picks. What a family's layer is, is the operation's parameters (`MlaOp`): its heads and
//! widths (the query's latent, the key-value latent, the shared key part, nope and value heads), where its input products are (one fused product, or q's and the
//! key-value latent's apart), the output gate, the norms' epsilon, and the indexer where it has one; its matrices and small tensors are the part's. One set of
//! kernels runs them all (kernels/mla.cu): the norms into the cache, the absorbed query, the attention over the latents (every position, or the indexer's picks),
//! the values through kv_b_proj, the gate.
//!
//! The cache is the executor's: a ring of keys alone (`Ring::values` false) of the key-value latent and the shared key part a token, and with an indexer a second
//! one of its normed key and gate score a token, a complete pool's key written over its last token's key. Paged (src/dynamic/pages.rs), they take blocks from
//! the pools as every attention's keys and values do, so a latent attention is served with the same page budget, swap and prefix sharing. Where the model has 12-bit
//! tables the latents' ring is 12-bit pages of one stream (src/dynamic/kvpages.rs): the latents are written to the staging rows and from there to their pages, and the
//! attention reads a sequence's pages unpacked to bf16 rows (`kv12_latents`), the same values as the bf16 ring's: the same kernels give the same bits. The indexer's
//! ring stays bf16.
//!
//! The indexer's pick (`IdxPick`) is every indexer's, DeepSeek-V4's (`Op::IdxPick`) too: the score of each complete entry, sum_h w_h relu(q_h . k), the best `top_k`
//! kept in order (kernels/mla.cu `idx_pick`); GLM-5.3's entries are pools of tokens, which the pick gives as their tokens and the open pool's after them. The scores'
//! arithmetic is the family's reference's: GLM-5.3's in f32 (transformers'), DeepSeek-V4's each term rounded to bf16 as the release computes it.
//!
//! Each family's arithmetic is the one its own block had, bit for bit: the same kernels in the same order on the same rows (the kernels moved to kernels/mla.cu with
//! the cache's rows read through a block table, their sums as they were). The q latent's norm is the kind's: KimiRMSNorm (`Rounding::Twice`) by its own kernel and order of sums.

use super::{Dyn, ModelDesc, Op, Part, Slot};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::kern35::Mat;
use crate::launch;
use crate::runtime::kinds;

/// Bytes of the indexer's scores and of the latent outputs a block of attention rows may take.
const SUB_SCORES: usize = 256 << 20;
const SUB_OLAT: usize = 64 << 20;
/// Rows at most that the attention of one call cuts its tokens into eight runs for (a decode step, the rows of a verified draft): `kern_glm::HC_FEW`.
const FEW: u32 = crate::kern::HC_FEW;

/// Where a latent layer's input products are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MlaIn {
    /// One product of the layer's input whose columns are q's latent, the output gate (heads v, where there is one), then the key-value latent and the shared key
    /// part (Kimi-K3's q_a, g and kv_a stacked).
    Fused { w: usize },
    /// q's latent and the key-value latent (and its shared key part) as two products (GLM-5.3's q_a and kv_a).
    Apart { q_a: usize, kv_a: usize },
}

/// Where the indexer's scores are computed as its family's reference computes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IdxScore {
    /// f32 sums of heads of `dim`, each dot product times `scale` (GLM-5.3's, as transformers in float32).
    F32 { dim: usize, scale: f32 },
    /// Heads of 128, each head's dot product, its relu times its weight and their sum rounded to bf16 (DeepSeek-V4's release; kernels/mla.cu `idx_score_bf16`).
    Release,
}

/// Where the indexer's entries' keys are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IdxKeys {
    /// A ring of [key | gate score] a token, a complete entry's key in its last token's key (`Op::Mla`'s indexer).
    Ring(usize),
    /// A ring of the compressed entries' keys, an entry a row (`Ring::entries`: DeepSeek-V4's compressor's cache).
    Entries(usize),
    /// A ring of the pools' keys, a pool a row (`Ring::entries` of 4 tokens: `Op::IdxMean`'s, the mean-pooled indexer's), scored in f32 (`IdxScore::F32`).
    Pools(usize),
}

/// The sparse-attention indexer's pick (`runtime::kinds::Indexer`), every family's: for each query, the score of each complete entry of `ratio` tokens that it can
/// see, sum over `heads` of w_h relu(q_h . k) with w_h = wscale times the head weights' product, the `top_k` best kept in order; `expand`: the picks given as their
/// tokens, and where `tail` the tokens of the incomplete entry after them (GLM-5.3's pools), else as the entries (DeepSeek-V4's).
#[derive(Clone, Debug, PartialEq)]
pub struct IdxPick {
    pub heads: usize,
    pub ratio: usize,
    pub top_k: usize,
    pub expand: bool,
    pub tail: bool,
    pub wscale: f32,
    pub score: IdxScore,
    pub keys: IdxKeys,
    /// The heads' scores weighed by the head weights' product (Iw); else summed alone (the mean-pooled indexer's).
    pub weighted: bool,
    /// A shared compressed attention's candidate blocks (`kinds::CompShare`); None: every visible entry is a candidate.
    pub cand: Option<Cand>,
}

/// Candidate blocks (`kinds::CompShare::publish`, `candidates`): the candidate source's pick publishes, for each query, the best `top` blocks of `block` entries (a
/// block's score its best entry's; the block of the query's newest visible entry kept whatever its score), in order; an index source after it (`Inside`) scores the
/// entries inside the blocks its query was published alone, the rest out of its top-k.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cand {
    Publish { block: usize, top: usize },
    Inside { block: usize, top: usize },
}

/// How an indexer norms each token's key (`runtime::kinds::IndexForm::Gated`'s `key`), at its eps: nn.LayerNorm with its bias (GLM-5.3's, DeepSeek-V3.2's), or an
/// RMS norm with a weight and no bias (the normed value rounded, times the weight, rounded again: Kimi-K3.1's `rmsnorm`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyNorm {
    Layer(f32),
    Rms(f32),
}

/// A latent attention's indexer: its pick, and its small tensors (bf16 as released): the queries' product `q_b` [heads dim, q_rank] (of the normed q latent), the
/// keys' `k` [dim, hidden], their norm's weight [dim] and, with a LayerNorm (`key`), its bias, the head weights' product [heads, hidden]; and where an entry is a
/// pool of tokens (`pick.ratio` above 1) the pooling bias `ape` [ratio, dim] and the gate's product [dim, hidden]. With a key a token (`pick.ratio` 1) there are no
/// pooling tensors: a token's normed key is its entry's. `k_bias` without a LayerNorm, and `ape` and `gate` without pools, are no tensors of the operation and are
/// not read. Its cache is the pick's ring (`IdxKeys::Ring`), 2 dim a token.
#[derive(Clone, Debug, PartialEq)]
pub struct MlaIndexer {
    pub pick: IdxPick,
    pub dim: usize,
    pub q_b: usize,
    pub k: usize,
    pub k_norm: usize,
    pub k_bias: usize,
    pub weights: usize,
    pub ape: usize,
    pub gate: usize,
    pub key: KeyNorm,
}

impl MlaIndexer {
    /// Whether an entry is a pool of tokens (the pooling bias and the gate's product are then its tensors).
    pub fn pooled(&self) -> bool {
        self.pick.ratio > 1
    }

    /// Its small tensors: the four every indexer has, a LayerNorm's bias, a pooled indexer's two.
    pub fn small_count(&self) -> usize {
        4 + matches!(self.key, KeyNorm::Layer(_)) as usize + 2 * self.pooled() as usize
    }

    /// The same indexer with its small tensors from `s`, in the order a loader holds them (q_b, k, the key norm's weight, its bias where it has one, the head
    /// weights, then a pooled indexer's bias and gate), and its cache the ring `ring`.
    pub fn at(&self, s: usize, ring: usize) -> MlaIndexer {
        let w = s + 3 + matches!(self.key, KeyNorm::Layer(_)) as usize;
        MlaIndexer {
            pick: IdxPick { keys: IdxKeys::Ring(ring), ..self.pick.clone() },
            q_b: s,
            k: s + 1,
            k_norm: s + 2,
            k_bias: s + 3,
            weights: w,
            ape: w + 1,
            gate: w + 2,
            ..self.clone()
        }
    }
}

/// `Op::Mla`: the latent attention of H into O. Matrices `input`'s, `q_b` [heads (nope + rope), q_rank] and `o` [hidden, heads v]; small tensors the q latent's
/// norm, the key-value latent's norm and kv_b_proj [heads (nope + v), kv_rank] (the absorbed form multiplies by its rows); the cache ring `ring`, kv_rank + rope a
/// token. Scores q . k times `scale` over the latent and the shared key part (rope channels as they are: none rotated), values the latent's; sigmoid(gate) on each
/// head's output where `gate` (its columns after q's latent in a fused product). `q_round`: the q latent's norm's rounding as the kind gives it; KimiRMSNorm's
/// (`Rounding::Twice`: the normed value rounded, then times the weight) runs its kernel (rms2_rows512: 512 threads, Kimi-K3's order of sums), any other the
/// transformers RMSNorm's (rms2_rows: 256 threads, which rounds the normed value as transformers in bf16 does: GLM-5.3's). Both are the function their engines had, bit
/// for bit.
#[derive(Clone, Debug, PartialEq)]
pub struct MlaOp {
    pub heads: usize,
    pub nope: usize,
    pub rope: usize,
    pub v: usize,
    pub kv_rank: usize,
    pub q_rank: usize,
    pub input: MlaIn,
    pub q_b: usize,
    pub o: usize,
    pub q_norm: usize,
    pub kv_norm: usize,
    pub kv_b: usize,
    pub q_round: kinds::Rounding,
    pub gate: bool,
    pub scale: f32,
    pub eps: f32,
    pub ring: usize,
    pub indexer: Option<MlaIndexer>,
}

impl MlaOp {
    /// The operation of layer kind `k` whose input products are `input`, its other matrices `q_b` and `o`, its small tensors from `s0` (the q latent's norm, the
    /// key-value latent's, kv_b_proj, then an indexer's in `MlaIndexer::at`'s order), its caches the rings `ring` and `ring + 1`. A gate takes a fused product.
    #[allow(clippy::too_many_arguments)]
    pub fn of(k: &kinds::Mla, input: MlaIn, q_b: usize, o: usize, s0: usize, ring: usize, tail: bool) -> Result<MlaOp> {
        let q_rank = k.q_lora.ok_or_else(|| Error("a latent attention whose query has no low rank: not an operation of this executor".into()))?;
        if k.rope.is_some() {
            return err("a latent attention whose shared key part is rotated: not an operation of this executor");
        }
        let gate = match (k.gate, input) {
            (None, _) => false,
            (Some(g), MlaIn::Fused { .. }) if g.act == kinds::Act::Sigmoid && g.norm.is_none() => true,
            (Some(g), _) => return err(format!("a latent attention's gate of {g:?}: the kernels take a sigmoid gate in a fused input product")),
        };
        let indexer = match &k.indexer {
            None => None,
            Some(x) if x.shared => return err("a latent attention that reuses another layer's picks: not an operation of this executor"),
            Some(x) => {
                // (the keys' norm as the kind's form gives it: a LayerNorm with its bias, or an RMS norm with a weight; the mean-pooled form is an attention's)
                let key = match x.form {
                    kinds::IndexForm::Gated { key } if key.weight == kinds::NormWeight::Plain && key.kind == kinds::NormKind::Layer => KeyNorm::Layer(key.eps),
                    kinds::IndexForm::Gated { key } if key.weight == kinds::NormWeight::Plain && key.kind == kinds::NormKind::Rms => KeyNorm::Rms(key.eps),
                    form => {
                        return err(format!(
                            "a latent attention's indexer of {form:?}: the kernels take keys LayerNorm-ed with a bias or RMS-normed with a weight"
                        ))
                    }
                };
                let ix = MlaIndexer {
                    pick: IdxPick {
                        heads: x.heads,
                        ratio: x.block.max(1),
                        top_k: x.top_k / x.block.max(1),
                        expand: true,
                        tail,
                        wscale: (x.heads as f32).powf(-0.5),
                        score: IdxScore::F32 { dim: x.head_dim, scale: (x.head_dim as f32).powf(-0.5) },
                        keys: IdxKeys::Ring(ring + 1),
                        weighted: true,
                        cand: None,
                    },
                    dim: x.head_dim,
                    q_b: 0,
                    k: 0,
                    k_norm: 0,
                    k_bias: 0,
                    weights: 0,
                    ape: 0,
                    gate: 0,
                    key,
                };
                Some(ix.at(s0 + 3, ring + 1))
            }
        };
        let a = MlaOp {
            heads: k.heads,
            nope: k.nope,
            rope: k.rope_dim,
            v: k.v_dim,
            kv_rank: k.kv_lora,
            q_rank,
            input,
            q_b,
            o,
            q_norm: s0,
            kv_norm: s0 + 1,
            kv_b: s0 + 2,
            q_round: k.q_norm.rounding,
            gate,
            scale: k.scale,
            eps: k.q_norm.eps,
            ring,
            indexer,
        };
        a.check()?;
        Ok(a)
    }

    /// The small tensors the operation reads: the three of the latent, and an indexer's (`MlaIndexer::small_count`).
    pub fn small_count(&self) -> usize {
        3 + self.indexer.as_ref().map_or(0, MlaIndexer::small_count)
    }

    /// The matrices it multiplies by, in the order a loader holds them: a fused input product, q_b, o; or q_a, q_b, kv_a, o.
    pub fn mat_count(&self) -> usize {
        match self.input {
            MlaIn::Fused { .. } => 3,
            MlaIn::Apart { .. } => 4,
        }
    }

    /// The rings it keeps: the latents', and an indexer's.
    pub fn ring_count(&self) -> usize {
        1 + self.indexer.is_some() as usize
    }

    /// The same operation with its matrices from `m0` (`mat_count`'s order), its small tensors from `s0` (`MlaOp::of`'s order) and its rings from `ring`: a layer's
    /// place in its description.
    pub fn placed(&self, m0: usize, s0: usize, ring: usize) -> MlaOp {
        let (input, q_b, o) = match self.input {
            MlaIn::Fused { .. } => (MlaIn::Fused { w: m0 }, m0 + 1, m0 + 2),
            MlaIn::Apart { .. } => (MlaIn::Apart { q_a: m0, kv_a: m0 + 2 }, m0 + 1, m0 + 3),
        };
        let indexer = self.indexer.as_ref().map(|x| x.at(s0 + 3, ring + 1));
        MlaOp { input, q_b, o, q_norm: s0, kv_norm: s0 + 1, kv_b: s0 + 2, ring, indexer, ..self.clone() }
    }

    /// Its caches as rings of the description: the latents', and an indexer's [key | gate score] a token; keys alone.
    pub fn rings(&self) -> Vec<super::Ring> {
        std::iter::once(super::Ring::keys(self.kv_rank + self.rope, 1, 0)).chain(self.indexer.as_ref().map(|x| super::Ring::keys(2 * x.dim, 1, 0))).collect()
    }

    /// The shapes the kernels take, or why not.
    pub fn check(&self) -> Result<()> {
        let ks = self.kv_rank + self.rope;
        if self.nope == 0
            || self.nope > 256
            || self.kv_rank == 0
            || self.kv_rank > 512
            || !self.kv_rank.is_multiple_of(64)
            || !ks.is_multiple_of(64)
            || self.q_rank > 8192
        {
            return err(format!(
                "latent attention of heads of {}, latents of {} + {} and a query latent of {}: the kernels take heads up to 256, latents of 64s up to 512 (with the \
                 shared part, in 64s) and a query latent up to 8,192",
                self.nope, self.kv_rank, self.rope, self.q_rank
            ));
        }
        if self.gate && !matches!(self.input, MlaIn::Fused { .. }) {
            return err("a latent attention's gate outside a fused input product");
        }
        if let Some(x) = &self.indexer {
            let (p, kp) = (&x.pick, x.pick.ratio);
            if x.dim == 0 || x.dim > 256 || kp == 0 || kp > 8 || p.top_k == 0 || self.rope > 0 {
                return err(format!(
                    "an indexer of channels {} picking {} groups of {kp} keys (beside a shared key part of {}): the kernels take indexer channels up to 256, groups \
                     up to 8 keys, and no shared key part",
                    x.dim, p.top_k, self.rope
                ));
            }
            // (the indexer's scores' dynamic shared memory must fit the 48 KiB a launch may take without an opt-in)
            if 4 * (p.heads * (x.dim + 1) + p.heads + 8 * x.dim) > 48 << 10 {
                return err(format!("an indexer of {} heads of {} channels: its scores' blocks need more than 48 KiB of shared memory", p.heads, x.dim));
            }
        }
        Ok(())
    }

    /// The tokens a row attends to at most where the indexer picks them: the best pools' and the incomplete last pool's; with none, the context.
    fn tmax(&self, ctx: usize) -> usize {
        match &self.indexer {
            Some(x) => x.pick.top_k * x.pick.ratio + x.pick.ratio - 1,
            None => ctx,
        }
    }

    /// The pools a row's scores hold at most at a context of `ctx`.
    fn pmax(&self, ctx: usize) -> usize {
        self.indexer.as_ref().map_or(0, |x| ctx / x.pick.ratio + 1)
    }

    /// Rows the attention takes at a time: with an indexer as many as the room for its scores and the latent outputs holds (GLM-5.3's engine's blocks of rows), else
    /// a sequence's rows at once (Kimi-K3's dense attention). A row's attention cuts its tokens into runs where a call has a few rows (`FEW`), so these blocks are
    /// what each family's engine took.
    fn rows_at_once(&self, ch: usize, ctx: usize) -> usize {
        if self.indexer.is_none() {
            return ch;
        }
        let sub = ch.min((SUB_OLAT / (self.heads * self.kv_rank * 4)).max(1)).min((SUB_SCORES / (self.pmax(ctx) * 4)).max(1));
        // (a development build's check can ask for more rows at a time, to see a long call's stages whole)
        crate::tune!("GLYD_ENGINE_MLA_ROWS").and_then(|v| v.parse::<usize>().ok()).map_or(sub, |v| v.clamp(1, ch))
    }
}

/// kernels/mla.cu's kernels, each by its type (`crate::kt`).
pub(crate) struct KernMla {
    rms2: crate::kt::rms2_rows,
    rms2_512: crate::kt::rms2_rows512,
    layernorm: crate::kt::layernorm_rows,
    scatter: crate::kt::scatter_rows,
    small: crate::kt::small_linear,
    absorb: crate::kt::mla_absorb,
    pool: crate::kt::idx_pool,
    score: crate::kt::idx_score,
    score_bf16: crate::kt::idx_score_bf16,
    pick: crate::kt::idx_pick,
    blocks: crate::kt::idx_blocks,
    inside: crate::kt::idx_inside,
    attn: crate::kt::mla_attn,
    /// The shared memory a block of `attn` may take here (the GPU's most, opted in to at load).
    smem: u32,
    merge: crate::kt::mla_merge,
    expand: crate::kt::mla_expand_v,
    gate: crate::kt::mla_gate,
    dense_cnt: crate::kt::mla_dense_cnt,
    /// The pages a 12-bit latent ring unpacks for a sequence (`Dyn::kv12_latents`).
    pub(super) pages: crate::kt::ml_pages,
}

impl KernMla {
    pub fn load(dev: &Dev) -> Result<KernMla> {
        // (the attention's tokens a pass, two words a token, are in shared memory: as much as the GPU gives a block)
        let (attn, smem) = (dev.typed::<crate::kt::mla_attn>()?, crate::kern::mla_smem(dev)?);
        attn.allow_smem(smem)?;
        Ok(KernMla {
            rms2: dev.typed()?,
            rms2_512: dev.typed()?,
            layernorm: dev.typed()?,
            scatter: dev.typed()?,
            small: dev.typed()?,
            absorb: dev.typed()?,
            pool: dev.typed()?,
            score: dev.typed()?,
            score_bf16: dev.typed()?,
            pick: dev.typed()?,
            blocks: dev.typed()?,
            inside: dev.typed()?,
            attn,
            smem,
            merge: dev.typed()?,
            expand: dev.typed()?,
            gate: dev.typed()?,
            dense_cnt: dev.typed()?,
            pages: dev.typed()?,
        })
    }

    /// y [m, rows] = x [m, cols] (rows xs apart) W^T for a bf16 W, rows ys apart.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    #[allow(clippy::too_many_arguments)]
    unsafe fn small_linear(&self, s: crate::ffi::Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, m: usize, rows: usize, cols: usize) -> Result<()> {
        launch!(self.small, [m as u32, (rows.max(1) as u32).div_ceil(8), 1], [256, 1, 1], s; x, xs as i32, w, y, ys as i32, rows as i32, cols as i32)
    }
}

/// The latent attentions' scratch, one for every layer, sized for the largest: the input products (`x`, a row as wide as the widest), the normed q latent, the
/// queries, the values' output, the indexer's key or gate product, its queries and head weights (bf16, a chunk's rows); for the rows the attention takes at a time
/// the absorbed queries and the latent outputs (fp32), the indexer's scores, the picked tokens and their counts, and a few rows' partial results.
pub(crate) struct MlaRun {
    x: Buf,
    qr: Buf,
    q: Buf,
    ov: Buf,
    ix: Buf,
    iq: Buf,
    wts: Buf,
    qa: Buf,
    olat: Buf,
    scores: Buf,
    sel: Buf,
    cnt: Buf,
    part: Buf,
}

impl MlaRun {
    /// The scratch of `d`'s latent attentions (`mats` its matrices) for chunks of `ch` rows at a context of `ctx`; None where it has none.
    pub fn new(d: &ModelDesc, mats: &[Mat], ch: usize, ctx: usize) -> Result<Option<MlaRun>> {
        let ops: Vec<&MlaOp> = d.layers.iter().flatten().filter_map(|o| if let Op::Mla(a) = o { Some(a) } else { None }).collect();
        if ops.is_empty() {
            return Ok(None);
        }
        for a in &ops {
            a.check()?;
        }
        let most = |f: &dyn Fn(&MlaOp) -> usize| ops.iter().map(|a| f(a)).max().unwrap_or(0).max(1);
        let sub = most(&|a| a.rows_at_once(ch, ctx));
        let ix = |a: &MlaOp, f: &dyn Fn(&MlaIndexer) -> usize| a.indexer.as_ref().map_or(0, f);
        let x = most(&|a| match a.input {
            MlaIn::Fused { w } => mats[w].whole_rows(),
            MlaIn::Apart { q_a, kv_a } => mats[q_a].whole_rows().max(mats[kv_a].whole_rows()),
        });
        let b = |n: usize| Buf::new(n.max(4));
        Ok(Some(MlaRun {
            x: b(ch * x * 2)?,
            qr: b(ch * most(&|a| a.q_rank) * 2)?,
            q: b(ch * most(&|a| a.heads * (a.nope + a.rope)) * 2)?,
            ov: b(ch * most(&|a| a.heads * a.v) * 2)?,
            ix: b(ch * most(&|a| ix(a, &|x| x.dim)) * 2)?,
            iq: b(ch * most(&|a| ix(a, &|x| x.pick.heads * x.dim)) * 2)?,
            wts: b(ch * most(&|a| ix(a, &|x| x.pick.heads)) * 2)?,
            qa: b(sub * most(&|a| a.heads * (a.kv_rank + a.rope)) * 4)?,
            olat: b(sub * most(&|a| a.heads * a.kv_rank) * 4)?,
            scores: b(sub * most(&|a| a.pmax(ctx)) * 4)?,
            sel: b(sub * most(&|a| if a.indexer.is_some() { a.tmax(ctx) } else { 0 }) * 4)?,
            cnt: b(ch.max(sub) * 4)?,
            part: b(FEW as usize * most(&|a| a.heads * 8 * (a.kv_rank + 2)) * 4)?,
        }))
    }
}

/// Where an indexer's pick reads and writes for `rows` rows: their queries (heads dim a row), head weights (heads a row), the entries' keys (`keys`, a ring's through
/// `tab`, rows `stride` apart), the rows' positions, the scores' room (`emax` a row), the picks (`sk` a row) and their counts.
pub(crate) struct PickAt {
    pub q: u64,
    pub w: u64,
    pub keys: u64,
    pub stride: usize,
    pub tab: u64,
    pub pos: u64,
    pub rows: usize,
    pub scores: u64,
    pub emax: usize,
    pub sel: u64,
    pub sk: usize,
    pub cnt: u64,
    /// The candidate blocks' room (`IdxPick::cand`): the rows' published blocks and their counts, the blocks' scores (`bmax` a row) and their counts.
    pub cand: CandAt,
}

/// Where a pick's candidate blocks are (`IdxPick::cand`; all 0 without them).
#[derive(Clone, Copy, Default)]
pub(crate) struct CandAt {
    pub blocks: u64,
    pub nblocks: u64,
    pub bsc: u64,
    pub bmax: usize,
    pub nent: u64,
}

impl Dyn {
    /// The sequences of a call's n rows: each one's first row, its rows, its keys' and values' room (its own, or the pools) and its slot (its block tables).
    pub(super) fn seq_parts(&self, n: usize) -> Vec<(usize, usize, u64, usize)> {
        if self.parts.is_empty() {
            vec![(0, n, self.kc.ptr, self.seq)]
        } else {
            self.parts.iter().map(|q| (q.row0, q.rows, q.kc, q.slot)).collect()
        }
    }

    /// Ring `ring`'s keys for a sequence whose room is `kc` (`seq_parts`), and its block table in the ring's group for the sequence in `slot` (0: none).
    pub(super) fn ring_at(&self, kc: u64, slot: usize, ring: usize) -> (u64, u64) {
        (kc + self.ring_off[ring].0, self.table_of(slot, ring))
    }

    /// An indexer's pick (`IdxPick`) for `at.rows` rows: the entries' scores in its arithmetic, then the best of each row (kernels/mla.cu `idx_pick`).
    ///
    /// # Safety
    /// `at`'s buffers live and of the sizes its fields give.
    pub(super) unsafe fn idx_pick(&self, x: &IdxPick, at: &PickAt) -> Result<()> {
        let (s, k, n) = (self.dev.stream, &self.km, at.rows as u32);
        match x.score {
            IdxScore::F32 { dim, scale } => {
                let smem = ((x.heads * (dim + 1) + x.heads + 8 * dim) * 4) as u32;
                // (a pool's key in its last token's row of a ring, or a row a pool; the head weights a row's, or the same ones for every row where unweighted)
                let (pstep, poff) = if matches!(x.keys, IdxKeys::Pools(_)) { (1, 0) } else { (x.ratio as i32, x.ratio as i32 - 1) };
                let wsr = if x.weighted { x.heads as i32 } else { 0 };
                launch!(k.score, [(at.emax as u32).div_ceil(8), n, 1], [256, 1, 1], smem smem, s;
                    at.q, (x.heads * dim) as i32, at.w, wsr, at.keys, at.stride as i32, pstep, poff, at.tab, at.pos, at.scores, at.emax as i32, x.heads as i32,
                    dim as i32, x.ratio as i32, x.top_k as i32, scale, x.wscale)?;
            }
            IdxScore::Release => {
                // (the queries and weights, then idx_score_bf16's warps' terms: at most 64 heads)
                let smem = (x.heads * 129 * 4 + 8 * 64 * 4) as u32;
                launch!(k.score_bf16, [n, (at.emax as u32).div_ceil(8), 1], [256, 1, 1], smem smem, s;
                    at.q, at.w, x.wscale, at.keys, at.tab, at.pos, at.scores, x.heads as i32, x.ratio as i32, at.emax as i32)?;
            }
        }
        // (the candidate source: the best blocks of its scores published; an index source after it: its scores outside them out of its candidates)
        let c = &at.cand;
        match x.cand {
            Some(Cand::Publish { block, top }) => {
                launch!(k.blocks, [n, 1, 1], [256, 1, 1], s; at.scores, at.emax as i32, at.pos, x.ratio as i32, block as i32, c.bsc, c.bmax as i32, c.nent)?;
                launch!(k.pick, [n, 1, 1], [1024, 1, 1], s; c.bsc, c.bmax as i32, at.pos, 1i32, top as i32, 1i32, 0i32, c.blocks, top as i32, c.nblocks, c.nent, 0i32)?;
            }
            Some(Cand::Inside { block, top }) => {
                launch!(k.inside, [n, 1, 1], [256, 1, 1], s; at.scores, at.emax as i32, at.pos, x.ratio as i32, block as i32, c.blocks, top as i32, c.nblocks)?;
            }
            None => {}
        }
        let masked = matches!(x.cand, Some(Cand::Inside { .. })) as i32;
        let expand = if x.expand { x.ratio } else { 1 } as i32;
        launch!(k.pick, [n, 1, 1], [1024, 1, 1], s; at.scores, at.emax as i32, at.pos, x.ratio as i32, x.top_k as i32, expand, x.tail as i32, at.sel, at.sk as i32, at.cnt, 0u64, masked)
    }

    /// `Op::Mla` (of part `p`) on n rows at positions `pos`: the input products, the q latent's norm and q's product; each sequence's key-value latents normed into its
    /// cache; with an indexer its queries, head weights, keys and gate scores (into its cache) and the pools the call completes; then for each sequence, the rows
    /// the attention takes at a time, the absorbed queries, the indexer's pick or every position, the attention over the latents and the values; the gate and o.
    pub(super) unsafe fn mla(&self, p: &Part, a: &MlaOp, n: usize, pos: u64) -> Result<()> {
        let r = self.mla.as_ref().ok_or_else(|| Error("a latent attention in a model loaded without its buffers".into()))?;
        let (s, k) = (self.dev.stream, &self.km);
        let (nh, kr, ks, qw, nr) = (a.heads, a.kv_rank, a.kv_rank + a.rope, a.nope + a.rope, n as u32);
        let h = p.slot(Slot::H);
        let seqs = self.seq_parts(n);
        // the q latent (and, fused, the gate and the key-value latent) product, its norm, q
        let (q_at, q_row) = match a.input {
            MlaIn::Fused { w } => (w, p.mats[w].whole_rows()),
            MlaIn::Apart { q_a, .. } => (q_a, p.mats[q_a].whole_rows()),
        };
        self.lin(&p.mats[q_at], h, n, r.x.ptr)?;
        self.mark("latent attention: q latent product")?;
        let (qn, x) = (p.small[a.q_norm].ptr, r.x.ptr);
        if a.q_round == kinds::Rounding::Twice {
            launch!(k.rms2_512, [nr, 1, 1], [512, 1, 1], s; x, q_row as i32, qn, r.qr.ptr, a.q_rank as i32, a.q_rank as i32, a.eps)?;
        } else {
            launch!(k.rms2, [nr, 1, 1], [256, 1, 1], s; x, q_row as i32, qn, r.qr.ptr, a.q_rank as i32, 0u64, 0u64, a.q_rank as i32, a.eps)?;
        }
        self.lin(&p.mats[a.q_b], r.qr.ptr, n, r.q.ptr)?;
        self.mark("latent attention: q")?;
        // the key-value latent normed into the cache, the shared key part after it
        let (kv, kv_row) = match a.input {
            MlaIn::Fused { w } => (x + ((a.q_rank + if a.gate { nh * a.v } else { 0 }) * 2) as u64, p.mats[w].whole_rows()),
            MlaIn::Apart { kv_a, .. } => {
                self.lin(&p.mats[kv_a], h, n, x)?;
                (x, p.mats[kv_a].whole_rows())
            }
        };
        // (12-bit pages: a row of the call to its staging row, `pw` wide, the channels past the latent's zero; then the rows into their pages)
        let staged = self.pages12(p, a.ring).map(|rp| (rp.shape.heads * rp.shape.channels, self.lossless.as_ref().expect("pages").rows_k.ptr));
        if let Some((pw, rows)) = staged.filter(|x| x.0 > ks) {
            crate::dev::memset_async(rows, 0, n * pw * 2, s)?;
        }
        for &(r0, m, kc, slot) in &seqs {
            let (xr, p0, mr) = (kv + (r0 * kv_row * 2) as u64, pos + (r0 * 4) as u64, m as u32);
            let (lat, tab, ys, at) = match staged {
                Some((pw, rows)) => (rows + (r0 * pw * 2) as u64, 0, pw, 0),
                None => {
                    let (lat, tab) = self.ring_at(kc, slot, a.ring);
                    (lat, tab, ks, p0)
                }
            };
            launch!(k.rms2, [mr, 1, 1], [256, 1, 1], s; xr, kv_row as i32, p.small[a.kv_norm].ptr, lat, ys as i32, at, tab, kr as i32, a.eps)?;
            if a.rope > 0 {
                launch!(k.scatter, [mr, 1, 1], [256, 1, 1], s; xr + (kr * 2) as u64, kv_row as i32, lat + (kr * 2) as u64, ys as i32, at, tab, a.rope as i32)?;
            }
        }
        if staged.is_some() {
            self.kv12_write(a.ring, n)?;
        }
        self.mark("latent attention: the latents into the cache")?;
        // the indexer: queries and head weights for the rows, keys and gate scores into its cache, the pools the rows complete keyed
        if let Some(ix) = &a.indexer {
            let (ih, id, kp, hid) = (ix.pick.heads, ix.dim, ix.pick.ratio, p.d.hidden);
            let sm = |i: usize| p.small[i].ptr;
            k.small_linear(s, r.qr.ptr, a.q_rank, sm(ix.q_b), r.iq.ptr, ih * id, n, ih * id, a.q_rank)?;
            k.small_linear(s, h, hid, sm(ix.k), r.ix.ptr, id, n, id, hid)?;
            // (each token's key normed into its cache's row, as the kind's form says: a LayerNorm with its bias, or the weighted RMS norm)
            for &(r0, m, kc, slot) in &seqs {
                let (ring, tab) = self.ring_at(kc, slot, a.ring + 1);
                let (x, at) = (r.ix.ptr + (r0 * id * 2) as u64, pos + (r0 * 4) as u64);
                match ix.key {
                    KeyNorm::Layer(eps) => {
                        launch!(k.layernorm, [m as u32, 1, 1], [128, 1, 1], s; x, id as i32, sm(ix.k_norm), sm(ix.k_bias), ring, (2 * id) as i32, at, tab, id as i32, eps)?
                    }
                    KeyNorm::Rms(eps) => {
                        launch!(k.rms2, [m as u32, 1, 1], [256, 1, 1], s; x, id as i32, sm(ix.k_norm), ring, (2 * id) as i32, at, tab, id as i32, eps)?
                    }
                }
            }
            // (pools: each token's gate scores beside its key, then the keys of the pools these rows complete; a key a token is its entry's as it is)
            if ix.pooled() {
                k.small_linear(s, h, hid, sm(ix.gate), r.ix.ptr, id, n, id, hid)?;
                for &(r0, m, kc, slot) in &seqs {
                    let (ring, tab) = self.ring_at(kc, slot, a.ring + 1);
                    launch!(k.scatter, [m as u32, 1, 1], [256, 1, 1], s; r.ix.ptr + (r0 * id * 2) as u64, id as i32, ring + (id * 2) as u64, (2 * id) as i32, pos + (r0 * 4) as u64, tab, id as i32)?;
                }
            }
            k.small_linear(s, h, hid, sm(ix.weights), r.wts.ptr, ih, n, ih, hid)?;
            if ix.pooled() {
                for &(r0, m, kc, slot) in &seqs {
                    let (ring, tab) = self.ring_at(kc, slot, a.ring + 1);
                    let w2 = (2 * id) as i32;
                    launch!(k.pool, [(m as u32).div_ceil(kp as u32) + 1, 1, 1], [128, 1, 1], s;
                        ring, w2, ring + (id * 2) as u64, w2, sm(ix.ape), ring, w2, kp as i32, kp as i32 - 1, tab, pos + (r0 * 4) as u64, m as i32, id as i32, kp as i32)?;
                }
            }
            self.mark("latent attention: the indexer's products and pools")?;
        }
        // the attention, a block of rows at a time
        let (sub, tmax, pmax) = (a.rows_at_once(self.chunk, self.ctx), a.tmax(self.ctx), a.pmax(self.ctx));
        let wkv = p.small[a.kv_b].ptr;
        // (12-bit pages: each sequence's rows' positions end here at most: its part's, or the selected sequence's from its length or the pass's start; a step's graph
        // being captured, the context)
        let ends: Vec<usize> = match (self.capturing.get(), self.parts.is_empty()) {
            (true, _) => vec![self.ctx; seqs.len()],
            (false, true) => vec![self.start.get().max(self.n) + n],
            (false, false) => self.parts.iter().map(|q| q.start + q.rows).collect(),
        };
        for (&(first, rows, kc, slot), &end) in seqs.iter().zip(&ends) {
            let (lat, tab, ls) = match staged {
                Some((pw, _)) => (self.kv12_latents(a.ring, slot, pos + ((first + rows - 1) * 4) as u64, end.div_ceil(16))?, 0, pw),
                None => {
                    let (lat, tab) = self.ring_at(kc, slot, a.ring);
                    (lat, tab, ks)
                }
            };
            let mut r0 = first;
            while r0 < first + rows {
                let m = sub.min(first + rows - r0);
                let p0 = pos + (r0 * 4) as u64;
                launch!(k.absorb, [nh as u32, (m as u32).div_ceil(8), 1], [256, 1, 1], s;
                    r.q.ptr + (r0 * nh * qw * 2) as u64, (nh * qw) as i32, qw as i32, wkv, r.qa.ptr, m as i32, nh as i32, a.nope as i32, a.v as i32, kr as i32, ks as i32)?;
                let sel = match &a.indexer {
                    Some(ix) => {
                        let (ih, id) = (ix.pick.heads, ix.dim);
                        let (keys, itab) = self.ring_at(kc, slot, a.ring + 1);
                        let at = PickAt {
                            q: r.iq.ptr + (r0 * ih * id * 2) as u64,
                            w: r.wts.ptr + (r0 * ih * 2) as u64,
                            keys,
                            stride: 2 * id,
                            tab: itab,
                            pos: p0,
                            rows: m,
                            scores: r.scores.ptr,
                            emax: pmax,
                            sel: r.sel.ptr,
                            sk: tmax,
                            cnt: r.cnt.ptr,
                            cand: CandAt::default(),
                        };
                        self.idx_pick(&ix.pick, &at)?;
                        r.sel.ptr
                    }
                    None => {
                        launch!(k.dense_cnt, [(m as u32).div_ceil(256), 1, 1], [256, 1, 1], s; p0, r.cnt.ptr, m as i32)?;
                        0
                    }
                };
                self.mla_attn(r, (lat, ls), tab, sel, tmax, a, m)?;
                launch!(k.expand, [nh as u32, (m as u32).div_ceil(4), 1], [256, 1, 1], smem (4 * kr * 4) as u32, s;
                    r.olat.ptr, wkv, r.ov.ptr + (r0 * nh * a.v * 2) as u64, m as i32, nh as i32, a.nope as i32, a.v as i32, kr as i32)?;
                r0 += m;
            }
        }
        self.mark("latent attention: attention")?;
        if a.gate {
            let w = nh * a.v;
            launch!(k.gate, [((n * w) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; r.ov.ptr, x + (a.q_rank * 2) as u64, q_row as i32, w as i32, n as i32)?;
        }
        self.lin(&p.mats[a.o], r.ov.ptr, n, p.slot(Slot::O))?;
        self.mark("latent attention: o product")
    }

    /// olat [m, heads, kv_rank] fp32: each row's and head's softmax attention over the tokens the counts (and picks, `sel`; 0: every position up to the row's) give,
    /// of the latents `lat.0` (rows `lat.1` apart: the ring's, through `tab`, or a 12-bit ring's pages unpacked, position p at row p). A few rows (at most `FEW`) cut
    /// their tokens into 8 runs over blocks and merge them; more rows are parallel enough by themselves.
    unsafe fn mla_attn(&self, r: &MlaRun, lat: (u64, usize), tab: u64, sel: u64, tmax: usize, a: &MlaOp, m: usize) -> Result<()> {
        let (s, k) = (self.dev.stream, &self.km);
        let (nh, kr, ks) = (a.heads as i32, a.kv_rank as i32, (a.kv_rank + a.rope) as i32);
        let splits: i32 = if m as u32 <= FEW { 8 } else { 1 };
        let tmax = tmax as i32;
        // (a block's tokens a tile at a time: as many as its shared memory holds, two words a token besides the query's KS)
        let mut tile = ((tmax + splits - 1) / splits).min(crate::kern::mla_positions(k.smem, ks) as i32).max(1);
        if tile < 64 && tile < (tmax + splits - 1) / splits {
            return err(format!("latent attention with scores over {ks} channels: {} bytes of shared memory a block hold too few tokens a pass", k.smem));
        }
        // (a development build's GLYD_MLA_TILE: smaller tiles, the passes of a long context on a short one)
        if let Some(t) = crate::tune!("GLYD_MLA_TILE").and_then(|v| v.parse::<i32>().ok()).filter(|&t| t > 0) {
            tile = tile.min(t);
        }
        let smem = ((ks + 2 * tile) * 4) as u32;
        launch!(k.attn, [nh as u32, m as u32, splits as u32], [256, 1, 1], smem smem, s;
            r.qa.ptr, lat.0, lat.1 as i32, tab, sel, r.cnt.ptr, r.olat.ptr, r.part.ptr, tmax, nh, kr, ks, a.scale, splits, tile)?;
        if splits > 1 {
            launch!(k.merge, [nh as u32, m as u32, 1], [256, 1, 1], s; r.part.ptr, r.olat.ptr, nh, kr, splits)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::{Indexer, Norm, OutGate};

    fn kind(rope: usize, gate: bool, indexer: bool) -> kinds::Mla {
        kinds::Mla {
            heads: 4,
            q_lora: Some(128),
            kv_lora: 128,
            nope: 64,
            rope_dim: rope,
            v_dim: 64,
            rope: None,
            q_norm: Norm::rms(1e-6),
            kv_norm: Norm::rms(1e-6),
            gate: gate.then_some(OutGate { act: kinds::Act::Sigmoid, from: kinds::GateFrom::Own, norm: None }),
            scale: ((64 + rope) as f32).powf(-0.5),
            absorbed: true,
            indexer: indexer.then_some(Indexer {
                heads: 4,
                head_dim: 32,
                top_k: 8,
                block: 4,
                shared: false,
                form: crate::runtime::kinds::IndexForm::LAYER_KEYS,
            }),
        }
    }

    /// Kimi-K3's and GLM-5.3-Flash's descriptions (their releases' configs) keep their latents, and GLM-5.3's indexer keys, in rings of keys alone that every latent
    /// attention reads through a block table: they are paged as every attention's keys and values are. So are DeepSeek-V4's window and compressed entries.
    #[test]
    fn the_latent_caches_are_paged_rings() {
        let fixture = |f: &str| std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f)).unwrap();
        let h: serde_json::Value = serde_json::from_str(&fixture("headers/Kimi-K3.json")).unwrap();
        let kc = crate::kimi::ConfigKimi::parse(h["config"].as_str().unwrap(), "Kimi-K3", None).unwrap();
        let kimi = super::super::kimi::describe(&kc, &super::super::Stage::whole(kc.layers), &vec![(true, true); kc.layers], false).unwrap();
        let gc = crate::glm::ConfigGlm::parse(&fixture("glm5_next_config.json"), "GLM-5.3-Flash", None).unwrap();
        let glm = super::super::glm::describe(&gc, &super::super::Stage::whole(gc.layers)).unwrap();
        for (d, latents, width, indexer) in [(&kimi, 24, 512 + 64, None), (&glm, 11, 512, Some(2 * 128))] {
            let ops: Vec<&MlaOp> = d.layers.iter().flatten().filter_map(|o| if let Op::Mla(a) = o { Some(a) } else { None }).collect();
            assert_eq!(ops.len(), latents, "{}", d.family);
            assert_eq!(d.rings.len(), latents * (1 + indexer.is_some() as usize), "{}", d.family);
            for a in ops {
                let r = &d.rings[a.ring];
                assert_eq!((r.d, r.nk, r.window, r.values), (width, 1, 0, false), "{}", d.family);
                if let Some(w) = indexer {
                    let r = &d.rings[a.ring + 1];
                    assert_eq!((r.d, r.values, a.indexer.as_ref().map(|x| x.pick.keys)), (w, false, Some(IdxKeys::Ring(a.ring + 1))));
                }
            }
            assert!(super::super::rows::pages_all(d), "{}", d.family);
        }
        // (DeepSeek-V4: its window is a ring of keys alone, its keys its values; its compressed entries, the attention's and the indexer's, rings of an entry a row
        // (`Ring::entries`), so paged; its states are its compressors' open windows alone; its indexer's pick every indexer's, over its entries' ring)
        let v4h: serde_json::Value = serde_json::from_str(&fixture("headers/DeepSeek-V4-Flash-0731.json")).unwrap();
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(v4h["config"].as_str().unwrap()).unwrap()).unwrap();
        let vo = super::super::deepseek_v4::V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 4096 };
        let v4 = super::super::lower_comp::lower(&kinds, vec![1], 1 << 20, &super::super::Stage::whole(43), &vo).unwrap().0;
        assert!(v4.rings.iter().all(|r| !r.values) && super::super::rows::pages_all(&v4));
        let ratios: std::collections::BTreeSet<usize> = v4.rings.iter().map(|r| r.ratio).collect();
        assert_eq!(ratios.into_iter().collect::<Vec<_>>(), vec![1, 4, 128]);
        for op in v4.layers.iter().flatten() {
            match *op {
                Op::Compress { cache, d, ratio, state, .. } => {
                    let r = &v4.rings[cache];
                    assert_eq!((r.d, r.ratio, r.window, r.values), (d, ratio, 0, false));
                    // (the open windows' rows and scores, fp32: twice the ratio's rows of twice the width where they overlap)
                    let coff = 1 + (ratio == 4) as usize;
                    assert_eq!((v4.states[state], v4.states[state + 1]), (coff * ratio * coff * d * 4, coff * ratio * coff * d * 4));
                }
                Op::CompAttn { ring, cache, ratio, .. } => {
                    assert_eq!((v4.rings[ring].ratio, cache.map(|c| v4.rings[c].ratio)), (1, (ratio > 0).then_some(ratio)));
                }
                _ => {}
            }
        }
        let picks: Vec<&IdxPick> = v4.layers.iter().flatten().filter_map(|o| if let Op::IdxPick(x) = o { Some(x) } else { None }).collect();
        assert!(
            !picks.is_empty()
                && picks.iter().all(|x| x.score == IdxScore::Release
                    && !x.expand
                    && matches!(x.keys, IdxKeys::Entries(r) if v4.rings[r].ratio == 4 && v4.rings[r].d == 128))
        );
    }

    /// The scores' factor the kind gives is the one each family's own block took: Kimi-K3's 1 / sqrt(nope + rope) and GLM-5.3's nope^-0.5 at their releases' and
    /// their random models' widths.
    #[test]
    fn the_scores_factor_is_the_old_blocks() {
        for qw in [64usize, 192, 256] {
            assert_eq!((qw as f32).powf(-0.5).to_bits(), (1.0 / (qw as f32).sqrt()).to_bits(), "{qw}");
        }
    }

    /// The kind's parameters land where the kernels read them: a fused product with its gate (Kimi-K3), two products and an indexer (GLM-5.3) whose small tensors
    /// follow the latent's three and whose cache is the next ring; what the kernels do not take is refused.
    #[test]
    fn the_kind_lowers_to_the_operation() {
        let twice = Norm { rounding: kinds::Rounding::Twice, ..Norm::rms(1e-6) };
        let k = MlaOp::of(&kinds::Mla { q_norm: twice, ..kind(64, true, false) }, MlaIn::Fused { w: 5 }, 6, 7, 10, 2, false).unwrap();
        assert_eq!((k.gate, k.q_norm, k.kv_norm, k.kv_b, k.small_count(), k.rope, k.q_rank), (true, 10, 11, 12, 3, 64, 128));
        // (the q latent's norm: KimiRMSNorm's rounding is the kind's, which picks its kernel; GLM-5.3's kind gives its norm in f32)
        assert_eq!(k.q_round, kinds::Rounding::Twice);
        let twice_not = kind(0, false, true).q_norm.rounding;
        let g = MlaOp::of(&kind(0, false, true), MlaIn::Apart { q_a: 1, kv_a: 2 }, 3, 4, 20, 6, true).unwrap();
        let x = g.indexer.as_ref().unwrap();
        assert_eq!((x.q_b, x.gate, x.pick.top_k, x.pick.keys, g.small_count(), g.q_round), (23, 29, 2, IdxKeys::Ring(7), 10, twice_not));
        assert_eq!((g.tmax(1024), g.pmax(1024), g.rows_at_once(64, 1024)), (2 * 4 + 3, 257, 64));
        // (a key a token, RMS-normed: no bias, no pooling tensors; its four small tensors follow the latent's three. LayerNorm-ed: its bias too)
        let per_token = |key: Norm| {
            let ix = Indexer { block: 1, top_k: 8, form: kinds::IndexForm::Gated { key }, ..kind(0, false, true).indexer.unwrap() };
            MlaOp::of(&kinds::Mla { indexer: Some(ix), ..kind(0, false, false) }, MlaIn::Fused { w: 0 }, 1, 2, 20, 6, false).unwrap()
        };
        let r = per_token(Norm::rms(1e-5));
        let x = r.indexer.as_ref().unwrap();
        assert_eq!(
            (x.key, x.pooled(), x.q_b, x.k, x.k_norm, x.weights, r.small_count(), x.pick.ratio, x.pick.top_k),
            (KeyNorm::Rms(1e-5), false, 23, 24, 25, 26, 7, 1, 8)
        );
        let l = per_token(Norm { kind: kinds::NormKind::Layer, ..Norm::rms(1e-5) });
        let x = l.indexer.as_ref().unwrap();
        assert_eq!((x.key, x.k_bias, x.weights, l.small_count()), (KeyNorm::Layer(1e-5), 26, 27, 8));
        // (a pooled indexer placed elsewhere keeps its order; the form of GLM-5.3's is the LayerNorm at 1e-6)
        let moved = g.placed(40, 50, 9);
        let x = moved.indexer.as_ref().unwrap();
        assert_eq!((x.key, x.q_b, x.k_bias, x.weights, x.ape, x.gate, x.pick.keys), (KeyNorm::Layer(1e-6), 53, 56, 57, 58, 59, IdxKeys::Ring(10)));
        // (an RMS norm whose weight is (1 + w), and the mean-pooled form, are not the latent attention's indexer's)
        let odd = |form| Indexer { form, ..kind(0, false, true).indexer.unwrap() };
        for form in [kinds::IndexForm::Gated { key: Norm::rms_plus_one(1e-6) }, kinds::IndexForm::Mean { norm: Norm::rms(1e-6) }] {
            assert!(MlaOp::of(&kinds::Mla { indexer: Some(odd(form)), ..kind(0, false, false) }, MlaIn::Apart { q_a: 1, kv_a: 2 }, 3, 4, 0, 0, true).is_err());
        }
        // (a gate with two products, a rotated shared part, a shared key part beside an indexer)
        assert!(MlaOp::of(&kind(0, true, false), MlaIn::Apart { q_a: 1, kv_a: 2 }, 3, 4, 0, 0, false).is_err());
        assert!(MlaOp::of(&kinds::Mla { rope: Some(kinds::Rope::plain(1e4, 64)), ..kind(64, false, false) }, MlaIn::Fused { w: 0 }, 1, 2, 0, 0, false).is_err());
        assert!(MlaOp::of(&kind(64, false, true), MlaIn::Fused { w: 0 }, 1, 2, 0, 0, false).is_err());
    }
}
