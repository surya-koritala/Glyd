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
//! A rotary (`MlaOp::rot`, GLM-5.3's) turns interleaved pairs at each token's position: each query head's rope channels and the shared key part before it is cached,
//! and the first channels of the indexer's queries and keys (`mla_rope`). A layer whose indexer is `shared` runs none (`MlaOp::lent`): its rows attend to the picks
//! the latent attention before it made for them, which every layer keeps at the rows' places in the pass (`MlaRun::sel`, `cnt`), as transformers hands a layer
//! `prev_topk_indices`.
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
/// A sequence's rows at most in a call that the attention cuts its tokens into eight runs for (a decode step, the rows of a verified draft): by the
/// sequence's own rows (`seq_parts`), so a row's sums are those of its sequence's pass alone whatever else the pass holds.
const FEW: u32 = 8;

/// Where a latent layer's input products are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MlaIn {
    /// One product of the layer's input whose columns are q's latent, the output gate (heads v, where there is one), then the key-value latent and the shared key
    /// part (Kimi-K3's q_a, g and kv_a stacked).
    Fused { w: usize },
    /// q's latent and the key-value latent (and its shared key part) as two products (GLM-5.3's q_a and kv_a).
    Apart { q_a: usize, kv_a: usize },
    /// The queries one product of the layer's input, with no latent of their own (the operation's `q_b` is then [heads (nope + rope), hidden]: Ling's
    /// q_proj), and the key-value latent (and its shared key part) its own product.
    Direct { kv_a: usize },
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

/// How an indexer norms each token's key (`runtime::kinds::IndexForm::Gated`'s `key`), at its eps: nn.LayerNorm with its bias (GLM-5.3's), or an
/// RMS norm with a weight and no bias (the normed value rounded, times the weight, rounded again: Kimi-K3.1's `rmsnorm`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyNorm {
    Layer(f32),
    Rms(f32),
}

impl IdxPick {
    /// A head's values (the release's score: 128).
    pub fn dim(&self) -> usize {
        match self.score {
            IdxScore::F32 { dim, .. } => dim,
            IdxScore::Release => 128,
        }
    }
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

/// The roles of the small tensors a latent attention takes released in FP8 (`MlaOp::f8_small`: kv_b_proj, its indexer's queries', keys', head weights' and a
/// pooled indexer's gate products), which the loader holds as their E4M3 codes' values in bf16 beside the fp32 block scales: two bytes a value, twice the
/// release's.
pub const F8_SMALL_ROLES: [&str; 5] = ["kv_b", "indexer_wq_b", "indexer_wk", "indexer_weights", "indexer_pool_gate"];

/// Whether the checkpoint tensor `name` is a latent attention's small tensor of one of those roles, by the roles' names in a layer (`names::part::MLA`,
/// `MLA_INDEXER`): what the planner counts at two bytes a value where the release holds it in FP8.
pub fn f8_small_name(name: &str) -> bool {
    use crate::runtime::names::part;
    part::MLA.iter().chain(part::MLA_INDEXER).any(|(r, n)| F8_SMALL_ROLES.contains(r) && name.strip_suffix(n).is_some_and(|p| p.ends_with('.')))
}

/// `Op::Mla`: the latent attention of H into O. Matrices `input`'s, `q_b` [heads (nope + rope), q_rank] and `o` [hidden, heads v]; small tensors the q latent's
/// norm, the key-value latent's norm and kv_b_proj [heads (nope + v), kv_rank] (the absorbed form multiplies by its rows); the cache ring `ring`, kv_rank + rope a
/// token. Scores q . k times `scale` over the latent and the shared key part (rope channels as they are: none rotated), values the latent's; sigmoid(gate) on each
/// head's output where `gate` (its columns after q's latent in a fused product). `q_round`: the q latent's norm's rounding as the kind gives it; `Rounding::Twice`
/// (the normed value rounded, then times the weight, rounded: KimiRMSNorm, and transformers' RMSNorm in bf16, GLM-5.3's since its description says so) runs
/// rms2_rows512 (512 threads, Kimi-K3's order of sums), any other rms2_rows (256 threads, the same roundings in another order of sums). Queries of one
/// product (`MlaIn::Direct`) have no latent: `q_rank` is 0, `q_b` multiplies the layer's input, and the operation has two small tensors of the latent, the
/// key-value latent's norm and kv_b_proj (`q_norm` is the first of them and is not read as a norm of the queries).
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
    /// The output gate of one value a head (`kinds::GateFrom::Head`, Ling's): the small tensor [heads, hidden] whose product of the layer's input, in fp32 as
    /// the kind's reference holds it, gates every channel of its head's output by its sigmoid (kernels/mla.cu `mla_gate_head`). After the latent's three
    /// small tensors and an indexer's; a release may hold it in FP8 with block scales (`f8_small`).
    pub head_gate: Option<usize>,
    pub scale: f32,
    pub eps: f32,
    pub ring: usize,
    pub indexer: Option<MlaIndexer>,
    /// The rotary (GLM-5.3's): None where no channel turns (the rope channels as they are: Kimi-K3's, a NoPE layer's).
    pub rot: Option<MlaRot>,
    /// A layer whose indexer is `shared`: no indexer of its own; its rows attend to the picks the latent attention with an indexer before it made for them, a pick
    /// of this shape (the lender's: its room a row, `tmax`).
    pub lent: Option<IdxPick>,
}

/// A latent attention's rotary: table `table` of the description's (`ModelDesc::ropes`, `rope / 2` pairs) turns interleaved pairs of each query head's rope
/// channels and of the shared key part at their tokens' positions, and of the first `index` channels of the indexer's keys and query heads (0: none of them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaRot {
    pub table: usize,
    pub index: usize,
}

impl MlaOp {
    /// The operation of layer kind `k` whose input products are `input`, its other matrices `q_b` and `o`, its small tensors from `s0` (the q latent's norm, the
    /// key-value latent's, kv_b_proj, then an indexer's in `MlaIndexer::at`'s order), its caches the rings `ring` and `ring + 1`; a kind with a rotary turns by
    /// the description's table `table` (made for the kind's rotary: `ModelDesc::ropes`). A gate takes a fused product.
    #[allow(clippy::too_many_arguments)]
    pub fn of(k: &kinds::Mla, input: MlaIn, q_b: usize, o: usize, s0: usize, ring: usize, tail: bool, table: Option<usize>) -> Result<MlaOp> {
        // (the forms the CPU reference computes and no kernel does: refused by their names)
        if let Some(why) = k.not_run() {
            return err(why);
        }
        // (the parts of a latent attention the CPU reference computes and no kernel does: each refused by its name, before the kind is read further)
        let not_run = [
            (k.head_mix != 0, kinds::HEAD_MIX_NOT_RUN),
            (k.sinks, kinds::Mla::SINKS_NOT_RUN),
            (k.window.is_some(), kinds::Mla::WINDOW_NOT_RUN),
            (k.indexer.as_ref().is_some_and(|x| x.rope_style.is_some()), kinds::Indexer::ROPE_STYLE_NOT_RUN),
            (k.gate.is_some_and(|g| g.from == kinds::GateFrom::Latent), kinds::GateFrom::LATENT_NOT_RUN),
        ];
        if let Some((_, part)) = not_run.iter().find(|x| x.0) {
            return err(format!("a latent attention: {part}"));
        }
        // (the queries: through a latent of their own, in a fused product or apart; or one product of the layer's input)
        let q_rank = match (k.q_lora, input) {
            (Some(r), MlaIn::Fused { .. } | MlaIn::Apart { .. }) => r,
            (None, MlaIn::Direct { .. }) => 0,
            (Some(r), MlaIn::Direct { .. }) => {
                return err(format!("a latent attention whose queries have a latent of {r}, given as one product of the layer's input"))
            }
            (None, _) => {
                return err("a latent attention whose query has no low rank, given a product of its latent: its queries are one product of the layer's input")
            }
        };
        let rot = match (&k.rope, table) {
            (None, _) => None,
            (Some(r), Some(t)) if r.style == kinds::RopeStyle::Interleaved && r.scaling == kinds::RopeScaling::None && r.mrope.is_none() && r.dim == k.rope_dim && r.freq_dim == r.dim && r.freqs * 2 == r.dim => {
                Some(MlaRot { table: t, index: k.indexer.as_ref().filter(|x| !x.shared).map_or(0, |x| x.rope) })
            }
            (Some(r), _) => {
                return err(format!(
                    "a latent attention's rotary of {r:?}: the kernels turn interleaved pairs of the shared key part at the default frequencies, by the description's table"
                ))
            }
        };
        // (the output gate: a value a channel in a fused input product, or one a head, a small tensor's product of the layer's input)
        let (gate, by_head) = match (k.gate, input) {
            (None, _) => (false, false),
            (Some(g), _) if g.from == kinds::GateFrom::Latent => return err(format!("a latent attention: {}", kinds::GateFrom::LATENT_NOT_RUN)),
            (Some(g), _) if g.from == kinds::GateFrom::Head && g.act == kinds::Act::Sigmoid && g.norm.is_none() => (false, true),
            (Some(g), MlaIn::Fused { .. }) if g.from != kinds::GateFrom::Head && g.act == kinds::Act::Sigmoid && g.norm.is_none() => (true, false),
            (Some(g), _) => {
                return err(format!(
                    "a latent attention's gate of {g:?}: the kernels take a sigmoid gate, a value a channel in a fused input product or one a head"
                ))
            }
        };
        if let Some(x) = k.indexer.as_ref().filter(|x| x.rope > 0 && rot.is_none()) {
            return err(format!("a latent attention's indexer whose first {} channels turn, under no rotary of the attention's", x.rope));
        }
        let pick = |x: &kinds::Indexer| IdxPick {
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
        };
        let lent = k.indexer.as_ref().filter(|x| x.shared).map(pick);
        let indexer = match &k.indexer {
            None => None,
            Some(x) if x.shared => None,
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
                let ix = MlaIndexer { pick: pick(x), dim: x.head_dim, q_b: 0, k: 0, k_norm: 0, k_bias: 0, weights: 0, ape: 0, gate: 0, key };
                Some(ix.at(s0 + 3, ring + 1))
            }
        };
        // (the latent's small tensors: the q latent's norm where the queries have one, the key-value latent's norm, kv_b_proj; the gate a head's after them
        // and an indexer's)
        let base = if q_rank == 0 { 2 } else { 3 };
        let head_gate = by_head.then(|| s0 + base + indexer.as_ref().map_or(0, MlaIndexer::small_count));
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
            kv_norm: s0 + base - 2,
            kv_b: s0 + base - 1,
            q_round: k.q_norm.rounding,
            gate,
            head_gate,
            scale: k.scale,
            eps: k.q_norm.eps,
            ring,
            indexer,
            rot,
            lent,
        };
        a.check()?;
        Ok(a)
    }

    /// The small tensors of the latent: the q latent's norm, the key-value latent's norm and kv_b_proj; the last two where the queries are one product.
    fn latent_small(&self) -> usize {
        if self.q_rank == 0 {
            2
        } else {
            3
        }
    }

    /// The small tensors the operation reads: the latent's (`latent_small`), an indexer's (`MlaIndexer::small_count`), and the gate a head's.
    pub fn small_count(&self) -> usize {
        self.latent_small() + self.indexer.as_ref().map_or(0, MlaIndexer::small_count) + self.head_gate.is_some() as usize
    }

    /// The small tensors it takes released in FP8 with fp32 block scales (the release's E4M3 codes in bf16 and their scales: `Part::scales`): kv_b_proj, and
    /// its indexer's products (its queries', keys', head weights' and a pooled indexer's gate), and the gate a head's.
    pub fn f8_small(&self) -> Vec<usize> {
        let ix = self.indexer.iter().flat_map(|x| [x.q_b, x.k, x.weights].into_iter().chain(x.pooled().then_some(x.gate)));
        std::iter::once(self.kv_b).chain(ix).chain(self.head_gate).collect()
    }

    /// The matrices it multiplies by, in the order a loader holds them: a fused input product, q_b, o; or q_a, q_b, kv_a, o; or the queries' one product,
    /// kv_a, o.
    pub fn mat_count(&self) -> usize {
        match self.input {
            MlaIn::Fused { .. } | MlaIn::Direct { .. } => 3,
            MlaIn::Apart { .. } => 4,
        }
    }

    /// The same operation with its matrices from `m0` (`mat_count`'s order), its small tensors from `s0` (`MlaOp::of`'s order) and its rings from `ring`: a layer's
    /// place in its description.
    pub fn placed(&self, m0: usize, s0: usize, ring: usize) -> MlaOp {
        let (input, q_b, o) = match self.input {
            MlaIn::Fused { .. } => (MlaIn::Fused { w: m0 }, m0 + 1, m0 + 2),
            MlaIn::Apart { .. } => (MlaIn::Apart { q_a: m0, kv_a: m0 + 2 }, m0 + 1, m0 + 3),
            MlaIn::Direct { .. } => (MlaIn::Direct { kv_a: m0 + 1 }, m0, m0 + 2),
        };
        let indexer = self.indexer.as_ref().map(|x| x.at(s0 + 3, ring + 1));
        let base = self.latent_small();
        let head_gate = self.head_gate.map(|_| s0 + base + indexer.as_ref().map_or(0, MlaIndexer::small_count));
        MlaOp { input, q_b, o, q_norm: s0, kv_norm: s0 + base - 2, kv_b: s0 + base - 1, ring, indexer, head_gate, ..self.clone() }
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
        if self.gate && self.head_gate.is_some() {
            return err("a latent attention with a gate a channel and a gate a head");
        }
        // (an indexer's queries are a product of the normed query latent, which queries of one product do not have)
        if (self.q_rank == 0) != matches!(self.input, MlaIn::Direct { .. }) || (self.q_rank == 0 && (self.indexer.is_some() || self.lent.is_some())) {
            return err(format!(
                "a latent attention whose queries have a latent of {} in {:?}{}: queries of one product have none, and no indexer, whose queries are its product",
                self.q_rank,
                self.input,
                if self.indexer.is_some() || self.lent.is_some() { ", with an indexer" } else { "" }
            ));
        }
        if let Some(r) = self.rot {
            let ix = self.indexer.as_ref().map_or(0, |x| x.dim);
            if self.rope == 0 || self.rope % 2 != 0 || self.rope > 128 || r.index % 2 != 0 || r.index > self.rope || r.index > ix {
                return err(format!(
                    "a rotary of {} channels, {} of the indexer's {ix}: the kernels turn an even count up to 128, the indexer's no more than the attention's",
                    self.rope, r.index
                ));
            }
        }
        if self.indexer.is_some() && self.lent.is_some() {
            return err("a latent attention with an indexer of its own and lent picks");
        }
        if let Some(x) = &self.indexer {
            let (p, kp) = (&x.pick, x.pick.ratio);
            if x.dim == 0 || x.dim > 256 || kp == 0 || kp > 8 || p.top_k == 0 {
                return err(format!(
                    "an indexer of channels {} picking {} groups of {kp} keys: the kernels take indexer channels up to 256 and groups up to 8 keys",
                    x.dim, p.top_k
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
        match self.pick() {
            Some(p) => p.top_k * p.ratio + p.ratio - 1,
            None => ctx,
        }
    }

    /// The pick its rows attend to: its indexer's, or the one it is lent (`lent`).
    fn pick(&self) -> Option<&IdxPick> {
        self.indexer.as_ref().map(|x| &x.pick).or(self.lent.as_ref())
    }

    /// The pools a row's scores hold at most at a context of `ctx`.
    fn pmax(&self, ctx: usize) -> usize {
        self.indexer.as_ref().map_or(0, |x| ctx / x.pick.ratio + 1)
    }

    /// Rows the attention takes at a time: with an indexer as many as the room for its scores and the latent outputs holds (GLM-5.3's engine's blocks of rows), else
    /// a sequence's rows at once (Kimi-K3's dense attention). A row's attention cuts its tokens into runs where a call has a few rows (`FEW`), so these blocks are
    /// what each family's engine took.
    fn rows_at_once(&self, ch: usize, ctx: usize) -> usize {
        if self.pick().is_none() {
            return ch;
        }
        // (a layer lent its picks scores nothing: its rows are bounded by the latent outputs' room alone)
        let scores = match self.pmax(ctx) {
            0 => usize::MAX,
            p => (SUB_SCORES / (p * 4)).max(1),
        };
        let sub = ch.min((SUB_OLAT / (self.heads * self.kv_rank * 4)).max(1)).min(scores);
        // (a development build's check can ask for more rows at a time, to see a long call's stages whole)
        crate::tune!("GLYD_ENGINE_MLA_ROWS").and_then(|v| v.parse::<usize>().ok()).map_or(sub, |v| v.clamp(1, ch))
    }
}

/// kernels/mla.cu's kernels, each by its type (`crate::kt`).
pub(crate) struct KernMla {
    rms2: crate::kt::rms2_rows,
    pub(super) rms2_512: crate::kt::rms2_rows512,
    layernorm: crate::kt::layernorm_rows,
    scatter: crate::kt::scatter_rows,
    small: crate::kt::small_linear,
    small_f8: crate::kt::small_linear_f8s,
    small_f32: crate::kt::small_linear_f32,
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
    gate_head: crate::kt::mla_gate_head,
    dense_cnt: crate::kt::mla_dense_cnt,
    rope: crate::kt::mla_rope,
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
            small_f8: dev.typed()?,
            small_f32: dev.typed()?,
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
            gate_head: dev.typed()?,
            dense_cnt: dev.typed()?,
            rope: dev.typed()?,
            pages: dev.typed()?,
        })
    }

    /// y [m, rows] = x [m, cols] (rows xs apart) W^T for a bf16 W (`w.0`), rows ys apart; where W is released in FP8, its values the E4M3 codes' and `w.1`
    /// their fp32 scales of 128 x 128 blocks, each value times its block's scale once (`small_linear_f8s`: the same grid and order of sums); 0: as it is.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    #[allow(clippy::too_many_arguments)]
    unsafe fn small_linear(
        &self,
        s: crate::ffi::Stream,
        x: u64,
        xs: usize,
        w: (u64, u64),
        y: u64,
        ys: usize,
        m: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let grid = [m as u32, (rows.max(1) as u32).div_ceil(8), 1];
        if w.1 != 0 {
            return launch!(self.small_f8, grid, [256, 1, 1], s; x, xs as i32, w.0, w.1, y, ys as i32, rows as i32, cols as i32);
        }
        launch!(self.small, grid, [256, 1, 1], s; x, xs as i32, w.0, y, ys as i32, rows as i32, cols as i32)
    }

    /// `small_linear` into fp32 rows, its sums unrounded (kernels/mla.cu `small_linear_f32`, the same grid and order of sums): an indexer's head weights.
    ///
    /// # Safety
    /// Live device buffers of those sizes, y of fp32 rows, on the engine's stream.
    #[allow(clippy::too_many_arguments)]
    unsafe fn small_linear_f32(
        &self,
        s: crate::ffi::Stream,
        x: u64,
        xs: usize,
        w: (u64, u64),
        y: u64,
        ys: usize,
        m: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let grid = [m as u32, (rows.max(1) as u32).div_ceil(8), 1];
        launch!(self.small_f32, grid, [256, 1, 1], s; x, xs as i32, w.0, w.1, y, ys as i32, rows as i32, cols as i32)
    }
}

/// The latent attentions' scratch, one for every layer, sized for the largest: the input products (`x`, a row as wide as the widest), the normed q latent, the
/// queries, the values' output, the indexer's key or gate product and its queries (bf16), its head weights (fp32, as the kind's reference holds them: a near
/// tie of the scores goes their way), a chunk's rows; for the rows the attention takes at a time
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
            MlaIn::Direct { kv_a } => mats[kv_a].whole_rows(),
        });
        let b = |n: usize| Buf::new(n.max(4));
        Ok(Some(MlaRun {
            x: b(ch * x * 2)?,
            qr: b(ch * most(&|a| a.q_rank) * 2)?,
            q: b(ch * most(&|a| a.heads * (a.nope + a.rope)) * 2)?,
            ov: b(ch * most(&|a| a.heads * a.v) * 2)?,
            ix: b(ch * most(&|a| ix(a, &|x| x.dim)) * 2)?,
            iq: b(ch * most(&|a| ix(a, &|x| x.pick.heads * x.dim)) * 2)?,
            // (an indexer's head weights, or the product of a gate a head: fp32, a value a head)
            wts: b(ch * most(&|a| ix(a, &|x| x.pick.heads).max(if a.head_gate.is_some() { a.heads } else { 0 })) * 4)?,
            qa: b(sub * most(&|a| a.heads * (a.kv_rank + a.rope)) * 4)?,
            olat: b(sub * most(&|a| a.heads * a.kv_rank) * 4)?,
            scores: b(sub * most(&|a| a.pmax(ctx)) * 4)?,
            // (every row's picks at its place in the pass, a chunk's: a layer whose indexer is shared reads the ones the layer before it left)
            sel: b(ch.max(sub) * most(&|a| if a.pick().is_some() { a.tmax(ctx) } else { 0 }) * 4)?,
            cnt: b(ch.max(sub) * 4)?,
            part: b(FEW as usize * most(&|a| a.heads * 8 * (a.kv_rank + 2)) * 4)?,
        }))
    }
}

/// Where an indexer's pick reads and writes for `rows` rows: their queries (heads dim a row), head weights (heads a row, fp32 where `w32`, else bf16), the entries' keys (`keys`, a ring's through
/// `tab`, rows `stride` apart; row r's table at `tab` + r `ts` entries where a step's rows that decode each read their own sequence's, `rows::RowTabs`; 0: one table),
/// the rows' positions, the scores' room (`emax` a row), the picks (`sk` a row) and their counts.
pub(crate) struct PickAt {
    pub q: u64,
    pub w: u64,
    pub w32: bool,
    pub keys: u64,
    pub stride: usize,
    pub tab: u64,
    pub ts: usize,
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

/// One latent attention's indexer in a pass `Dyn::trace_picks` traced: its layer, each row's position and picks (the positions its query attends to, in order),
/// and the indexer's inputs as the GPU held them: each row's queries [heads dim] (bf16) and head weights' product [heads] (before heads^-0.5; fp32 where the GPU
/// holds it so, a latent attention's, else bf16), and the key of
/// each complete entry the pass's last row sees, in order (a pool's as pooled; a token's own where an entry is a token); and where a row picks inside a
/// candidate source's published blocks (`Cand::Inside`), the entries those blocks hold (`inside`, None: every entry it sees). What a check runs the reference's
/// choice on (`runtime::reference::mla::choose_among`) where the engine's picks are not the reference's own.
#[derive(Clone, Debug)]
pub struct PickTrace {
    pub layer: usize,
    pub pos: Vec<usize>,
    pub picks: Vec<Vec<usize>>,
    pub q: Vec<Vec<f32>>,
    pub w: Vec<Vec<f32>>,
    pub keys: Vec<Vec<f32>>,
    pub inside: Vec<Option<Vec<usize>>>,
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

    /// `Dyn::trace_picks`' record of `op` of layer `l` on n rows, a latent attention with an indexer (`PickTrace`): each row's picks as the attention left them (`MlaRun`'s
    /// sel and cnt, which hold every row of a pass of one sequence whose rows the attention takes at once: `rows_at_once`), its queries and head weights (iq, wts)
    /// and the keys of the complete entries in the ring (unpaged); another pass is an error. `pos`: the rows' positions as the operation read them (a decode step's
    /// are in its own buffer, not the pass's start).
    pub(super) fn trace_pick(&self, p: &Part, l: usize, op: &Op, n: usize, pos: u64) -> Result<()> {
        let Op::Mla(a @ MlaOp { indexer: Some(ix), .. }) = op else { return Ok(()) };
        // (the model's passes, and a draft head's of the model (src/dynamic/head.rs), whose rings follow the model's)
        if !(std::ptr::eq(p, &self.big) || matches!(p.d.first, Op::MtpIn { .. })) || self.pick_trace.borrow().is_none() {
            return Ok(());
        }
        let (most, tab) = (a.rows_at_once(self.chunk, self.ctx), self.table_of(self.seq, a.ring + 1));
        if n > most || !self.parts.is_empty() || tab != 0 {
            return err(format!(
                "layer {l}: picks traced on a pass of {n} rows of one sequence, at most {most}, its ring unpaged (this one: {} sequences, {})",
                self.parts.len().max(1),
                if tab != 0 { "paged" } else { "unpaged" }
            ));
        }
        let r = self.mla.as_ref().ok_or_else(|| Error("a latent attention in a model loaded without its buffers".into()))?;
        let (tmax, ih, id, kp) = (a.tmax(self.ctx), ix.pick.heads, ix.dim, ix.pick.ratio);
        let (mut cnt, mut sel, mut q, mut w, mut at) = (vec![0i32; n], vec![0i32; n * tmax], vec![0u16; n * ih * id], vec![0f32; n * ih], vec![0i32; n]);
        self.dev.sync()?;
        // SAFETY: the n positions the operation read (a buffer of the pass's), into host memory of n of them; the stream is synchronized after.
        unsafe { crate::dev::copy_d2h_async(at.as_mut_ptr() as *mut _, pos, n * 4, self.dev.stream)? };
        self.dev.sync()?;
        if at.iter().any(|&x| x < 0) || at.windows(2).any(|w| w[1] != w[0] + 1) {
            return err(format!("layer {l}: picks traced on rows at positions {at:?}, where a pass's rows are one sequence's in order"));
        }
        // (the ring's rows of the complete entries the last row sees: [key | gate score] a token, a pool's key in its last token's row)
        let last = at.last().map_or(0, |&x| x as usize);
        let entries = (last + 1) / kp;
        let mut ring = vec![0u16; entries * kp * 2 * id];
        r.cnt.read(0, &mut cnt)?;
        // (each row's picks alone: the rest of a row of sel is never written)
        for (t, &c) in cnt.iter().enumerate() {
            let c = (c.max(0) as usize).min(tmax);
            r.sel.read(t * tmax * 4, &mut sel[t * tmax..][..c])?;
        }
        r.iq.read(0, &mut q)?;
        r.wts.read(0, &mut w)?;
        self.kc.read(self.ring_off[a.ring + 1].0 as usize, &mut ring)?;
        let f = |v: &[u16]| v.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect::<Vec<f32>>();
        let mut picks = Vec::with_capacity(n);
        for (t, &c) in cnt.iter().enumerate() {
            let got = &sel[t * tmax..][..(c.max(0) as usize).min(tmax)];
            if c < 0 || c as usize > tmax || got.iter().any(|&e| e < 0) {
                return err(format!("layer {l} row {t}: {c} picks of at most {tmax}, or one below 0"));
            }
            picks.push(got.iter().map(|&e| e as usize).collect());
        }
        let rec = PickTrace {
            layer: l,
            pos: at.iter().map(|&x| x as usize).collect(),
            inside: vec![None; picks.len()],
            picks,
            q: q.chunks(ih * id).map(f).collect(),
            w: w.chunks(ih).map(<[f32]>::to_vec).collect(),
            keys: (0..entries).map(|j| f(&ring[(j * kp + kp - 1) * 2 * id..][..id])).collect(),
        };
        if let Some(t) = self.pick_trace.borrow_mut().as_mut() {
            t.push(rec);
        }
        Ok(())
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
                    at.q, (x.heads * dim) as i32, at.w, wsr, at.w32 as i32, at.keys, at.stride as i32, pstep, poff, at.tab, at.ts as i64, at.pos, at.scores, at.emax as i32,
                    x.heads as i32, dim as i32, x.ratio as i32, x.top_k as i32, scale, x.wscale)?;
            }
            IdxScore::Release => {
                // (the queries and weights, then idx_score_bf16's warps' terms: at most 64 heads)
                let smem = (x.heads * 129 * 4 + 8 * 64 * 4) as u32;
                launch!(k.score_bf16, [n, (at.emax as u32).div_ceil(8), 1], [256, 1, 1], smem smem, s;
                    at.q, at.w, x.wscale, at.keys, at.tab, at.ts as i64, at.pos, at.scores, x.heads as i32, x.ratio as i32, at.emax as i32)?;
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
    /// A pass's first rows that decode, a token of a sequence each (`Dyn::lead`), take each of these at once, each row through its own block tables (`rows::RowTabs`):
    /// each row's kernels and sums are its own sequence's pass's on its one token (the attention cut into runs as a decode row's is, `FEW` rows a launch). Not where the
    /// latents' ring is in 12-bit pages: each sequence's pages are unpacked into the one staging ring by itself.
    pub(super) unsafe fn mla(&self, p: &Part, a: &MlaOp, n: usize, pos: u64) -> Result<()> {
        let r = self.mla.as_ref().ok_or_else(|| Error("a latent attention in a model loaded without its buffers".into()))?;
        let (s, k) = (self.dev.stream, &self.km);
        let (nh, kr, ks, qw, nr) = (a.heads, a.kv_rank, a.kv_rank + a.rope, a.nope + a.rope, n as u32);
        let h = p.slot(Slot::H);
        // (12-bit pages: a row of the call to its staging row, `pw` wide, the channels past the latent's zero; then the rows into their pages)
        let staged = self.pages12(p, a.ring).map(|rp| (rp.shape.heads * rp.shape.channels, self.lossless.as_ref().expect("pages").rows_k.ptr));
        // (the rows that decode taken at once, and the pass's other sequences, each by itself)
        let lead = if staged.is_some() || self.parts.is_empty() { 0 } else { self.lead.get() };
        let seqs: Vec<(usize, usize, u64, usize)> = self.seq_parts(n).into_iter().skip(lead).collect();
        let lr = lead as u32;
        // the q latent (and, fused, the gate and the key-value latent) product, its norm, q
        let x = r.x.ptr;
        // (the row stride of the product q's latent is in; none where the queries are one product of the layer's input)
        let q_row = match a.input {
            MlaIn::Fused { w } => p.mats[w].whole_rows(),
            MlaIn::Apart { q_a, .. } => p.mats[q_a].whole_rows(),
            MlaIn::Direct { .. } => 0,
        };
        if let MlaIn::Fused { w: q_at } | MlaIn::Apart { q_a: q_at, .. } = a.input {
            self.lin(&p.mats[q_at], h, n, x)?;
            self.mark("latent attention: q latent product")?;
            let qn = p.small[a.q_norm].ptr;
            if a.q_round == kinds::Rounding::Twice {
                launch!(k.rms2_512, [nr, 1, 1], [512, 1, 1], s; x, q_row as i32, qn, r.qr.ptr, a.q_rank as i32, a.q_rank as i32, a.eps)?;
            } else {
                launch!(k.rms2, [nr, 1, 1], [256, 1, 1], s; x, q_row as i32, qn, r.qr.ptr, a.q_rank as i32, 0u64, 0u64, 0i64, a.q_rank as i32, a.eps)?;
            }
            self.lin(&p.mats[a.q_b], r.qr.ptr, n, r.q.ptr)?;
        } else {
            // (queries of one product: no latent, no norm)
            self.lin(&p.mats[a.q_b], h, n, r.q.ptr)?;
        }
        // (a rotary: each head's rope channels, after its nope ones, turned at the row's position)
        if let Some(rt) = a.rot {
            self.mla_rope(r.q.ptr, nh * qw, (nh, qw), a.nope, pos, rt.table, a.rope / 2, n)?;
        }
        self.mark("latent attention: q")?;
        // the key-value latent normed into the cache, the shared key part after it
        let (kv, kv_row) = match a.input {
            MlaIn::Fused { w } => (x + ((a.q_rank + if a.gate { nh * a.v } else { 0 }) * 2) as u64, p.mats[w].whole_rows()),
            MlaIn::Apart { kv_a, .. } | MlaIn::Direct { kv_a } => {
                self.lin(&p.mats[kv_a], h, n, x)?;
                (x, p.mats[kv_a].whole_rows())
            }
        };
        // (a rotary: the shared key part, after the latent, turned at the row's position before it is cached)
        if let Some(rt) = a.rot {
            self.mla_rope(kv, kv_row, (1, 0), kr, pos, rt.table, a.rope / 2, n)?;
        }
        if let Some((pw, rows)) = staged.filter(|x| x.0 > ks) {
            crate::dev::memset_async(rows, 0, n * pw * 2, s)?;
        }
        if lead > 0 {
            let (lat, tab, ts) = self.lead_ring(a.ring);
            launch!(k.rms2, [lr, 1, 1], [256, 1, 1], s; kv, kv_row as i32, p.small[a.kv_norm].ptr, lat, ks as i32, pos, tab, ts as i64, kr as i32, a.eps)?;
            if a.rope > 0 {
                launch!(k.scatter, [lr, 1, 1], [256, 1, 1], s; kv + (kr * 2) as u64, kv_row as i32, lat + (kr * 2) as u64, ks as i32, pos, tab, ts as i64, a.rope as i32)?;
            }
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
            launch!(k.rms2, [mr, 1, 1], [256, 1, 1], s; xr, kv_row as i32, p.small[a.kv_norm].ptr, lat, ys as i32, at, tab, 0i64, kr as i32, a.eps)?;
            if a.rope > 0 {
                launch!(k.scatter, [mr, 1, 1], [256, 1, 1], s; xr + (kr * 2) as u64, kv_row as i32, lat + (kr * 2) as u64, ys as i32, at, tab, 0i64, a.rope as i32)?;
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
            // (a product's matrix and its block scales where it is released in FP8)
            let wt = |i: usize| (p.small[i].ptr, p.scales(i));
            k.small_linear(s, r.qr.ptr, a.q_rank, wt(ix.q_b), r.iq.ptr, ih * id, n, ih * id, a.q_rank)?;
            k.small_linear(s, h, hid, wt(ix.k), r.ix.ptr, id, n, id, hid)?;
            // (a rotary of the indexer's: each query head's first channels, and each key's after its norm, turned at the row's position: the keys normed in
            // place, turned, then into the cache)
            let turn = a.rot.filter(|rt| rt.index > 0);
            if let Some(rt) = turn {
                self.mla_rope(r.iq.ptr, ih * id, (ih, id), 0, pos, rt.table, rt.index / 2, n)?;
                match ix.key {
                    KeyNorm::Layer(eps) => {
                        launch!(k.layernorm, [nr, 1, 1], [128, 1, 1], s; r.ix.ptr, id as i32, sm(ix.k_norm), sm(ix.k_bias), r.ix.ptr, id as i32, 0u64, 0u64, 0i64, id as i32, eps)?
                    }
                    KeyNorm::Rms(eps) => {
                        launch!(k.rms2, [nr, 1, 1], [256, 1, 1], s; r.ix.ptr, id as i32, sm(ix.k_norm), r.ix.ptr, id as i32, 0u64, 0u64, 0i64, id as i32, eps)?
                    }
                }
                self.mla_rope(r.ix.ptr, id, (1, 0), 0, pos, rt.table, rt.index / 2, n)?;
            }
            // (each token's key normed into its cache's row, as the kind's form says: a LayerNorm with its bias, or the weighted RMS norm; the rows that decode at once,
            // each through its own table, then each other sequence's rows)
            let lead_keys = (lead > 0).then(|| self.lead_ring(a.ring + 1));
            let keyed = lead_keys.map(|(ring, tab, ts)| (0usize, lead, ring, tab, ts)).into_iter().chain(seqs.iter().map(|&(r0, m, kc, slot)| {
                let (ring, tab) = self.ring_at(kc, slot, a.ring + 1);
                (r0, m, ring, tab, 0)
            }));
            for (r0, m, ring, tab, ts) in keyed.clone() {
                let (x, at, ts) = (r.ix.ptr + (r0 * id * 2) as u64, pos + (r0 * 4) as u64, ts as i64);
                if turn.is_some() {
                    launch!(k.scatter, [m as u32, 1, 1], [256, 1, 1], s; x, id as i32, ring, (2 * id) as i32, at, tab, ts, id as i32)?;
                    continue;
                }
                match ix.key {
                    KeyNorm::Layer(eps) => {
                        launch!(k.layernorm, [m as u32, 1, 1], [128, 1, 1], s; x, id as i32, sm(ix.k_norm), sm(ix.k_bias), ring, (2 * id) as i32, at, tab, ts, id as i32, eps)?
                    }
                    KeyNorm::Rms(eps) => {
                        launch!(k.rms2, [m as u32, 1, 1], [256, 1, 1], s; x, id as i32, sm(ix.k_norm), ring, (2 * id) as i32, at, tab, ts, id as i32, eps)?
                    }
                }
            }
            // (pools: each token's gate scores beside its key, then the keys of the pools these rows complete; a key a token is its entry's as it is. A row that decodes
            // completes its pool alone, a call of one row: grid y its rows)
            if ix.pooled() {
                k.small_linear(s, h, hid, wt(ix.gate), r.ix.ptr, id, n, id, hid)?;
                for (r0, m, ring, tab, ts) in keyed.clone() {
                    launch!(k.scatter, [m as u32, 1, 1], [256, 1, 1], s; r.ix.ptr + (r0 * id * 2) as u64, id as i32, ring + (id * 2) as u64, (2 * id) as i32, pos + (r0 * 4) as u64, tab, ts as i64, id as i32)?;
                }
            }
            k.small_linear_f32(s, h, hid, wt(ix.weights), r.wts.ptr, ih, n, ih, hid)?;
            if ix.pooled() {
                for (r0, m, ring, tab, ts) in keyed {
                    // (the rows that decode (their tables `ts` apart): a call of one row each, grid y; a sequence's rows: one call)
                    let (w2, (calls, rows)) = ((2 * id) as i32, if ts > 0 { (1, m) } else { (m, 1) });
                    launch!(k.pool, [(calls as u32).div_ceil(kp as u32) + 1, rows as u32, 1], [128, 1, 1], s;
                        ring, w2, ring + (id * 2) as u64, w2, sm(ix.ape), ring, w2, kp as i32, kp as i32 - 1, tab, ts as i64, pos + (r0 * 4) as u64, calls as i32, id as i32, kp as i32)?;
                }
            }
            self.mark("latent attention: the indexer's products and pools")?;
        }
        // the attention, a block of rows at a time
        let (sub, tmax, pmax) = (a.rows_at_once(self.chunk, self.ctx), a.tmax(self.ctx), a.pmax(self.ctx));
        // (kv_b_proj, and its fp32 block scales where it is released in FP8: each value times its block's scale once, kernels/mla.cu)
        let (wkv, wsc) = (p.small[a.kv_b].ptr, p.scales(a.kv_b));
        // (the rows of the pass from `r0`, m of them: their absorbed queries, the indexer's picks (its keys through `itab`, `its` apart) or every position, the attention over the
        // latents `lat` (rows `ls` apart, through `tab`; row j of them through the table `ts` entries on, where they are rows that decode), the values)
        let block = |r0: usize, m: usize, (lat, ls): (u64, usize), (tab, ts): (u64, usize), (keys, itab, its): (u64, u64, usize)| -> Result<()> {
            let p0 = pos + (r0 * 4) as u64;
            // (the rows' picks and counts at their rows' places in the pass: a layer whose indexer is shared reads the ones the latent attention before it left)
            let (sel_at, cnt_at) = (r.sel.ptr + (r0 * tmax * 4) as u64, r.cnt.ptr + (r0 * 4) as u64);
            launch!(k.absorb, [nh as u32, (m as u32).div_ceil(8), 1], [256, 1, 1], s;
                r.q.ptr + (r0 * nh * qw * 2) as u64, (nh * qw) as i32, qw as i32, wkv, wsc, r.qa.ptr, m as i32, nh as i32, a.nope as i32, a.v as i32, kr as i32, ks as i32)?;
            let sel = match &a.indexer {
                Some(ix) => {
                    let (ih, id) = (ix.pick.heads, ix.dim);
                    let at = PickAt {
                        q: r.iq.ptr + (r0 * ih * id * 2) as u64,
                        w: r.wts.ptr + (r0 * ih * 4) as u64,
                        w32: true,
                        keys,
                        stride: 2 * id,
                        tab: itab,
                        ts: its,
                        pos: p0,
                        rows: m,
                        scores: r.scores.ptr,
                        emax: pmax,
                        sel: sel_at,
                        sk: tmax,
                        cnt: cnt_at,
                        cand: CandAt::default(),
                    };
                    self.idx_pick(&ix.pick, &at)?;
                    sel_at
                }
                None if a.lent.is_some() => sel_at,
                None => {
                    launch!(k.dense_cnt, [(m as u32).div_ceil(256), 1, 1], [256, 1, 1], s; p0, cnt_at, m as i32)?;
                    0
                }
            };
            self.mla_attn(r, (lat, ls), (tab, ts), (sel, cnt_at), tmax, a, m)?;
            launch!(k.expand, [nh as u32, (m as u32).div_ceil(4), 1], [256, 1, 1], smem (4 * kr * 4) as u32, s;
                r.olat.ptr, wkv, wsc, r.ov.ptr + (r0 * nh * a.v * 2) as u64, m as i32, nh as i32, a.nope as i32, a.v as i32, kr as i32)
        };
        if lead > 0 {
            // (the rows that decode, `FEW` at a time: each row's attention cut into runs as a decode row's of its own sequence is)
            let ((lat, tab, ts), keys) = (self.lead_ring(a.ring), if a.indexer.is_some() { self.lead_ring(a.ring + 1) } else { (0, 0, 0) });
            for r0 in (0..lead).step_by(sub.min(FEW as usize)) {
                let m = sub.min(FEW as usize).min(lead - r0);
                block(r0, m, (lat, ks), (super::rows::tab_at(tab, ts, r0), ts), (keys.0, super::rows::tab_at(keys.1, keys.2, r0), keys.2))?;
            }
        }
        // (12-bit pages: each sequence's rows' positions end here at most: its part's, or the selected sequence's from its length or the pass's start; a step's graph
        // being captured, the context)
        let ends: Vec<usize> = match (self.capturing.get(), self.parts.is_empty()) {
            (true, _) => vec![self.ctx; seqs.len()],
            (false, true) => vec![self.start.get().max(self.n) + n],
            (false, false) => self.parts.iter().skip(lead).map(|q| q.start + q.rows).collect(),
        };
        for (&(first, rows, kc, slot), &end) in seqs.iter().zip(&ends) {
            let (lat, tab, ls) = match staged {
                Some((pw, _)) => (self.kv12_latents(a.ring, slot, pos + ((first + rows - 1) * 4) as u64, end.div_ceil(16))?, 0, pw),
                None => {
                    let (lat, tab) = self.ring_at(kc, slot, a.ring);
                    (lat, tab, ks)
                }
            };
            let (keys, itab) = if a.indexer.is_some() { self.ring_at(kc, slot, a.ring + 1) } else { (0, 0) };
            let mut r0 = first;
            while r0 < first + rows {
                let m = sub.min(first + rows - r0);
                block(r0, m, (lat, ls), (tab, 0), (keys, itab, 0))?;
                r0 += m;
            }
        }
        self.mark("latent attention: attention")?;
        if a.gate {
            let w = nh * a.v;
            launch!(k.gate, [((n * w) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; r.ov.ptr, x + (a.q_rank * 2) as u64, q_row as i32, w as i32, n as i32)?;
        }
        if let Some(g) = a.head_gate {
            // (a gate a head: its product of the layer's input in fp32, a value a head (its block scales where it is released in FP8), into the indexer's
            // head weights' room, which the picks are done with; then every channel of a head's output times its sigmoid)
            let hid = p.d.hidden;
            k.small_linear_f32(s, h, hid, (p.small[g].ptr, p.scales(g)), r.wts.ptr, nh, n, nh, hid)?;
            launch!(k.gate_head, [((n * nh * a.v) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; r.ov.ptr, r.wts.ptr, nh as i32, a.v as i32, n as i32)?;
        }
        self.lin(&p.mats[a.o], r.ov.ptr, n, p.slot(Slot::O))?;
        self.mark("latent attention: o product")
    }

    /// `mla_rope` (kernels/mla.cu) on n rows of x (rows xs apart) at the rows' positions `pos`: `heads` heads `hs` apart, `pairs` interleaved pairs from channel
    /// `off`, by the description's rotary table `t`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mla_rope(&self, x: u64, xs: usize, (heads, hs): (usize, usize), off: usize, pos: u64, t: usize, pairs: usize, n: usize) -> Result<()> {
        let (cs, sn) = self.ropes.get(t).ok_or_else(|| Error(format!("a latent attention's rotary table {t}, of {}", self.ropes.len())))?;
        launch!(self.km.rope, [n as u32, heads.max(1) as u32, 1], [pairs.next_multiple_of(32) as u32, 1, 1], self.dev.stream;
            x, xs as i32, hs as i32, off as i32, pos, cs.ptr, sn.ptr, pairs as i32)
    }

    /// olat [m, heads, kv_rank] fp32: each row's and head's softmax attention over the tokens the counts (and picks, `sel`; 0: every position up to the row's) give,
    /// of the latents `lat.0` (rows `lat.1` apart: the ring's, through `tab`, or a 12-bit ring's pages unpacked, position p at row p). A few rows (at most `FEW`) cut
    /// their tokens into 8 runs over blocks and merge them; more rows are parallel enough by themselves.
    unsafe fn mla_attn(&self, r: &MlaRun, lat: (u64, usize), (tab, ts): (u64, usize), (sel, cnt): (u64, u64), tmax: usize, a: &MlaOp, m: usize) -> Result<()> {
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
            r.qa.ptr, lat.0, lat.1 as i32, tab, ts as i64, sel, cnt, r.olat.ptr, r.part.ptr, tmax, nh, kr, ks, a.scale, splits, tile)?;
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
                rope: 0,
                rope_style: None,
                q_in: None,
                pos_scale: false,
            }),
            head_mix: 0,
            sinks: false,
            window: None,
            pos_scale: false,
            kv_from: None,
            rope_learned: false,
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

    /// The scores' factor is the releases' own, to the bit: each latent attention the description builder reads from Kimi-K3's and GLM-5.3-Flash's released configs has
    /// 1 / sqrt(nope + rope) (GLM-5.3's rope part is 0: nope^-0.5), and the operation takes the kind's factor as it is.
    #[test]
    fn the_scores_factor_is_the_releases() {
        use crate::runtime::kinds::{Body, Mixer};
        let fixture = |f: &str| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(format!("{}/tests/fixtures/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
        };
        let kimi: serde_json::Value = serde_json::from_str(fixture("headers/Kimi-K3.json")["config"].as_str().unwrap()).unwrap();
        for (name, cfg, width) in [("Kimi-K3", kimi, 128 + 64), ("GLM-5.3-Flash", fixture("glm5_next_config.json"), 256)] {
            let tc = cfg.get("text_config").unwrap_or(&cfg);
            assert_eq!(tc["qk_nope_head_dim"].as_u64().unwrap() + tc["qk_rope_head_dim"].as_u64().unwrap(), width, "{name}");
            let d = crate::runtime::describe::describe(&cfg).unwrap_or_else(|e| panic!("{name}: {}", e.msg));
            let mlas: Vec<&kinds::Mla> = d
                .desc
                .blocks
                .iter()
                .filter_map(|b| match &b.body {
                    Body::Mixer(Mixer::Mla(m)) => Some(m),
                    _ => None,
                })
                .collect();
            assert!(!mlas.is_empty(), "{name}: no latent attention described");
            for m in mlas {
                assert_eq!(m.scale.to_bits(), (1.0 / (width as f32).sqrt()).to_bits(), "{name}: {}", m.scale);
            }
        }
        let k = kind(64, true, false);
        assert_eq!(MlaOp::of(&k, MlaIn::Fused { w: 5 }, 6, 7, 10, 2, false, None).unwrap().scale.to_bits(), k.scale.to_bits());
    }

    /// The kind's parameters land where the kernels read them: a fused product with its gate (Kimi-K3), two products and an indexer (GLM-5.3) whose small tensors
    /// follow the latent's three and whose cache is the next ring; what the kernels do not take is refused.
    #[test]
    fn the_kind_lowers_to_the_operation() {
        let twice = Norm { rounding: kinds::Rounding::Twice, ..Norm::rms(1e-6) };
        let k = MlaOp::of(&kinds::Mla { q_norm: twice, ..kind(64, true, false) }, MlaIn::Fused { w: 5 }, 6, 7, 10, 2, false, None).unwrap();
        assert_eq!((k.gate, k.q_norm, k.kv_norm, k.kv_b, k.small_count(), k.rope, k.q_rank), (true, 10, 11, 12, 3, 64, 128));
        // (the q latent's norm: KimiRMSNorm's rounding is the kind's, which picks its kernel; GLM-5.3's kind gives its norm in f32)
        assert_eq!(k.q_round, kinds::Rounding::Twice);
        let twice_not = kind(0, false, true).q_norm.rounding;
        let g = MlaOp::of(&kind(0, false, true), MlaIn::Apart { q_a: 1, kv_a: 2 }, 3, 4, 20, 6, true, None).unwrap();
        let x = g.indexer.as_ref().unwrap();
        assert_eq!((x.q_b, x.gate, x.pick.top_k, x.pick.keys, g.small_count(), g.q_round), (23, 29, 2, IdxKeys::Ring(7), 10, twice_not));
        assert_eq!((g.tmax(1024), g.pmax(1024), g.rows_at_once(64, 1024)), (2 * 4 + 3, 257, 64));
        // (a key a token, RMS-normed: no bias, no pooling tensors; its four small tensors follow the latent's three. LayerNorm-ed: its bias too)
        let per_token = |key: Norm| {
            let ix = Indexer { block: 1, top_k: 8, form: kinds::IndexForm::Gated { key }, ..kind(0, false, true).indexer.unwrap() };
            MlaOp::of(&kinds::Mla { indexer: Some(ix), ..kind(0, false, false) }, MlaIn::Fused { w: 0 }, 1, 2, 20, 6, false, None).unwrap()
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
            assert!(
                MlaOp::of(&kinds::Mla { indexer: Some(odd(form)), ..kind(0, false, false) }, MlaIn::Apart { q_a: 1, kv_a: 2 }, 3, 4, 0, 0, true, None).is_err()
            );
        }
        // (an indexer beside a shared key part, not rotated: the attention's scores are over the latent and the shared part of the tokens it picks)
        let both = MlaOp::of(&kind(64, false, true), MlaIn::Fused { w: 0 }, 1, 2, 0, 0, false, None).unwrap();
        assert_eq!((both.rope, both.indexer.as_ref().map(|x| x.pick.ratio), both.rings().len()), (64, Some(4), 2));
        // (a gate with two products, a rotated shared part)
        assert!(MlaOp::of(&kind(0, true, false), MlaIn::Apart { q_a: 1, kv_a: 2 }, 3, 4, 0, 0, false, None).is_err());
        assert!(MlaOp::of(&kinds::Mla { rope: Some(kinds::Rope::plain(1e4, 64)), ..kind(64, false, false) }, MlaIn::Fused { w: 0 }, 1, 2, 0, 0, false, None)
            .is_err());
        // (Mistral Large's forms, which the CPU reference computes and no kernel does: each refused by its name)
        let on_input = |x: Indexer| Indexer { q_in: Some(Norm::rms(1e-6)), pos_scale: true, ..x };
        for (k, why) in [
            (kinds::Mla { pos_scale: true, ..kind(0, false, false) }, kinds::Mla::POS_SCALE_NOT_RUN),
            (kinds::Mla { kv_from: Some(0), ..kind(0, false, false) }, kinds::Mla::KV_FROM_NOT_RUN),
            (kinds::Mla { rope_learned: true, ..kind(64, false, false) }, kinds::Mla::ROPE_LEARNED_NOT_RUN),
            (kinds::Mla { indexer: kind(0, false, true).indexer.map(on_input), ..kind(0, false, false) }, kinds::Mla::INDEX_INPUT_NOT_RUN),
        ] {
            let Err(e) = MlaOp::of(&k, MlaIn::Fused { w: 0 }, 1, 2, 0, 0, false, None) else { panic!("lowered: {why}") };
            assert_eq!(e.msg, why);
            assert!(why.ends_with("which the executor does not run yet"));
        }
        assert_eq!(kind(64, true, true).not_run(), None);
    }
}
