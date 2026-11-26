//! The compressed attention layer kind (`kinds::Comp`) with the streams around its blocks (`Residual::Streams`), the experts after it (`Ffn::Moe`) and the n-gram
//! blocks before it (`Body::Ngram`), lowered from a model's description to the executor's operations and to what the loader reads for each index (`Plan`, the
//! tensors by the description's names). One walk over the blocks for every form of the kinds: each layer's compressor and indexer its own (a ratio with windows
//! that overlap and an indexer, or a ratio whose entries a query reads all of), or shared from source layers (`CompShare`: the entries, the indexer's keys made of
//! them, the picks, the candidate blocks); the streams' mix (`StreamsP::lagged`), the head's collapse (`Collapse::Weighted`, `Collapse::LastPre`), the router's
//! logit factor. dynamic/deepseek_v4.rs loads a plan in the release's formats.

use super::deepseek_v4::V4Opts;
use super::ngram::{NgramAdd, NgramConv, NgramRows};
use super::{Cand, HeadMix, IdxKeys, IdxPick, IdxScore, ModelDesc, MoeOp, NormW, Op, Ring, Slot, Stage, StreamsMix, V4Dims, SLOTS};
use crate::ffi::{err, Error, Result};
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{Body, Collapse, Ffn, Mixer, Moe, Ngram, NgramForm, NormWeight, Residual, Rope, StreamsP};

/// A part of a small tensor as the loader makes it: a checkpoint tensor's bytes as saved (or as fp32: a bf16 one widened, exact), zeros, int64 values the lowering
/// computes, or the tokenizer's compressed vocabulary (Engram's: int32 [the model's vocabulary], `NgramForm::Engram`).
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Tensor(String),
    Fp32(String),
    Zeros(usize),
    I64(Vec<i64>),
    TokenMap,
}

/// How a product's matrix is read: a quantized matrix (FP8 with its scales; rows `rows` of it, or `groups` stacked groups each sliced on its own across GPUs), a bf16
/// one (sliced across GPUs where `slice`), a table whose rows are looked up (held as the bf16 it is, an FP8 one decoded: exact), or none (another stage's embedding
/// or head).
#[derive(Clone, Debug, PartialEq)]
pub enum LinForm {
    Fp8 { rows: Option<(usize, usize)>, groups: usize },
    Bf16 { slice: bool },
    Table,
    Absent,
}

/// A matrix of the plan: tensor `name`, read as `form`, into the FP8 list where `f8` (the level holds it as released), else the matrices'.
#[derive(Clone, Debug, PartialEq)]
pub struct LinLoad {
    pub name: String,
    pub form: LinForm,
    pub f8: bool,
}

/// A mixture of experts of the plan: layer `layer`'s block, its roles and its kind.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeLoad {
    pub layer: usize,
    pub roles: Vec<(String, Source)>,
    pub moe: Moe,
}

/// What the loader reads for the lowering's indices, in their order: the matrices (`ModelDesc::embed` and `lm` the last two), the small tensors (each its pieces
/// one after another), the experts' blocks; the two rotary tables (the window-only layers', the compressing layers').
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub lins: Vec<LinLoad>,
    pub small: Vec<Vec<Piece>>,
    pub moe: Vec<MoeLoad>,
    pub ropes: Vec<Rope>,
}

impl Plan {
    fn small(&mut self, p: Vec<Piece>) -> usize {
        self.small.push(p);
        self.small.len() - 1
    }

    /// A quantized product's matrix (rows `rows` of `cols`, from row `r0`): its index in the list the level puts it in.
    fn lin(&mut self, n: &mut (usize, usize), vo: &V4Opts, name: String, (rows, cols, r0): (usize, usize, usize), part: Option<(usize, usize)>) -> Lin {
        let f8 = vo.f8(rows, cols, r0);
        self.lins.push(LinLoad { name, form: LinForm::Fp8 { rows: part, groups: 1 }, f8 });
        Lin::next(n, f8)
    }
}

/// A product's matrix: in the matrices' list, or the FP8 matrices'.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lin {
    Mat(usize),
    F8(usize),
}

impl Lin {
    /// The next index of the FP8 list (`f8`) or the matrices' (`n`: their counts so far).
    fn next(n: &mut (usize, usize), f8: bool) -> Lin {
        if f8 {
            n.1 += 1;
            Lin::F8(n.1 - 1)
        } else {
            n.0 += 1;
            Lin::Mat(n.0 - 1)
        }
    }

    fn op(self, x: Slot, y: Slot, name: &'static str) -> Op {
        match self {
            Lin::Mat(w) => Op::Linear { w, x, y, name },
            Lin::F8(w) => Op::F8Lin { w, x, y, name },
        }
    }
}

/// The checkpoint tensor of a block's role (one tensor).
fn one(roles: &[(String, Source)], r: &str, at: usize) -> Result<String> {
    match roles.iter().find(|(n, _)| n == r).map(|(_, s)| s) {
        Some(Source::Tensor(t)) => Ok(t.clone()),
        Some(s) => err(format!("block {at}: role {r} is {s:?}, not one tensor")),
        None => err(format!("block {at}: no role {r}")),
    }
}

fn has(roles: &[(String, Source)], r: &str) -> bool {
    roles.iter().any(|(n, _)| n == r)
}

/// The compressed attention's own state of each block that makes it: its entries' cache, their raw form (before the rotation) and the indexer's keys of them; and
/// the candidate blocks a publishing source keeps.
#[derive(Clone, Copy, Default)]
struct Made {
    cache: Option<usize>,
    raw: Option<usize>,
    keys: Option<usize>,
    publish: Option<(usize, usize)>,
}

/// The executor's description of a model of compressed attention layers for stage `st` (`vo`: what the level and the switches make of its products and caches),
/// and its loader's plan. Each layer is an optional n-gram block, the compressed attention, the experts, each attention and experts block on the streams.
pub fn lower(d: &Described, eos: Vec<i64>, max_pos: usize, st: &Stage, vo: &V4Opts) -> Result<(ModelDesc, Plan)> {
    let m = &d.desc;
    let what = "a compressed attention model";
    // (the release's quantizations the kernels simulate are those of FP8 in blocks of 128: activations in 128s before each quantized product, the window's and
    // the compressed entries' non-rotary channels FP8 in 64s, the indexer's FP4 after its Hadamard rotation. A release quantizing otherwise (DeepSeek-V4.1's
    // blocks of 32: activations in 32s, its window FP8 whole, its entries FP4 in 16s, its indexer FP4 unrotated) runs without them: its weights' arithmetic)
    let sim = vo.sim && d.fp8_block.is_none_or(|b| b == 128);
    // (and the library's FP8 products take blocks of 128 alone: a release of other blocks has its FP8 matrices held as the bf16 they decode to)
    let exact8 = vo.exact8 && d.fp8_block.is_none_or(|b| b == 128);
    let vo = &V4Opts { sim, exact8, ..*vo };
    // the layers' blocks: (an n-gram block, the attention, the experts)
    let mut at = Vec::new();
    let mut i = 0;
    while i < m.blocks.len() {
        let ng = matches!(m.blocks[i].body, Body::Ngram(_)).then_some(i);
        let a = i + ng.is_some() as usize;
        match (m.blocks.get(a).map(|b| &b.body), m.blocks.get(a + 1).map(|b| &b.body)) {
            (Some(Body::Mixer(Mixer::Compressed(_))), Some(Body::Ffn(Ffn::Moe(_)))) => at.push((ng, a, a + 1)),
            (x, y) => return err(format!("{what}: blocks {a} and {} are {x:?} and {y:?}, where a layer is a compressed attention then its experts", a + 1)),
        }
        i = a + 2;
    }
    let comp = |b: usize| match &m.blocks[b].body {
        Body::Mixer(Mixer::Compressed(c)) => c,
        _ => unreachable!("checked above"),
    };
    let streams = |b: usize| -> Result<StreamsP> {
        match m.blocks[b].residual {
            Residual::Streams(s) => Ok(s),
            ref r => err(format!("{what}: block {b}'s residual is {r:?}, not the streams'")),
        }
    };
    let c0 = comp(at.first().ok_or_else(|| Error(format!("{what}: no layers")))?.1).clone();
    let s0 = streams(at[0].1)?;
    let (d_, nh, hd, ql, og, ol) = (m.hidden, c0.heads, c0.head_dim, c0.q_lora, c0.o_groups, c0.o_lora);
    let (ih, idim, itop) =
        c0.indexer.as_ref().or_else(|| at.iter().find_map(|x| comp(x.1).indexer.as_ref())).map_or((0, 0, 0), |x| (x.heads, x.head_dim, x.top_k));
    if hd != 512 || c0.rope.dim != 64 || (ih > 0 && idim != 128) || s0.streams != 4 || d_ % 512 != 0 || ql % 128 != 0 || ol % 128 != 0 {
        return err(format!(
            "{what}: heads of {hd} ({} rotary), indexer heads of {idim}, {} streams, widths {d_} / {ql} / {ol}: this engine runs heads of 512 (64 rotary), indexer heads of 128, 4 streams and widths of 128s",
            c0.rope.dim, s0.streams
        ));
    }
    if ih > 64 {
        return err(format!("{what}: {ih} indexer heads: this engine runs up to 64"));
    }
    if s0.lagged && st.layers.0 > 0 {
        return err(format!("{what}: lagged streams on a stage after the first: the previous stage's block weights are not handed over"));
    }
    let hpg = nh / og.max(1);
    let wo_cols = nh * hd / og.max(1);
    let (mut plan, mut n, mut rings, mut states, mut layers) = (Plan::default(), (0usize, 0usize), Vec::new(), Vec::new(), Vec::new());
    let mut made = vec![Made::default(); m.blocks.len()];
    // (the block whose pick ran last: the picks' buffer holds its rows)
    let mut picked_by = None;
    let (mut first_lag, mut n_moe, mut overlap_any, mut engram_w) = (true, 0usize, false, (0usize, 0usize));
    // (the smallest ratio a query reads every entry of, and an indexer's smallest: the model's, whatever the stage, as the buffers are sized by them)
    let min_nz = |a: usize, b: usize| if a == 0 { b } else { a.min(b) };
    let (mut all_ratio, mut idx_ratio) = (0usize, 0usize);
    for c in at.iter().map(|x| comp(x.1)).filter(|c| c.ratio > 0) {
        if c.indexer.is_some() {
            idx_ratio = min_nz(idx_ratio, c.ratio);
        } else {
            all_ratio = min_nz(all_ratio, c.ratio);
        }
    }
    for (l, &(ng, ab, eb)) in at.iter().enumerate() {
        let c = comp(ab);
        // (the widths the executor sizes its buffers by are one model's: every layer's the first's)
        if (c.heads, c.head_dim, c.q_lora, c.o_groups, c.o_lora, c.window, c.rope.dim) != (nh, hd, ql, og, ol, c0.window, 64)
            || c.indexer.as_ref().is_some_and(|x| (x.heads, x.head_dim, x.top_k) != (ih, idim, itop))
        {
            return err(format!("{what}: layer {l}'s attention is of other widths than layer 0's: {c:?}"));
        }
        // the rotary tables: the window-only layers' first, the compressing layers' second, each the same for every layer of its kind
        let rope = (c.ratio != 0) as usize;
        while plan.ropes.len() <= rope {
            plan.ropes.push(c0.rope.clone());
        }
        match at[..l].iter().map(|x| comp(x.1)).find(|x| (x.ratio != 0) as usize == rope) {
            Some(p) if p.rope != c.rope => return err(format!("{what}: layer {l}'s rotary is not that of the layers of its kind before it")),
            None => plan.ropes[rope] = c.rope.clone(),
            _ => {}
        }
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        // (a part shared from a layer of another stage is not handed over)
        for b in c.share.iter().flat_map(|sh| [sh.entries, sh.picks, sh.candidates]).flatten() {
            if at.iter().position(|x| x.1 == b).is_none_or(|src| src < st.layers.0) {
                return err(format!("{what}: layer {l} reads block {b}'s compressed attention, which this stage does not run"));
            }
        }
        let mut ops = Vec::new();
        // an n-gram block before the layer, on the streams
        if let Some(b) = ng {
            let Body::Ngram(g) = &m.blocks[b].body else { unreachable!("checked above") };
            ops.extend(ngram(&mut plan, &mut n, &mut states, vo, g, &d.blocks[b], b, d_)?);
            engram_w = (engram_w.0.max(g.dim), engram_w.1.max((g.streams + 1) * d_));
        }
        let (sa, sf) = (streams(ab)?, streams(eb)?);
        let roles = &d.blocks[ab];
        let hc = |plan: &mut Plan, roles: &[(String, Source)], b: usize| -> Result<[usize; 3]> {
            Ok([
                plan.small(vec![Piece::Tensor(one(roles, "hc_fn", b)?)]),
                plan.small(vec![Piece::Tensor(one(roles, "hc_base", b)?)]),
                plan.small(vec![Piece::Tensor(one(roles, "hc_scale", b)?)]),
            ])
        };
        let mut mix = |[f, base, scale]: [usize; 3], s: &StreamsP| {
            if s.lagged {
                let first = std::mem::replace(&mut first_lag, false);
                StreamsMix::HcLagged { f, base, scale, first }
            } else {
                StreamsMix::Hc { f, base, scale }
            }
        };
        let h = hc(&mut plan, roles, ab)?;
        let t = |r: &str| one(roles, r, ab).map(|x| vec![Piece::Tensor(x)]);
        let attn_norm = plan.small(t("norm_in")?);
        let q_norm = plan.small(t("q_a_norm")?);
        let kv_norm = plan.small(t("kv_norm")?);
        let sink = plan.small(t("sinks")?);
        let wq_a = plan.lin(&mut n, vo, one(roles, "q_a", ab)?, (ql, d_, 0), None);
        let wq_b = plan.lin(&mut n, vo, one(roles, "q_b", ab)?, (nh * hd, ql, 0), None);
        let wkv = plan.lin(&mut n, vo, one(roles, "kv", ab)?, (hd, d_, 0), None);
        let stack = vo.stack(og * ol, wo_cols, ol);
        let (wo_a0, wo_a_f8) = if stack {
            plan.lins.push(LinLoad { name: one(roles, "o_a", ab)?, form: LinForm::Fp8 { rows: None, groups: og }, f8: true });
            n.1 += 1;
            (n.1 - 1, true)
        } else {
            let oa = one(roles, "o_a", ab)?;
            let first: Vec<Lin> = (0..og).map(|g| plan.lin(&mut n, vo, oa.clone(), (ol, wo_cols, g * ol), Some((g * ol, (g + 1) * ol)))).collect();
            match first[0] {
                Lin::F8(i) if first.iter().enumerate().all(|(g, x)| *x == Lin::F8(i + g)) => (i, true),
                Lin::Mat(i) if first.iter().enumerate().all(|(g, x)| *x == Lin::Mat(i + g)) => (i, false),
                _ => return err(format!("{what}: layer {l}'s output groups: their matrices are not one after another in one list")),
            }
        };
        let wo_b = plan.lin(&mut n, vo, one(roles, "o_b", ab)?, (d_, og * ol, 0), None);
        let ring = rings.len();
        rings.push(Ring::keys(hd, 1, c.window));
        ops.extend([
            Op::StreamsPre { mix: mix(h, &sa), norm: attn_norm, round: sa.rounding },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            wq_a.op(Slot::Hq, Slot::Proj, "attention: q down"),
            Op::RmsNorm { x: Slot::Proj, w: q_norm, y: Slot::Qr, weight: NormW::Plain, eps: c.norm_eps },
            Op::Fp8 { x: Slot::Qr, y: Slot::Proj, block: 128 },
            wq_b.op(Slot::Proj, Slot::Qo, "attention: q up"),
            Op::CompQ { rope, heads: nh, norm: c.q_head_norm },
            wkv.op(Slot::Hq, Slot::Kv, "attention: kv"),
            Op::CompKv { norm: kv_norm, rope, ring },
        ]);
        let ratio = c.ratio;
        let wscale = (idim as f32).powf(-0.5) * (ih as f32).powf(-0.5);
        // a compressor of the layer's own (`pre`'s roles: the products of its tokens, their gate, the window bias, the norm) at width `dd`: its op and its entries' ring (paged as
        // any keys; its raw entries' a ring of the same group), its state the open windows' rows
        let compressor =
            |plan: &mut Plan, rings: &mut Vec<Ring>, states: &mut Vec<usize>, pre: &str, dd: usize, indexer: bool, raw: bool| -> Result<(Op, usize)> {
                let coff = 1 + c.overlap as usize;
                let w = coff * dd;
                // (a window of one token has no gate: zeros, whose softmax over the one token is 1; a window with no bias, zeros)
                let gate = if ratio > 1 || has(roles, &format!("{pre}_gate")) {
                    Piece::Tensor(one(roles, &format!("{pre}_gate"), ab)?)
                } else {
                    Piece::Zeros(w * d_ * 2)
                };
                let wb = plan.small(vec![Piece::Tensor(one(roles, &format!("{pre}_kv"), ab)?), gate]);
                let ape = plan.small(vec![if c.ape { Piece::Tensor(one(roles, &format!("{pre}_ape"), ab)?) } else { Piece::Zeros(ratio * w * 4) }]);
                let norm = plan.small(vec![Piece::Tensor(one(roles, &format!("{pre}_norm"), ab)?)]);
                let (cache, state) = (rings.len(), states.len());
                rings.push(Ring::entries(dd, ratio));
                states.push(coff * ratio * coff * dd * 4);
                states.push(coff * ratio * coff * dd * 4);
                let raw = raw.then(|| {
                    rings.push(Ring::entries(dd, ratio));
                    rings.len() - 1
                });
                Ok((Op::Compress { w: wb, ape, norm, d: dd, ratio, overlap: c.overlap, rope, cache, state, indexer, raw }, cache))
            };
        let mut cache = None;
        // (DeepSeek-V4's indexer: its own compressor over windows that overlap, its own queries)
        if let (None, Some(_)) = (&c.share, &c.indexer) {
            if !c.overlap || !c.ape {
                return err(format!("{what}: layer {l} has an indexer of its own over windows that do not overlap: no operation of this executor"));
            }
            let iq = plan.lin(&mut n, vo, one(roles, "idx_q_b", ab)?, (ih * idim, ql, 0), None);
            plan.lins.push(LinLoad { name: one(roles, "idx_weights", ab)?, form: LinForm::Bf16 { slice: true }, f8: false });
            let iw = Lin::next(&mut n, false);
            let (ic, icache) = compressor(&mut plan, &mut rings, &mut states, "idx", idim, true, false)?;
            ops.extend([
                iq.op(Slot::Proj, Slot::Iq, "indexer: q"),
                Op::IdxQ { rope, heads: ih },
                ic,
                iw.op(Slot::H, Slot::Iw, "indexer: weights"),
                Op::IdxPick(pick(ih, ratio, itop, wscale, icache, None)),
            ]);
            picked_by = Some(ab);
        }
        if ratio > 0 && c.share.is_none_or(|s| s.entries.is_none()) {
            // (a V4.1 kv source with the indexer's key product: its entries before their rotation kept for the keys)
            let keys = c.share.is_some() && has(roles, "idx_k");
            let (mc, k) = compressor(&mut plan, &mut rings, &mut states, "comp", hd, false, keys)?;
            let Op::Compress { raw, .. } = mc else { unreachable!() };
            cache = Some(k);
            ops.push(mc);
            made[ab].cache = Some(k);
            made[ab].raw = raw;
            if let Some(raw) = raw {
                let w = plan.small(t("idx_k")?);
                let norm = plan.small(t("idx_k_norm")?);
                let keys = rings.len();
                rings.push(Ring::entries(idim, ratio));
                ops.push(Op::IdxKeys { w, norm, rope, ratio, raw, keys });
                made[ab].keys = Some(keys);
            }
            overlap_any |= c.overlap;
        }
        if let Some(sh) = &c.share {
            if ratio == 0 {
                return err(format!("{what}: layer {l} shares a compressed attention's parts and compresses nothing"));
            }
            if let Some(src) = sh.entries {
                let s = made.get(src).copied().unwrap_or_default();
                cache = Some(s.cache.ok_or_else(|| Error(format!("{what}: layer {l} reads the entries of block {src}, which makes none")))?);
                made[ab].keys = s.keys;
            }
            let Some(ix) = &c.indexer else { return err(format!("{what}: layer {l} shares its entries and has no indexer: no operation of this executor")) };
            match sh.picks {
                // (an index source: its own queries and weights over the entries' keys)
                None => {
                    let keys = made[ab].keys.ok_or_else(|| Error(format!("{what}: layer {l} picks from entries that have no indexer keys")))?;
                    let cand = match (sh.publish, sh.candidates) {
                        (Some((block, top)), _) => Some(Cand::Publish { block, top }),
                        (None, Some(src)) => {
                            let (block, top) = made
                                .get(src)
                                .and_then(|s| s.publish)
                                .ok_or_else(|| Error(format!("{what}: layer {l} reads the candidate blocks of block {src}, which publishes none")))?;
                            Some(Cand::Inside { block, top })
                        }
                        (None, None) => None,
                    };
                    made[ab].publish = sh.publish;
                    let iq = plan.lin(&mut n, vo, one(roles, "idx_q_b", ab)?, (ih * idim, ql, 0), None);
                    plan.lins.push(LinLoad { name: one(roles, "idx_weights", ab)?, form: LinForm::Bf16 { slice: true }, f8: false });
                    let iw = Lin::next(&mut n, false);
                    ops.extend([
                        iq.op(Slot::Proj, Slot::Iq, "indexer: q"),
                        Op::IdxQ { rope, heads: ih },
                        iw.op(Slot::H, Slot::Iw, "indexer: weights"),
                        Op::IdxPick(pick(ih, ratio, ix.top_k, wscale, keys, cand)),
                    ]);
                    picked_by = Some(ab);
                }
                Some(src) if picked_by == Some(src) => {}
                Some(src) => return err(format!("{what}: layer {l} reads block {src}'s picks, and the latest pick before it is block {picked_by:?}'s")),
            }
        }
        ops.extend([
            Op::CompAttn { ring, cache, ratio, picked: c.indexer.is_some(), sink, rope, heads: nh, hpg },
            if stack {
                Op::OGroupLoraStack { w: wo_a0, groups: og, ol, y: Slot::M, fp8: true }
            } else {
                Op::OGroupLora { w: wo_a0, in_f8: wo_a_f8, groups: og, hpg, ol, y: Slot::M, fp8: true }
            },
            wo_b.op(Slot::M, Slot::O, "attention: o"),
            Op::StreamsPost { round: sa.rounding, gated: false },
        ]);
        // the experts
        let Body::Ffn(Ffn::Moe(e)) = &m.blocks[eb].body else { unreachable!("checked above") };
        let er = &d.blocks[eb];
        let f = hc(&mut plan, er, eb)?;
        let ffn_norm = plan.small(vec![Piece::Tensor(one(er, "norm_in", eb)?)]);
        let block = n_moe;
        n_moe += 1;
        plan.moe.push(MoeLoad { layer: l, roles: er.clone(), moe: e.clone() });
        let hash = if e.router.hash { Some(plan.small(vec![Piece::Tensor(one(er, "hash", eb)?)])) } else { None };
        ops.extend([
            Op::StreamsPre { mix: mix(f, &sf), norm: ffn_norm, round: sf.rounding },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            Op::Moe(MoeOp { x: Slot::Hq, hash, weigh_act: true, fp8: vo.sim, logit_scale: e.router.logit_scale, ..MoeOp::of(block) }),
            Op::StreamsPost { round: sf.rounding, gated: false },
        ]);
        layers.push(ops);
    }
    // the ends: the embedding where this stage is the first, the head where the last (another stage's are empty places), the head's collapse and the final norm
    let (embed, lm) = (n.0, n.0 + 1);
    plan.lins.push(LinLoad { name: d.embed.clone(), form: if st.first { LinForm::Bf16 { slice: false } } else { LinForm::Absent }, f8: false });
    plan.lins.push(LinLoad {
        name: d.head.clone().unwrap_or_else(|| d.embed.clone()),
        form: if st.last { LinForm::Bf16 { slice: true } } else { LinForm::Absent },
        f8: false,
    });
    if st.last {
        let x = |r: &str| -> Result<String> {
            d.extra
                .iter()
                .find(|(n, _)| n == r)
                .and_then(|(_, s)| if let Source::Tensor(t) = s { Some(t.clone()) } else { None })
                .ok_or_else(|| Error(format!("{what}: no model role {r}")))
        };
        let mix = match m.head.collapse {
            Collapse::Weighted { .. } => {
                let h = [x("out_hc_fn")?, x("out_hc_base")?, x("out_hc_scale")?].map(|t| plan.small(vec![Piece::Tensor(t)]));
                HeadMix::Weighted(h)
            }
            Collapse::LastPre if s0.lagged => HeadMix::LastPre,
            c => return err(format!("{what}: the head's collapse {c:?} on streams of {s0:?}")),
        };
        let norm = plan.small(vec![Piece::Tensor(d.final_norm.clone())]);
        layers.last_mut().ok_or_else(|| Error(format!("{what}: no layers")))?.push(Op::StreamsHead { mix, norm, round: s0.rounding });
    }
    let mut widths = [0usize; SLOTS];
    let wmax = if overlap_any || at.iter().any(|x| comp(x.1).overlap) { 2 * hd } else { hd };
    for (sl, w) in [
        (Slot::X, d_),
        (Slot::S, 4 * d_),
        (Slot::H, d_),
        (Slot::Hq, d_),
        (Slot::Proj, ql),
        (Slot::Qr, ql),
        (Slot::Qo, nh * hd),
        (Slot::Kv, hd),
        (Slot::Cp, 4 * wmax),
        (Slot::Iq, ih * idim),
        (Slot::Iw, ih),
        (Slot::Att, nh * hd),
        (Slot::Mid, og * ol),
        (Slot::M, og * ol),
        (Slot::O, d_),
    ] {
        widths[sl as usize] = w;
    }
    // (an n-gram block's rows of its tables, its key and value product, its gated value and that normed)
    if engram_w.0 > 0 {
        for (sl, w) in [(Slot::In, engram_w.0), (Slot::Gu, engram_w.1), (Slot::D, 4 * d_), (Slot::M, (og * ol).max(4 * d_))] {
            widths[sl as usize] = w;
        }
    }
    let desc = ModelDesc {
        family: "DeepSeek-V4",
        hidden: d_,
        vocab: m.vocab,
        eps: m.blocks[at[0].1].norm_in.eps,
        max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: Op::Streams,
        layers,
        states,
        final_norm: None,
        decoded: false,
        skinny: false,
        embed,
        lm,
        softcap: 0.0,
        eos,
        post: None,
        hc: (s0.eps, s0.sinkhorn_iters as i32),
        logit_mult: 0.0,
        host: Vec::new(),
        v4: Some(V4Dims { heads: nh, idx_heads: ih, topk: itop, window: c0.window, all_ratio, sim: vo.sim, o_groups: og, idx_ratio }),
    };
    Ok((desc, plan))
}

/// The pick of an indexer of `heads` over entries of `ratio` tokens (state `keys`), its `top_k` best kept as entries, its scores the release's bf16 arithmetic.
fn pick(heads: usize, ratio: usize, top_k: usize, wscale: f32, keys: usize, cand: Option<Cand>) -> IdxPick {
    IdxPick { heads, ratio, top_k, expand: false, tail: false, wscale, score: IdxScore::Release, keys: IdxKeys::Entries(keys), weighted: true, cand }
}

/// An n-gram block on the streams (`Body::Ngram`) of block `b`: its rows of the tables into In, their key and value product into Gu, the gated add into the streams.
#[allow(clippy::too_many_arguments)]
fn ngram(
    plan: &mut Plan,
    n: &mut (usize, usize),
    states: &mut Vec<usize>,
    vo: &V4Opts,
    g: &Ngram,
    roles: &[(String, Source)],
    b: usize,
    d: usize,
) -> Result<Vec<Op>> {
    if g.order < 2 || g.order > 4 || g.streams != 4 || g.dim % ((g.order - 1) * g.heads) != 0 || g.dim > 8192 {
        return err(format!("block {b}: an n-gram block of {g:?}: the kernels take orders 2 to 4 over four streams"));
    }
    let mult: Vec<i64> = crate::runtime::reference::ngram::multipliers(g)?.into_iter().map(|x| x as i64).collect();
    let (sizes, offsets) = crate::runtime::reference::ngram::tables(g);
    let i64s = |v: Vec<u64>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    let hash = [plan.small(vec![Piece::I64(mult)]), plan.small(vec![Piece::I64(i64s(sizes))]), plan.small(vec![Piece::I64(i64s(offsets))])];
    let map = (g.form == NgramForm::Engram).then(|| plan.small(vec![Piece::TokenMap]));
    plan.lins.push(LinLoad { name: one(roles, "ngram", b)?, form: LinForm::Table, f8: false });
    let table = Lin::next(n, false);
    let Lin::Mat(table) = table else { unreachable!() };
    let kv = plan.lin(n, vo, one(roles, "kv", b)?, ((g.streams + 1) * d, g.dim, 0), None);
    // (Engram's gate reads its plain weights in fp32)
    let w = |r: &str| one(roles, r, b).map(|t| if g.form == NgramForm::Engram { Piece::Fp32(t) } else { Piece::Tensor(t) });
    let q = plan.small(vec![w("norm_in")?]);
    let k = plan.small(vec![w("norm_key")?]);
    let hist = states.len();
    states.push((g.order - 1) * 4);
    let conv = if g.conv > 0 {
        let norm = plan.small(vec![Piece::Tensor(one(roles, "norm_conv", b)?)]);
        let taps = plan.small(vec![Piece::Tensor(one(roles, "conv", b)?)]);
        states.push((g.conv - 1) * g.dilation * g.streams * d * 2);
        Some(NgramConv { norm, taps, state: states.len() - 1, taps_n: g.conv, dilation: g.dilation })
    } else {
        None
    };
    Ok(vec![
        Op::NgramRows(NgramRows {
            table,
            hash,
            hist,
            per: g.heads,
            row: g.dim / ((g.order - 1) * g.heads),
            eos: g.eos,
            order: g.order,
            map,
            segments: g.form == NgramForm::Segmented,
        }),
        kv.op(Slot::In, Slot::Gu, "n-gram block: key and value product"),
        Op::NgramAdd(NgramAdd { q, k, conv, plus_one: g.norm.weight == NormWeight::PlusOne, eps: g.norm.eps }),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn described(rel: &str) -> Described {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        if let Some(t) = v.get("config").and_then(|c| c.as_str()) {
            v = serde_json::from_str(t).unwrap();
        }
        crate::runtime::describe::describe(&v).unwrap_or_else(|e| panic!("{}", e.msg))
    }

    const VO: V4Opts = V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 256 };

    /// The tiny DeepSeek-V4.1 (ratios 0, 2, 2, 1, 1, 1; kv sources 1, 3, 5; index sources 1-4; candidate source 3; Engram of orders 2 to 4 before layers 1 and 4) lowers
    /// to V4.1's forms of the operations: lagged streams (one-hot before the first block alone) and the last weights' collapse; compressors on the kv sources alone,
    /// the indexer's keys on those that are index sources too; picks on the index sources, published by layer 3, read inside its blocks by layer 4; every
    /// compressing layer's attention over the picks, no head norm; Engram's rows through the token map; the router's logits divided by the gate temperature; and
    /// no simulated quantization (its release's FP8 blocks of 32 are not those the kernels simulate) where the options ask for it.
    #[test]
    fn v41s_forms_lower_to_their_operations() {
        let d = described("tests/fixtures/deepseek_v41_tiny_config.json");
        let (m, plan) = lower(&d, vec![1], 4096, &Stage::whole(6), &VO).unwrap_or_else(|e| panic!("{}", e.msg));
        let lag: Vec<bool> = m
            .layers
            .iter()
            .flatten()
            .filter_map(|op| if let Op::StreamsPre { mix: StreamsMix::HcLagged { first, .. }, .. } = op { Some(*first) } else { None })
            .collect();
        assert_eq!(lag.len(), 12);
        assert!(lag[0] && lag[1..].iter().all(|f| !f));
        assert!(matches!(m.layers[5].last(), Some(Op::StreamsHead { mix: HeadMix::LastPre, .. })));
        let on = |l: usize, f: &dyn Fn(&Op) -> bool| m.layers[l].iter().filter(|op| f(op)).count();
        for l in 0..6 {
            let comp = on(l, &|op| matches!(op, Op::Compress { .. }));
            let keys = on(l, &|op| matches!(op, Op::IdxKeys { .. }));
            let picks = on(l, &|op| matches!(op, Op::IdxPick(_)));
            let ng = on(l, &|op| matches!(op, Op::NgramRows(NgramRows { map: Some(_), segments: false, order: 4, .. })));
            assert_eq!((comp, keys, picks, ng), ([0, 1, 0, 1, 0, 1][l], [0, 1, 0, 1, 0, 0][l], [0, 1, 1, 1, 1, 0][l], [0, 1, 0, 0, 1, 0][l]), "layer {l}");
            assert_eq!(on(l, &|op| matches!(op, Op::CompAttn { picked: true, .. })), (l > 0) as usize, "layer {l}");
            assert_eq!(on(l, &|op| matches!(op, Op::CompQ { norm: false, .. })), 1);
        }
        let cand = |l: usize| m.layers[l].iter().find_map(|op| if let Op::IdxPick(p) = op { Some(p.cand) } else { None }).flatten();
        assert_eq!((cand(1), cand(2), cand(3), cand(4)), (None, None, Some(Cand::Publish { block: 4, top: 3 }), Some(Cand::Inside { block: 4, top: 3 })));
        let Some(Op::Moe(e)) = m.layers[0].iter().find(|op| matches!(op, Op::Moe(_))) else { panic!() };
        assert_eq!(e.logit_scale, 1.0);
        // (every index the operations read is the plan's)
        assert_eq!(plan.lins.len(), m.lm + 1 + plan.lins.iter().filter(|x| x.f8).count());
        assert_eq!(plan.moe.len(), 6);
        assert_eq!(m.v4.as_ref().map(|v| (v.all_ratio, v.idx_ratio, v.sim)), Some((0, 1, false)));
        assert!(VO.sim && m.layers.iter().flatten().all(|op| !matches!(op, Op::Moe(MoeOp { fp8: true, .. }))));
    }

    /// Every tensor the plan reads is one the description says (its weight map: V4.1's shapes; DeepSeek-V4-Flash-0731's recorded header), the quantized ones with
    /// their scales.
    #[test]
    fn the_plans_tensors_are_the_checkpoints() {
        let hdr: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/DeepSeek-V4-Flash-0731.json")).unwrap(),
        )
        .unwrap();
        // (the header's groups of tensors: a name a layer where it says {L}, an expert where {E})
        let mut names = std::collections::BTreeSet::new();
        for g in hdr["groups"].as_array().unwrap() {
            let mask = g["mask"].as_str().unwrap();
            let layers: Vec<String> = g["layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap().to_string()).collect();
            for l in if mask.contains("{L}") { layers } else { vec![String::new()] } {
                for e in 0..g["experts"].as_u64().unwrap_or(0).max(1) {
                    names.insert(mask.replace("{L}", &l).replace("{E}", &e.to_string()));
                }
            }
        }
        for (rel, layers, known) in
            [("tests/fixtures/deepseek_v41_tiny_config.json", 6, None), ("tests/fixtures/headers/DeepSeek-V4-Flash-0731.json", 43, Some(&names))]
        {
            let d = described(rel);
            let have = |t: &str| known.map_or_else(|| d.shapes.contains_key(t), |k| k.contains(t));
            let (_, plan) = lower(&d, vec![1], 4096, &Stage::whole(layers), &VO).unwrap_or_else(|e| panic!("{rel}: {}", e.msg));
            for p in plan.small.iter().flatten() {
                if let Piece::Tensor(t) | Piece::Fp32(t) = p {
                    assert!(have(t), "{rel}: {t}");
                }
            }
            for x in plan.lins.iter().filter(|x| x.form != LinForm::Absent) {
                assert!(have(&x.name), "{rel}: {}", x.name);
                if x.f8 {
                    assert!(d.scales.contains_key(&x.name), "{rel}: {} has no scales", x.name);
                }
            }
        }
    }
}

#[cfg(test)]
mod v4_today {
    use super::*;

    /// FNV-1a of a description's text.
    fn digest(m: &ModelDesc) -> u64 {
        format!("{m:?}").bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    }

    /// DeepSeek-V4-Flash-0731 lowered at each level's options, whole and as the two halves of a split, is the executor's description it was before V4.1's forms
    /// (recorded 2026-11-25 from the family's own reader of the config with its entries in paged rings, which the lowering then equalled to the byte on this
    /// config and on the random V4, and which this replaces).
    #[test]
    fn deepseek_v4s_lowering_is_the_one_it_had() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/headers/DeepSeek-V4-Flash-0731.json");
        let h: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let d = crate::runtime::describe::describe(&serde_json::from_str(h["config"].as_str().unwrap()).unwrap()).unwrap();
        let base = V4Opts { exact8: true, f8_min: 16 << 20, ostack: true, sim: true, ctx: 4096 };
        let mut got = Vec::new();
        for vo in [base, V4Opts { exact8: false, ..base }, V4Opts { ostack: false, ..base }, V4Opts { f8_min: 0, sim: false, ..base }] {
            for st in [Stage::whole(43), Stage { layers: (0, 21), first: true, last: false }, Stage { layers: (21, 43), first: false, last: true }] {
                got.push(digest(&lower(&d, vec![1], 1 << 20, &st, &vo).unwrap().0));
            }
        }
        let want: [u64; 12] = [
            6274911006358578454,
            7960358971613121486,
            2375456957471879598,
            9886925887969473836,
            13957852289930100916,
            11897994103295081470,
            11060615554011635558,
            9430456114592192634,
            3180334719980402341,
            7432593811500680815,
            1689972873128606854,
            2749147499816240112,
        ];
        assert_eq!(got, want, "{got:#x?}");
    }
}
