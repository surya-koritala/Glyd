//! Kimi-K3 (src/kimi.rs; the math: research/kimi-k3.md) as a description. No residual stream: the partial sum of the current block of layers is X, the completed
//! blocks (the embedding first) are Blk's, and every block's input is their depth mix and norm (`Op::DepthMix`, the layer before's add in it). The delta
//! attention is GLM's operations (q, k, v, the full-rank output gate, the forget gate's first product and beta one product of the layer's input, read where it
//! leaves them: `View`); the latent attention the layer kind's (`Op::Mla`, src/dynamic/mla.rs: q_a, the gate and kv_a one product; the absorbed query with the key part
//! the heads share, a dense attention, the sigmoid output gate); the experts a block on the mixture-of-experts core (MXFP4 as released, packed exactly as DeepSeek-V4's
//! FP4, situ, on the latent, its sum normed before the up projection), the shared experts the core's shared MLP; layer 0's dense MLP with situ. Norms are
//! KimiRMSNorm's (two roundings). Loaded by the executor's loader (`Dyn::new_placed`): the lowered description (`lower::kimi`) names every matrix and small
//! tensor, and each layer's block of experts is made from its roles (`block`). The CPU reference (src/kimi_ref.rs) checks it (`kimicheck`).

use super::experts::{latent_block, Guess, KernLatent, LatentHeld, GUESS_COPIES};
use super::lower::fused;
use super::{DepthMixOp, Dyn, MlaIn, MlaOp, ModelDesc, MoeOp, Op, Opts, Placing, Slot, Stage, View};
use crate::dev::Dev;
use crate::dynamic::experts::MoeBlock;
use crate::ffi::{err, Error, Result};
use crate::glm::GlmKind;
use crate::kimi::ConfigKimi;
use crate::moe::Act;
use crate::offload::ExpertStore;
use crate::runtime::config::ModelConfig;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;
use std::rc::Rc;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The GPU memory one of the experts takes (a place of the expert cache): its first and second matrices in the larger of the two FP4 layouts a layer may be packed
/// in (packed scales where its scales allow, else as released: about 4.5% more).
pub fn expert_bytes(c: &ConfigKimi) -> Result<u64> {
    let e = &c.experts;
    let one = |tag: u32| -> Result<u64> {
        let ((ud, us), (dd, ds)) = (crate::moe::expert_strides(tag, 2 * e.inter, e.latent)?, crate::moe::expert_strides(tag, e.latent, e.inter)?);
        Ok(ud + us + dd + ds)
    };
    Ok(one(glyd_gpu::pack::FP4R_TAG)?.max(one(glyd_gpu::pack::FP4_TAG)?))
}

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_stage(dir, o, Stage::whole(ConfigKimi::load(dir)?.layers))
}

/// Stage `st` of the model (its layers; the embedding where it is the first, the final mix, norm and head where the last), every expert on the GPU.
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    load_with_places(dir, &ModelConfig::load(dir)?, o, st, None)
}

/// The dynamic route's stage (src/dynamic/split.rs `StageLoad`): `load_stage_hb` with the caching given; with none, a development build's GLYD_KIMI_PLACES (the
/// experts of a layer on the GPU, the rest in the expert cache's host tier: `load_with_places`).
pub fn stage(dir: &Path, m: &ModelConfig, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    if caching.is_some() {
        return load_stage_hb(dir, m, o, st, caching);
    }
    load_with_places(dir, m, o, st, crate::tune!("GLYD_KIMI_PLACES").and_then(|v| v.parse::<usize>().ok()))
}

/// Stage `st` with `places` of each mixture-of-experts layer's experts on the GPU (None, or as many as it has: all; fewer: the expert cache, its host tier pinned),
/// nothing else of the expert cache asked: `load_stage_hb`'s.
pub fn load_with_places(dir: &Path, m: &ModelConfig, o: &Opts, st: Stage, places: Option<usize>) -> Result<Dyn> {
    let caching = places.map(|places| crate::hbg_store::HbCaching { places, ram: None, offload: Default::default(), dense: 0 });
    load_stage_hb(dir, m, o, st, caching.as_ref())
}

/// Stage `st` with its experts as `caching` says, where it is given (the stage's places and memory, src/hbg_store.rs): a hummingbird directory's in the expert
/// cache, the release's in the cache's places (`places` of each layer's, every expert in the computer's memory) where the directory holds no hummingbird
/// experts. Loaded by the executor's loader (`Dyn::new_placed`) from the lowered description, which names every matrix and small tensor (`lower::kimi`);
/// each layer's experts are the Moe kind's latent block (`experts::latent_block`). Every layer is across the thread's group of GPUs where it has one
/// (src/dynamic/tp.rs): this GPU's slice of a big matrix's rows, of each expert's; the stage's last layers' matrices in the computer's memory where the cache
/// says so (src/dense_ram.rs).
pub fn load_stage_hb(dir: &Path, m: &ModelConfig, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's; the cache's places are the release's where
    // there is none)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let places = caching.map(|c| c.places).filter(|_| hb.is_none());
    let c = m.read_with(dir, ConfigKimi::of)?;
    let (kd, md) = (&c.kda, &c.mla);
    if kd.dim != crate::kern::GM_D || c.hidden % 64 != 0 || c.hidden > 8192 || md.kv_rank > 512 || (md.kv_rank + md.rope) % 64 != 0 {
        return err(format!("delta heads of {}, a width of {}, latents of {} + {}: this engine runs heads of 128, widths of 64s up to 8,192 and latents up to 512 (with the shared part, in 64s)", kd.dim, c.hidden, md.kv_rank, md.rope));
    }
    let dev = Dev::new(o.device)?;
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let w32 = dtypes(&io(Checkpoint::open(dir))?, &c)?;
    let hb_store = caching.map(crate::hbg_store::store_for);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let store = ExpertStore::new();
    // (the host's misses on a crew of 64 threads at most, each kept to a core: a decode step's call of one to three of Kimi-K3's 17 MB experts took 4.2 ms on all
    // 240 of an 8x A100 box's cores, 0.6 ms on 64 (fp4bench); a development build's GLYD_KIMI_HOST: the mode, copy, cpu, split or auto (the default))
    // (the experts in the computer's memory: each expert's bytes dropped from the page cache once they are read and packed, so that each layer's 15 GB of
    // pinned memory comes from what the layers before freed; the download's cache had made each layer wait on reclaim and compaction, most of a 45-minute
    // load. Not the whole checkpoint dropped first: a box whose disk reads 0.5 GB/s (Lambda's 8x A100, 2026-10-22) then read all of it again from disk)
    // (every layer across a group of GPUs: this GPU's slice of a big matrix's rows, of each expert's: src/dynamic/tp.rs)
    let tp = super::tp::comm();
    let crew = crate::cpu_experts::default_threads().min(64);
    let guess_copies: Rc<[usize]> = crate::tune!("GLYD_KIMI_EARLY")
        .and_then(|v| v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<Vec<usize>>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![GUESS_COPIES])
        .into();
    let (guessing, guess_turn) = (guess_copies.iter().any(|&k| k > 0), Rc::new(std::cell::Cell::new(0usize)));
    // (else the caching's mode, where one is given: tiercheck's --mode)
    let mode = crate::tune!("GLYD_KIMI_HOST")
        .as_deref()
        .and_then(crate::offload::Mode::parse)
        .or(caching.map(|c| c.offload.mode))
        .unwrap_or(crate::offload::Mode::Auto);
    let device = caching.is_some_and(|c| c.offload.device);
    store.set_offload(crate::offload::Offload { mode, threads: crew, pin: true, device, ..Default::default() });
    // (the expert cache across a group of GPUs: its host tier and plans shared, this GPU's copies and pinned buffers: src/dynamic/tp_cache.rs)
    let tp_rank = match (&tp, places.filter(|&p| p < c.experts.experts)) {
        (Some(t), Some(_)) => {
            let cache = t.shared(super::tp_cache::TpCache::default)?;
            let slice = expert_bytes(&c)? / t.n as u64;
            Some(Rc::new(super::tp_cache::Rank::new(cache, t.rank, t.n, o.chunk.max(1), c.experts.top_k, c.experts.latent, store.offload(), slice)?))
        }
        _ => None,
    };
    // (this GPU's place and the group's size, for the threads that pack the experts)
    let tpr = tp.as_ref().map(|t| (t.rank, t.n));
    if let Some((_, n)) = tpr.filter(|&(_, n)| (2 * c.experts.inter) % (64 * n) != 0 || c.experts.latent % (64 * n) != 0) {
        return err(format!("experts of {} x {} across {n} GPUs: each GPU's rows in 64s are wanted", 2 * c.experts.inter, c.experts.latent));
    }
    let early = guessing && places.is_some_and(|p| p < c.experts.experts);
    let d = lowered_of(dir, m, &c, &st, &w32, early).map_err(|e| e.context(&label))?;
    let held =
        LatentHeld { hb: hb.as_ref(), hb_store: hb_store.as_ref(), caching, places, tp_rank: tp_rank.as_ref(), tpr, store: &store, threads, ordinal: o.device };
    let guess = |top_k: usize| -> Result<Option<Guess>> {
        guessing
            .then(|| -> Result<Guess> {
                Ok(Guess {
                    copies: guess_copies.clone(),
                    turn: guess_turn.clone(),
                    ids: crate::dev::Pinned::new(top_k * 8)?,
                    ev: crate::dev::Ev::new()?,
                    pending: Default::default(),
                })
            })
            .transpose()
    };
    let experts = |ck: &Checkpoint, layout: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        let k = Rc::new(KernLatent::load(&dev)?);
        let mut blocks: Vec<Box<dyn MoeBlock>> = Vec::new();
        for (ix, m, roles) in super::experts::of_layers(&kinds, st.layers, 2) {
            let l = ix.layer;
            crate::status::STATUS.beat();
            // (the stage's first cached layer reports its store where a development build asks: GLYD_KIMI_STATS)
            let first = !blocks.iter().any(|b| b.resident().is_none());
            blocks.push(latent_block(m, roles, kinds.desc.hidden, &kinds.scales, ck, layout, l, &held, &k, first, &guess)?);
        }
        Ok(blocks)
    };
    let placing = Placing { across: true, ram: crate::dense_ram::Placer::last(st.layers.0, st.layers.1, caching.map_or(0, |x| x.dense)), hi: None };
    let o = Opts { layout: Some(o.layout.unwrap_or(Layout::Tiered)), ..o.clone() };
    Dyn::new_placed(dir, m, d, None, st, &o, &experts, &placing)
}

/// Kimi-K3's latent attention as the layer kind's operation (`MlaOp`, its matrices, small tensors and ring at 0: `MlaOp::placed` puts a layer's): q_a, the
/// output gate and kv_a one product, the q latent's norm KimiRMSNorm (rounded twice, its epsilon 1e-6), every position attended to.
pub(crate) fn latent(c: &ConfigKimi) -> MlaOp {
    let m = &c.mla;
    MlaOp {
        heads: m.heads,
        nope: m.nope,
        rope: m.rope,
        v: m.v,
        kv_rank: m.kv_rank,
        q_rank: m.q_rank,
        input: MlaIn::Fused { w: 0 },
        q_b: 0,
        o: 0,
        q_norm: 0,
        kv_norm: 0,
        kv_b: 0,
        q_round: crate::runtime::kinds::Rounding::Twice,
        gate: true,
        head_gate: None,
        scale: ((m.nope + m.rope) as f32).powf(-0.5),
        eps: 1e-6,
        ring: 0,
        indexer: None,
        rot: None,
        lent: None,
    }
}

/// Layer `l`'s latent attention: `latent`, and on a layer the newer config's sparse attention names (`ConfigKimi::dsa`) the kind's indexer of a key a token: its
/// keys RMS-normed or LayerNorm-ed at the model's epsilon, the heads' scores weighed, the best `top_k` tokens attended to.
pub(crate) fn latent_at(c: &ConfigKimi, l: usize) -> MlaOp {
    let mut op = latent(c);
    if let Some(d) = c.dsa.as_ref().filter(|d| d.layers.contains(&l)) {
        let ix = super::MlaIndexer {
            pick: super::IdxPick {
                heads: d.heads,
                ratio: 1,
                top_k: d.top_k,
                expand: true,
                tail: false,
                wscale: (d.heads as f32).powf(-0.5),
                score: super::IdxScore::F32 { dim: d.dim, scale: (d.dim as f32).powf(-0.5) },
                keys: super::IdxKeys::Ring(1),
                weighted: true,
                cand: None,
            },
            dim: d.dim,
            q_b: 0,
            k: 0,
            k_norm: 0,
            k_bias: 0,
            weights: 0,
            ape: 0,
            gate: 0,
            key: if d.layer_norm { super::KeyNorm::Layer(c.eps) } else { super::KeyNorm::Rms(c.eps) },
        };
        op.indexer = Some(ix.at(3, 1));
    }
    op
}

/// Each layer's (convolutions fp32, gated norm fp32) as the checkpoint holds them (F32 or BF16; any other type refused), a delta layer's three convolutions alike;
/// (false, false) for a latent attention layer.
/// The kernels read them as they are (Kimi-K3 releases them fp32: values no bf16 holds).
pub(crate) fn dtypes(ck: &Checkpoint, c: &ConfigKimi) -> Result<Vec<(bool, bool)>> {
    // (F32 or BF16 alone: the kernels read 4 or 2 bytes a value by these flags, and the loader's checks count values, so a narrower type would be read past its end)
    let dtype = |n: &str| -> Result<bool> {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        match t.dtype.as_str() {
            "F32" => Ok(true),
            "BF16" => Ok(false),
            d => err(format!("{n}: {d}, where F32 or BF16 is read")),
        }
    };
    (0..c.layers)
        .map(|l| {
            if c.kinds[l] != GlmKind::Kda {
                return Ok((false, false));
            }
            let a = format!("{}layers.{l}.self_attn", c.prefix);
            let cw32 = dtype(&format!("{a}.q_conv1d.weight"))?;
            for x in ["k", "v"] {
                let n = format!("{a}.{x}_conv1d.weight");
                if dtype(&n)? != cw32 {
                    return err(format!("{n}: not the queries' convolution's dtype ({})", if cw32 { "F32" } else { "bf16" }));
                }
            }
            Ok((cw32, dtype(&format!("{a}.o_norm.weight"))?))
        })
        .collect()
}

/// Kimi-K3's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::kimi`, every tensor named), held to
/// `describe` with the names left out (the loader's own description names none). `w32` the delta layers' dtypes (`dtypes`), `early` whether the cached experts
/// are guessed and copied ahead (the `MoeEarly` step).
pub fn lowered(dir: &Path, c: &ConfigKimi, st: &Stage, w32: &[(bool, bool)], early: bool) -> Result<ModelDesc> {
    lowered_of(dir, &ModelConfig::load(dir)?, c, st, w32, early)
}

/// `lowered` of a config.json parsed once (`m`: `dir`'s).
pub fn lowered_of(dir: &Path, m: &ModelConfig, c: &ConfigKimi, st: &Stage, w32: &[(bool, bool)], early: bool) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let lowered = super::lower::kimi(&kinds, c.eos.clone(), c.max_pos, st, w32, early)?;
    super::lower::agree(nameless(&lowered), &describe(c, st, w32, early)?, &label)?;
    Ok(lowered)
}

pub(crate) use super::lower::nameless;

/// The executor's description of Kimi-K3 for stage `st`, as the loader indexes what it loads. Each layer's small tensors: its input and post-attention norms, the
/// two depth mixes' norms and projections, then a delta layer's convolutions, A_log, dt_bias and gated norm; the last stage's output mix's norm and projection and
/// the final norm after the last layer's. Its matrices: a delta layer's three (q, k, v, the output gate, the forget gate's first product and beta as one; o; the
/// forget gate's second product), a dense MLP's gate and up and its down; then the embedding and the head. A latent attention layer's and a mixture of experts' are
/// their blocks'.
pub fn describe(c: &ConfigKimi, st: &Stage, w32: &[(bool, bool)], early: bool) -> Result<ModelDesc> {
    let (kd, e) = (&c.kda, &c.experts);
    let (h, qd, nh) = (c.hidden, kd.heads * kd.dim, kd.heads);
    let starts: Vec<bool> = (0..c.layers).map(|l| l % c.res_block == 0).collect();
    let moe_rows = e.latent + 2 * e.shared * e.inter;
    let indexed = |l: usize| c.kinds[l] == GlmKind::Mla && c.dsa.as_ref().is_some_and(|d| d.layers.contains(&l));
    let kinds: Vec<KimiLayer> = (0..c.layers)
        .map(|l| KimiLayer { kda: c.kinds[l] == GlmKind::Kda, moe: c.moe[l], w32: w32.get(l).copied().unwrap_or_default(), ix: indexed(l) })
        .collect();
    let mla_ix = (0..c.layers).find(|&l| indexed(l)).map(|l| latent_at(c, l));
    Ok(kimi_desc(
        &KimiDims { h, qd, nh, kd: kd.dim, bound: kd.bound, inter: c.inter, situ: c.situ, eps: c.eps, moe_rows, mla: Some(latent(c)), mla_ix },
        &kinds,
        &starts,
        st,
        early,
        c.vocab,
        c.max_pos,
        c.eos.clone(),
    ))
}

/// What a Kimi-K3 layer is to the executor: delta attention or latent attention, a mixture of experts or a dense MLP, its delta layer's dtypes, and whether its
/// latent attention has the indexer (`KimiDims::mla_ix`).
pub(crate) struct KimiLayer {
    pub kda: bool,
    pub moe: bool,
    pub w32: (bool, bool),
    pub ix: bool,
}

/// Kimi-K3's widths and constants as the executor's operations take them: the width, the delta heads' channels, heads and size, the forget gate's bound, the dense
/// MLP's width, SiTU's two betas, the norms' epsilon, the rows of a mixture of experts' first product (the latent's down projection and the shared experts' gate and up),
/// and the latent attention (`latent`, unplaced) without and with the newer config's indexer.
pub(crate) struct KimiDims {
    pub h: usize,
    pub qd: usize,
    pub nh: usize,
    pub kd: usize,
    pub bound: f32,
    pub inter: usize,
    pub situ: (f32, f32),
    pub eps: f32,
    pub moe_rows: usize,
    pub mla: Option<MlaOp>,
    pub mla_ix: Option<MlaOp>,
}

/// Kimi-K3's executor description: its layers `ls`, the depth mix's groups starting where `starts` says, for stage `st`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kimi_desc(k: &KimiDims, ls: &[KimiLayer], starts: &[bool], st: &Stage, early: bool, vocab: usize, max_pos: usize, eos: Vec<i64>) -> ModelDesc {
    let (h, qd, nh, eps) = (k.h, k.qd, k.nh, k.eps);
    // (the delta attention's fused product: q, k, v, the output gate, the forget gate's first product and beta, padded)
    let (fa, row) = (k.kd, fused(4 * qd + k.kd + nh));
    let (mut mix_w, mut proj_w) = (3 * qd, k.kd);
    let (mut n_small, mut n_mats, mut n_moe, mut states, mut layers, mut rings) = (0, 0, 0, Vec::new(), Vec::new(), Vec::new());
    for (l, ly) in ls.iter().enumerate() {
        // (the depth mix's groups before this layer: a group starts at its first layer)
        let before = starts[..l].iter().filter(|&&s| s).count();
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let s = n_small;
        let (in_n, post_n, an, ap, mn, mp) = (s, s + 1, s + 2, s + 3, s + 4, s + 5);
        n_small += 6;
        let nb = before + usize::from(starts[l]);
        let mut ops = vec![Op::DepthMix(DepthMixOp { add: l > st.layers.0, nb: before, w_n: an, w_p: ap, push: starts[l].then_some(before), w: in_n, eps })];
        if ly.moe && early {
            ops.push(Op::MoeEarly { block: n_moe, mix: DepthMixOp { add: false, nb, w_n: mn, w_p: mp, push: None, w: post_n, eps } });
        }
        if ly.kda {
            let m0 = n_mats;
            n_mats += 3;
            (mix_w, proj_w) = (mix_w.max(row), proj_w.max(fa));
            let s0 = n_small;
            n_small += 4;
            let (cv, ssm) = (states.len(), states.len() + 1);
            states.push(3 * 3 * qd * 2);
            states.push(nh * k.kd * k.kd * 4);
            ops.extend([
                Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "delta attention: q, k, v and gates product" },
                Op::KdaConv { w: s0, state: cv, c: 3 * qd, row, w32: ly.w32.0 },
                Op::Parts { row, parts: [(3 * qd, qd, Slot::Iq), (4 * qd, fa, Slot::Proj), (4 * qd + fa, nh, Slot::Ba)] },
                Op::LinearAt { w: m0 + 2, x: View::at(Slot::Proj, 4 * qd), y: Slot::Qo, name: "delta attention: forget gate product" },
                Op::KdaDelta { a_log: s0 + 1, dt_bias: s0 + 2, state: ssm, heads: nh, bound: k.bound, ba: View::at(Slot::Ba, 4 * qd + fa) },
                Op::KdaOut { w: s0 + 3, heads: nh, w32: ly.w32.1, iq: View::at(Slot::Iq, 3 * qd) },
                Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "delta attention: o product" },
            ]);
        } else {
            let op = (if ly.ix { &k.mla_ix } else { &k.mla }).as_ref().expect("a latent attention layer's kind").placed(n_mats, n_small, rings.len());
            (n_mats, n_small) = (n_mats + op.mat_count(), n_small + op.small_count());
            rings.extend(op.rings());
            ops.push(Op::Mla(op));
        }
        ops.push(Op::DepthMix(DepthMixOp { add: true, nb, w_n: mn, w_p: mp, push: None, w: post_n, eps }));
        if ly.moe {
            mix_w = mix_w.max(fused(k.moe_rows));
            ops.push(Op::Moe(MoeOp::of(n_moe)));
            n_moe += 1;
        } else {
            let (gu, down) = (n_mats, n_mats + 1);
            n_mats += 2;
            ops.extend([
                Op::Linear { w: gu, x: Slot::H, y: Slot::Gu, name: "MLP: gate and up product" },
                Op::GatedAct { inter: k.inter, act: Act::Situ { beta: k.situ.0, linear: k.situ.1 } },
                Op::Linear { w: down, x: Slot::M, y: Slot::O, name: "MLP: down product" },
            ]);
        }
        if l + 1 == st.layers.1 && !st.last {
            ops.push(Op::Add { x: Slot::O });
        }
        layers.push(ops);
    }
    if st.last {
        let all = starts.iter().filter(|&&s| s).count();
        if let Some(last) = layers.last_mut() {
            last.push(Op::DepthMix(DepthMixOp { add: st.layers.1 > st.layers.0, nb: all, w_n: n_small, w_p: n_small + 1, push: None, w: n_small + 2, eps }));
        }
    }
    let blocks = starts.iter().filter(|&&s| s).count();
    let mut widths = [0usize; super::SLOTS];
    for (s, wd) in [
        (Slot::X, h),
        (Slot::H, h),
        (Slot::O, h),
        (Slot::Blk, blocks * h),
        (Slot::Mix, mix_w.max(h)),
        (Slot::Xc, (3 * qd).max(h)),
        (Slot::Proj, proj_w),
        (Slot::Qo, qd),
        (Slot::Iq, qd),
        (Slot::Ba, nh.max(8)),
        (Slot::Core, qd),
        (Slot::Att, qd),
        (Slot::Gu, 2 * k.inter),
        (Slot::M, k.inter),
    ] {
        widths[s as usize] = wd;
    }
    ModelDesc {
        family: "Kimi-K3",
        hidden: h,
        vocab,
        eps,
        max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: Op::Nop,
        layers,
        states,
        final_norm: None,
        decoded: false,
        skinny: false,
        embed: n_mats,
        lm: n_mats + 1,
        softcap: 0.0,
        eos,
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: 0.0,
        host: Vec::new(),
        context: Vec::new(),
        stored: Default::default(),
    }
}
