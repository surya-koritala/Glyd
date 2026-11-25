//! DeepSeek-V4 (`DeepseekV4ForCausalLM`, src/deepseek_v4.rs) as a description: the four streams and their mixing around each layer's two blocks, the attention (the query's low rank, the shared key-value
//! head over the window and the compressed cache, the indexer's picks, the sink, the grouped low rank of its output) and the mixture of experts, each product's input quantized to FP8 as the release's are.
//! The weights load exactly: an FP8 matrix as the bf16 it is (packed at the level asked, lossless by default), the FP4 experts as five-bit runs (their codes and power-of-two scales: the same values),
//! the rest as the checkpoint holds it. The arithmetic is kernels/dsv4.cu's, the streams' the streams kind's (kernels/residual.cu); src/dsv4_ref.rs is the CPU reference it is checked against (`dsv4check`).

use super::lower_comp::{LinForm, MoeLoad, Piece};
use super::{Cand, Dyn, IdxPick, ModelDesc, Op, Opts, ResidentMoe, Stage, V4Dims};
use crate::deepseek_v4::{fp4_runs, freqs};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::kern35::Mat;
use crate::kern_nh::MoeBlock;
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeWeights, Router, Scoring, Shared};
use crate::moe_rows::Down;
use crate::runtime::describe::Source;
use crate::runtime::kinds::Body;
use crate::weights::{as_bytes, pack_words, upload_bytes, upload_packed, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// kernels/dsv4.cu's kernels.
pub(crate) struct KernV4 {
    pub(crate) fp8: crate::kt::v4_fp8,
    pub(crate) q_prep: crate::kt::v4_q_prep,
    pub(crate) kv_prep: crate::kt::v4_kv_prep,
    pub(crate) lin32: crate::kt::v4_lin32,
    pub(crate) lin32_1: crate::kt::v4_lin32_1,
    pub(crate) comp_pool: crate::kt::v4_comp_pool,
    pub(crate) comp_out: crate::kt::v4_comp_out,
    pub(crate) idx_q: crate::kt::v4_idx_q,
    pub(crate) attn_scores: crate::kt::v4_attn_scores,
    pub(crate) attn_soft: crate::kt::v4_attn_soft,
    pub(crate) attn_out: crate::kt::v4_attn_out,
    pub(crate) ogather: crate::kt::v4_ogather,
    pub(crate) idx_keys: crate::kt::v4_idx_keys,
}

impl KernV4 {
    fn load(dev: &Dev) -> Result<KernV4> {
        Ok(KernV4 {
            fp8: dev.typed()?,
            q_prep: dev.typed()?,
            kv_prep: dev.typed()?,
            lin32: dev.typed()?,
            lin32_1: dev.typed()?,
            comp_pool: dev.typed()?,
            comp_out: dev.typed()?,
            idx_q: dev.typed()?,
            attn_scores: dev.typed()?,
            attn_soft: dev.typed()?,
            attn_out: dev.typed()?,
            ogather: dev.typed()?,
            idx_keys: dev.typed()?,
        })
    }
}

/// The executor's DeepSeek-V4 buffers: the attention's scores and softmax sums (`ra` rows at a time), the indexer's scores, picks and their counts, a compressor's pooled entries.
pub(crate) struct V4Run {
    pub k: KernV4,
    pub dims: V4Dims,
    pub sc: Buf,
    pub den: Buf,
    pub isc: Buf,
    pub sel: Buf,
    pub cnt: Buf,
    pub pooled: Buf,
    /// the grouped low rank's pairs' groups (int64 [chunk, groups]: pair t groups + g is group g) and their plan
    pub gids: Buf,
    pub gplan: Buf,
    /// a part of a prompt's chunk of more than the grouped product's 64 rows (`Op::OGroupLoraStack`): its rows of Att and of Mid, group by group, one after
    /// another (Att and Mid hold a group's rows of the whole chunk, then the next group's)
    pub ox: Buf,
    pub oy: Buf,
    /// the candidate blocks (`IdxPick::cand`): each row's published blocks (`ctop` a row) and their counts, the blocks' scores (`bmax` a row, `ra` rows) and their counts
    pub cand: Buf,
    pub ncand: Buf,
    pub bsc: Buf,
    pub nent: Buf,
    pub ctop: usize,
    pub bmax: usize,
    /// keys a query reads at most, the indexer's entries at most, rows the attention takes at a time
    pub lmax: usize,
    pub emax: usize,
    pub ra: usize,
}

impl V4Run {
    pub fn new(dev: &Dev, d: &ModelDesc, dims: &V4Dims, ch: usize, ctx: usize) -> Result<V4Run> {
        let lmax = dims.window + dims.topk.max(if dims.all_ratio > 0 { ctx / dims.all_ratio + 1 } else { 0 });
        // (the entries an indexer scores, a chunk's compressors make: the smallest ratio's; 4 where there is none, as before the ratios of DeepSeek-V4.1)
        let emax = ctx / if dims.idx_ratio > 0 { dims.idx_ratio } else { 4 } + 1;
        let minr = [dims.idx_ratio, dims.all_ratio, 4].into_iter().filter(|&r| r > 0).min().unwrap_or(4);
        let ra = ch.min(64);
        let cands: Vec<(usize, usize)> = d
            .layers
            .iter()
            .flatten()
            .filter_map(|op| match op {
                Op::IdxPick(IdxPick { cand: Some(Cand::Publish { block, top } | Cand::Inside { block, top }), .. }) => Some((*block, *top)),
                _ => None,
            })
            .collect();
        let ctop = cands.iter().map(|c| c.1).max().unwrap_or(0);
        let bmax = cands.iter().map(|c| c.0).min().map_or(0, |b| emax / b + 1);
        let g = dims.o_groups.max(1);
        let gids = Buf::new(ch * g * 8)?;
        gids.write(0, &(0..ch * g).map(|j| (j % g) as i64).collect::<Vec<i64>>())?;
        Ok(V4Run {
            k: KernV4::load(dev)?,
            dims: dims.clone(),
            sc: Buf::zeroed(ra * dims.heads * lmax * 4)?,
            den: Buf::zeroed(ra * dims.heads * 4)?,
            isc: Buf::zeroed(ra * emax * 4)?,
            sel: Buf::zeroed(ch * dims.topk.max(1) * 4)?,
            cnt: Buf::zeroed(ch * 4)?,
            pooled: Buf::zeroed((ch / minr + 2) * 512 * 4)?,
            gids,
            gplan: Buf::zeroed((2 + 2 * g + ch * g) * 4)?,
            ox: Buf::new(if ch > crate::f8::MOE_TOKENS { crate::f8::MOE_TOKENS * dims.heads * 512 * 2 } else { 16 })?,
            oy: Buf::new(if ch > crate::f8::MOE_TOKENS { crate::f8::MOE_TOKENS * d.width(crate::dynamic::Slot::Mid) * 2 } else { 16 })?,
            cand: Buf::zeroed((ch * ctop * 4).max(4))?,
            ncand: Buf::zeroed(ch * 4)?,
            bsc: Buf::zeroed((ra * bmax * 4).max(4))?,
            nent: Buf::zeroed(ra * 4)?,
            ctop,
            bmax,
            lmax,
            emax,
            ra,
        })
    }
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The rotary tables of frequencies `f` for positions 0..mp: cos and sin [mp, f.len()] fp32, the angle pos x f in fp32 and its cosine and sine in f64 (as the CPU reference).
fn tables(f: &[f32], mp: usize) -> Result<(Buf, Buf)> {
    let (mut c, mut s) = (Vec::with_capacity(mp * f.len()), Vec::with_capacity(mp * f.len()));
    for p in 0..mp {
        for &fr in f {
            let (sn, cs) = ((p as f32 * fr) as f64).sin_cos();
            c.push(cs as f32);
            s.push(sn as f32);
        }
    }
    Ok((upload_bytes(as_bytes(&c))?, upload_bytes(as_bytes(&s))?))
}

/// The model's layers (its compressed attention blocks), its end tokens (generation_config.json's, else the config's) and its context, from its config.
pub fn ends(dir: &Path) -> Result<(usize, Vec<i64>, usize)> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let layers = kinds.desc.blocks.iter().filter(|b| matches!(b.body, Body::Mixer(_))).count();
    let c = crate::runtime::config::ModelConfig::load(dir)?;
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let mut eos = crate::runtime::config::eos_ids(&c.text, generation.as_deref());
    if eos.is_empty() {
        eos = crate::runtime::config::eos_ids(&c.top, None);
    }
    eos.dedup();
    let max_pos = c
        .text
        .get("max_position_embeddings")
        .or(c.top.get("max_position_embeddings"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error(format!("{label}: no max_position_embeddings")))?;
    Ok((layers, eos, max_pos as usize))
}

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_stage(dir, o, Stage::whole(ends(dir)?.0))
}

/// Stage `st` of the model (its layers; the embedding where it is the first, the head where the last).
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    load_stage_hb(dir, o, st, None)
}

/// A rotary table's frequencies from its kind: model.py's (`freqs`), YaRN's stretch where the kind has it.
fn freqs_of(r: &crate::runtime::kinds::Rope) -> Vec<f32> {
    let yarn = match r.scaling {
        crate::runtime::kinds::RopeScaling::Yarn { factor, original_max, beta_fast, beta_slow, .. } => {
            Some(crate::deepseek_v4::Yarn { factor, orig: original_max as usize, beta_fast, beta_slow })
        }
        _ => None,
    };
    freqs(r.dim, r.theta, yarn)
}

/// The blocks of rows and columns a quantized matrix of `rows` x `cols` has one scale of, from its scales' shape (one row of scales a row, or blocks of 32 or 128).
fn scale_blocks(rows: usize, cols: usize, shape: &[u64]) -> Option<[usize; 2]> {
    let (sr, sc) = (shape.first().copied()? as usize, shape.get(1).copied().unwrap_or(1) as usize);
    let of = |n: usize, k: usize| [128usize, 32, 1].into_iter().find(|&b| n.div_ceil(b) == k);
    Some([of(rows, sr)?, of(cols, sc)?])
}

/// `load_stage` with a hummingbird directory's experts in the expert cache (`caching`: the stage's places and memory, src/hbg_store.rs). The model's layer kinds
/// lowered (`lower_comp::lower`), each index of its operations loaded as the plan says, in the release's formats.
pub fn load_stage_hb(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    load_stage_with(dir, o, st, caching, V4Opts::of(o.layout.unwrap_or(Layout::Tiered), o.ctx))
}

/// The whole model with the release's simulated quantizations on or off (`V4Opts::sim`, which GLYD_V4_SIM sets in every other load): a check's engine of
/// either kind (src/logits.rs) without the environment's say.
pub fn load_sim(dir: &Path, o: &Opts, sim: bool) -> Result<Dyn> {
    let vo = V4Opts { sim, ..V4Opts::of(o.layout.unwrap_or(Layout::Tiered), o.ctx) };
    load_stage_with(dir, o, Stage::whole(ends(dir)?.0), None, vo)
}

/// `load_stage_hb` with the options `vo` given.
pub fn load_stage_with(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>, vo: V4Opts) -> Result<Dyn> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let (_, eos, max_pos) = ends(dir)?;
    let layout = o.layout.unwrap_or(Layout::Tiered);
    let (desc, plan) = super::lower_comp::lower(&kinds, eos, max_pos, &st, &vo).map_err(|e| e.context(&label))?;
    let cfg = crate::runtime::config::ModelConfig::load(dir)?;
    let inter = cfg.text.get("moe_intermediate_size").and_then(|v| v.as_u64()).ok_or_else(|| Error(format!("{label}: no moe_intermediate_size")))? as usize;
    let dev = Dev::new(o.device)?;
    let ck = io(Checkpoint::open(dir))?;
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let store = caching.map(crate::hbg_store::store_for);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let (d, ch) = (desc.hidden, o.chunk.max(1));
    let ctx = o.ctx.div_ceil(16) * 16;
    let read = |name: &str| io(ck.read(name));
    let dims = |name: &str| -> Result<(usize, usize)> {
        let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        Ok((t.shape[0] as usize, t.shape.get(1).copied().unwrap_or(1) as usize))
    };
    // a quantized matrix's scales (E8M0 bytes) and the blocks they are of; None: the matrix is stored as the bf16 it is
    let scaled = |name: &str, rows: usize, cols: usize| -> Result<Option<(Vec<u8>, [usize; 2])>> {
        let Some(sn) = kinds.scales.get(name) else { return Ok(None) };
        let (_, t) = ck.get(sn).ok_or_else(|| Error(format!("no tensor {sn}")))?;
        let b = scale_blocks(rows, cols, &t.shape).ok_or_else(|| Error(format!("{sn}: scales of {:?} for {rows} x {cols}", t.shape)))?;
        Ok(Some((read(sn)?, b)))
    };
    let words = |b: &[u8]| -> Vec<u16> { b.chunks(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect() };
    // rows [r0, r1) of matrix `name` as bf16 words: an FP8 one decoded with its scales (exact), a bf16 one as stored
    let dense = |name: &str, r0: usize, r1: usize| -> Result<Vec<u16>> {
        let (rows, cols) = dims(name)?;
        let q = read(name)?;
        match scaled(name, rows, cols)? {
            None => Ok(words(&q[r0 * cols * 2..r1 * cols * 2])),
            Some((sc, [br, bc])) => {
                if r0 % br != 0 {
                    return err(format!("{name}: rows from {r0}, inside a block of {br} scales"));
                }
                let gc = cols.div_ceil(bc);
                let s: Vec<f32> = sc[r0 / br * gc..r1.div_ceil(br) * gc].iter().map(|&x| crate::deepseek_v4::e8m0(x)).collect();
                let mut out = vec![0u16; (r1 - r0) * cols];
                glyd_gpu::fp8::dequant(&q[r0 * cols..r1 * cols], &s, r1 - r0, cols, [br, bc], &mut out).map_err(Error)?;
                Ok(out)
            }
        }
    };
    // (every layer across a group of GPUs: this GPU's slice of a big matrix's rows, of each expert's: src/dynamic/tp.rs)
    let tp = super::tp::comm();
    let tpr = tp.as_ref().map(|t| (t.rank, t.n));
    // (a matrix the layouts do not take, rows not a multiple of 64: held dense, as it is; sliced where `slice` and the group takes it)
    let pack_as = |w: &[u16], rows: usize, cols: usize, name: &str, slice: bool| -> Result<Mat> {
        if rows % 64 != 0 || cols % 16 != 0 {
            return Ok(Mat::Dense(upload_bytes(as_bytes(w))?, rows, cols));
        }
        if let Some(t) = tp.as_ref().filter(|t| slice && t.slices(rows, cols)) {
            let (r0, rn) = t.rows(rows);
            let mut pk = upload_packed(pack_words(&w[r0 * cols..(r0 + rn) * cols], rn, cols, layout, name)?)?;
            pk.tp_rows = rows;
            return Ok(Mat::Packed(pk));
        }
        Ok(Mat::Packed(upload_packed(pack_words(w, rows, cols, layout, name)?)?))
    };
    let pack = |w: &[u16], rows: usize, cols: usize, name: &str| pack_as(w, rows, cols, name, true);
    // (the library's FP8 products take blocks of 128: a release of other blocks has its shared experts held as their bf16 too, as `lower_comp::lower` holds its matrices)
    let exact8 = vo.exact8 && kinds.fp8_block.is_none_or(|b| b == 128);
    let lib = crate::ffi::Lib::find()?;
    let (mut mats, mut f8): (Vec<Mat>, Vec<crate::f8::F8Mat>) = Default::default();
    // the products' matrices, in the plan's order: the large FP8 ones held FP8 as released where the lowering says (`LinLoad::f8`: the level and their size), the rest as
    // the bf16 they are or decode to, packed lossless (or at the level asked)
    for x in &plan.lins {
        crate::status::STATUS.beat();
        let label = x.name.trim_end_matches(".weight");
        match x.form {
            LinForm::Fp8 { rows, groups } => {
                let (all, cols) = dims(&x.name)?;
                let (r0, r1) = rows.unwrap_or((0, all));
                let what = if rows.is_some() { format!("{label} rows {r0}..{r1}") } else { label.to_string() };
                if !x.f8 {
                    mats.push(pack(&dense(&x.name, r0, r1)?, r1 - r0, cols, &what)?);
                    continue;
                }
                let (sc, b) = scaled(&x.name, all, cols)?.ok_or_else(|| Error(format!("{}: held FP8, and it has no scales", x.name)))?;
                if b != [128, 128] {
                    return err(format!(
                        "{}: FP8 in blocks of {b:?}: the library's FP8 products take blocks of 128 (GLYD_V4_F8=0 holds it as its bf16)",
                        x.name
                    ));
                }
                let q = read(&x.name)?;
                let gc = cols.div_ceil(128);
                // (a group of GPUs: this GPU's rows of each group's, whole blocks of 128 for the scales)
                let per = (r1 - r0) / groups.max(1);
                let parts: Vec<(usize, usize)> = match tpr.filter(|&(_, n)| per % (128 * n) == 0 && tp.as_ref().is_some_and(|t| t.slices(r1 - r0, cols))) {
                    Some((rank, n)) => (0..groups).map(|g| (r0 + g * per + rank * per / n, r0 + g * per + (rank + 1) * per / n)).collect(),
                    None => vec![(r0, r1)],
                };
                let (mut qq, mut ss) = (Vec::new(), Vec::new());
                for &(a, b) in &parts {
                    qq.extend_from_slice(&q[a * cols..b * cols]);
                    ss.extend(sc[a / 128 * gc..b.div_ceil(128) * gc].iter().map(|&x| crate::deepseek_v4::e8m0(x)));
                }
                let rows_here = parts.iter().map(|p| p.1 - p.0).sum();
                let mut m = crate::f8::F8Mat::pack(&dev, &lib, &qq, &ss, rows_here, cols, cfg!(feature = "dev"))?;
                m.tp_rows = if parts.len() > 1 || rows_here < r1 - r0 { per } else { 0 };
                f8.push(m);
            }
            LinForm::Bf16 { slice } => {
                let (rows, cols) = dims(&x.name)?;
                mats.push(pack_as(&words(&read(&x.name)?), rows, cols, label, slice)?);
            }
            LinForm::Table => {
                let (rows, cols) = dims(&x.name)?;
                mats.push(Mat::Dense(upload_bytes(as_bytes(&dense(&x.name, 0, rows)?))?, rows, cols));
            }
            LinForm::Absent => mats.push(Mat::Dense(Buf::new(2)?, 0, 0)),
        }
    }
    if mats.len() != desc.lm + 1 || f8.len() != plan.lins.iter().filter(|x| x.f8).count() {
        return err(format!(
            "{label}: {} matrices and {} FP8 ones loaded for a description of {} and {}",
            mats.len(),
            f8.len(),
            desc.lm + 1,
            plan.lins.iter().filter(|x| x.f8).count()
        ));
    }
    // the small tensors: each its pieces one after another (a tensor as stored, zeros, the lowering's numbers, the tokenizer's compressed vocabulary)
    let token_map = || -> Result<Vec<u8>> {
        let p = dir.join("engram_token_map.json");
        let v: Vec<i64> = serde_json::from_str(&io(std::fs::read_to_string(&p))?).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        if v.len() != desc.vocab {
            return err(format!("{}: {} ids for a vocabulary of {}", p.display(), v.len(), desc.vocab));
        }
        Ok(v.iter().flat_map(|&t| (t as i32).to_le_bytes()).collect())
    };
    let sm: Vec<Buf> = plan
        .small
        .iter()
        .map(|pieces| {
            let mut b = Vec::new();
            for p in pieces {
                match p {
                    Piece::Tensor(t) => b.extend(read(t)?),
                    Piece::Fp32(t) => match ck.get(t).map(|(_, x)| x.dtype.as_str()) {
                        Some("F32") => b.extend(read(t)?),
                        Some("BF16") => {
                            b.extend(read(t)?.chunks(2).flat_map(|w| f32::from_bits((u16::from_le_bytes([w[0], w[1]]) as u32) << 16).to_le_bytes()))
                        }
                        d => return err(format!("{t}: {d:?}, where fp32 or bf16 is read")),
                    },
                    Piece::Zeros(n) => b.resize(b.len() + n, 0),
                    Piece::I64(v) => b.extend(v.iter().flat_map(|x| x.to_le_bytes())),
                    Piece::TokenMap => b.extend(token_map()?),
                }
            }
            upload_bytes(&b)
        })
        .collect::<Result<_>>()?;
    // the experts, block by block
    let mut moe: Vec<Box<dyn MoeBlock>> = Vec::new();
    for ml in &plan.moe {
        crate::status::STATUS.beat();
        moe.push(experts(
            &ExpertsAt {
                ck: &ck,
                kinds: &kinds,
                dev: &dev,
                lib: &lib,
                hb: hb.as_ref(),
                caching,
                store: store.as_ref(),
                threads,
                tpr,
                exact8,
                // (the description's: a release whose quantizations the kernels do not simulate runs without them, `lower_comp::lower`)
                sim: desc.v4.as_ref().is_some_and(|v| v.sim),
                inter,
            },
            ml,
            d,
            &pack,
            &dense,
        )?);
    }
    let mp = ctx + ch + 16;
    let ropes = plan.ropes.iter().map(|r| tables(&freqs_of(r), mp)).collect::<Result<Vec<_>>>()?;
    Dyn::from_loaded_f8(dev, desc, mats, f8, sm, moe, ropes, layout, st, o)
}

/// What a layer's experts are loaded with: the checkpoint and its description, the GPU and the library, a hummingbird directory and the expert cache where
/// there are, the host's threads, this GPU's place in a group, the level's FP8 and its simulated quantizations, the experts' width (the config's).
struct ExpertsAt<'a> {
    ck: &'a Checkpoint,
    kinds: &'a crate::runtime::describe::Described,
    dev: &'a Dev,
    lib: &'a crate::ffi::Lib,
    hb: Option<&'a crate::hb_pack::HbDir>,
    caching: Option<&'a crate::hbg_store::HbCaching>,
    store: Option<&'a std::rc::Rc<crate::offload::ExpertStore>>,
    threads: usize,
    tpr: Option<(usize, usize)>,
    exact8: bool,
    sim: bool,
    inter: usize,
}

/// A layer's experts (`MoeLoad`): the routed ones in the smallest layout that holds them exactly, the shared one as released or as its bf16, the router's weights.
/// (`pack`: a bf16 matrix packed at the level; `dense`: rows of a matrix as the bf16 it is or decodes to)
type PackFn<'a> = &'a dyn Fn(&[u16], usize, usize, &str) -> Result<Mat>;
type DenseFn<'a> = &'a dyn Fn(&str, usize, usize) -> Result<Vec<u16>>;

fn experts(x: &ExpertsAt, ml: &MoeLoad, d: usize, pack: PackFn, dense: DenseFn) -> Result<Box<dyn MoeBlock>> {
    // (what the experts' packing threads read: the checkpoint, the scales' names, this GPU's place in a group)
    let (ck, l, m, scales, tpr) = (x.ck, ml.layer, &ml.moe, &x.kinds.scales, x.tpr);
    if m.router.score != crate::runtime::kinds::RouterScore::SqrtSoftplus || m.router.groups.is_some() || !m.expert.gated || m.shared_gate || m.latent.is_some()
    {
        return err(format!("layer {l}'s experts: {m:?}: these experts run the sqrt-softplus router and SwiGLU experts with a shared one"));
    }
    let read = |name: &str| io(ck.read(name));
    let role = |r: &str| ml.roles.iter().find(|(n, _)| n == r).map(|(_, s)| s).ok_or_else(|| Error(format!("layer {l}'s experts: no role {r}")));
    let tensor = |r: &str| -> Result<String> {
        match role(r)? {
            Source::Tensor(t) => Ok(t.clone()),
            s => err(format!("layer {l}'s experts: role {r} is {s:?}")),
        }
    };
    let per_expert = |r: &str| -> Result<Vec<Vec<String>>> {
        match role(r)? {
            Source::Experts(v) => Ok(v.clone()),
            s => err(format!("layer {l}'s experts: role {r} is {s:?}")),
        }
    };
    let scale_of =
        |w: &str| scales.get(w).cloned().ok_or_else(|| Error(format!("{w}: an expert stored without its scales: the FP4 layouts take the release's")));
    let (ups, downs) = (per_expert("experts_gate_up")?, per_expert("experts_down")?);
    let (e, k) = (m.experts, m.router.top_k);
    if ups.len() != e || downs.len() != e || ups.iter().any(|u| u.len() != 2) {
        return err(format!("layer {l}: {} and {} experts' matrices for {e} experts, each gate and up", ups.len(), downs.len()));
    }
    // (the experts' width as the config says it: a hummingbird directory holds them as codes, not as the release's matrices)
    let i = x.inter;
    if i % 128 != 0 {
        return err(format!("experts of width {i}: this engine runs widths of 128s"));
    }
    let limit = m.expert.clamp.filter(|&c| c > 0.0);
    // each expert's two matrices in the smallest layout that holds the layer's experts exactly (one pack for the stack, so one layout for all of them): the FP4 layout with packed scales (4.07
    // bits a weight), else the FP4 layout as released (4.25), else five-bit runs (5.5; GLYD_V4_FP4=0 asks for them); packed on all the cores (an expert's pack is its rows of the stacked matrix's:
    // block layouts have fixed strides), then joined
    // (a group of GPUs: the FP4 layouts alone, each expert's rows sliced)
    // (the host's pairs of the cache: the release's SwiGLU with its clamp, weighed after the second product, where the GPU weighs the activation and rounds it to FP8)
    let act = Act::SwiGlu { limit };
    let hbx = crate::hbg_store::hb_layer(x.hb, l, x.tpr.is_some(), x.caching.zip(x.store), act, crate::cpu_hbg::Finish::Rotated, x.dev.ordinal, k)?;
    let mut tiered = None;
    let (up, down) = if hbx.is_some() {
        (ExpertStack::shape_only(e, 2 * i, d), ExpertStack::shape_only(e, d, i))
    } else {
        let kinds: &[u8] = if x.tpr.is_some() {
            &[2, 1]
        } else if crate::tune!("GLYD_V4_FP4").as_deref() != Some("0") {
            &[2, 1, 0]
        } else {
            &[0]
        };
        let one = |ex: usize, kind: u8| -> Result<Option<(glyd_gpu::pack::Tiered, glyd_gpu::pack::Tiered)>> {
            let raw = |w: &str| -> Result<(Vec<u8>, Vec<u8>)> { Ok((read(w)?, read(&scale_of(w)?)?)) };
            let (g, u, dn) = (&ups[ex][0], &ups[ex][1], &downs[ex][0]);
            if kind > 0 {
                let pack = if kind == 2 { glyd_gpu::pack::pack_fp4r_checked } else { glyd_gpu::pack::pack_fp4_checked };
                let ((mut uq, mut ue), (gq, ge), (dq, de)) = (raw(g)?, raw(u)?, raw(dn)?);
                // (an expert's first matrix: the gate's rows, then the up's)
                uq.extend(gq);
                ue.extend(ge);
                // (a group of GPUs: this GPU's rows of each, codes half a byte a value and a scale a run of 32)
                let rows_of = |q: Vec<u8>, sc: Vec<u8>, rows: usize, cols: usize| -> (Vec<u8>, Vec<u8>, usize) {
                    match tpr {
                        Some((rank, n)) => {
                            let (r0, rn) = (rank * rows / n, rows / n);
                            (q[r0 * cols / 2..(r0 + rn) * cols / 2].to_vec(), sc[r0 * cols / 32..(r0 + rn) * cols / 32].to_vec(), rn)
                        }
                        None => (q, sc, rows),
                    }
                };
                let (uq, ue, ur) = rows_of(uq, ue, 2 * i, d);
                let (dq, de, dr) = rows_of(dq, de, d, i);
                return Ok(pack(&uq, &ue, ur, d).zip(pack(&dq, &de, dr, i)));
            }
            let runs = |w: &str, rows: usize, cols: usize| -> Result<(Vec<i8>, Vec<u16>)> {
                let (q, s) = raw(w)?;
                fp4_runs(&q, &s, rows, cols)
            };
            let ((mut uc, mut us), (gc, gs)) = (runs(g, i, d)?, runs(u, i, d)?);
            uc.extend(gc);
            us.extend(gs);
            let (dc, ds) = runs(dn, d, i)?;
            let bad = || Error(format!("layer {l}'s expert {ex}: not five-bit runs"));
            Ok(Some((
                glyd_gpu::pack::pack_int5_runs_checked(&uc, &us, 2 * i, d).ok_or_else(bad)?,
                glyd_gpu::pack::pack_int5_runs_checked(&dc, &ds, d, i).ok_or_else(bad)?,
            )))
        };
        let (mut ups, mut downs) = (Vec::with_capacity(e), Vec::with_capacity(e));
        for &kind in kinds {
            (ups.clear(), downs.clear());
            let mut all = true;
            crate::weights::ordered(e, x.threads, 2 * x.threads, &|ex| one(ex, kind), |_, p| {
                match p {
                    Some((u, dn)) => (ups.push(u), downs.push(dn)),
                    None => (all = false, ()),
                };
                Ok(())
            })?;
            if all {
                break;
            }
        }
        if ups.len() != e {
            return err(format!(
                "layer {l}: experts the FP4 layouts do not hold exactly{}",
                if x.tpr.is_some() { " (across GPUs, the five-bit runs are not taken)" } else { "" }
            ));
        }
        let (ju, jd) = (Joined::of(&ups.iter().collect::<Vec<_>>())?, Joined::of(&downs.iter().collect::<Vec<_>>())?);
        drop((ups, downs));
        match x.caching.zip(x.store) {
            // (the expert cache, R12: places for `places` experts a layer, every expert in the computer's memory, the misses copied in)
            Some((cg, st)) => {
                if x.tpr.is_some() || cg.ram.is_some() {
                    return err("DeepSeek-V4's experts in the expert cache: on one GPU a stage, every expert in the computer's memory (the disk holds hummingbird directories' experts alone)");
                }
                // (the host computes the misses the plan gives it where the experts are in an exact FP4 layout, as the GPU's act_w does: src/cpu_fp4.rs; the plan on the GPU
                // copies from the host tier through the same half, in copy mode too)
                let fp4 = [glyd_gpu::pack::FP4_TAG, glyd_gpu::pack::FP4R_TAG].contains(&ju.tag);
                let half = (fp4 && (st.wants_cpu() || st.offload().device)).then_some(crate::offload::HostHalf::Fp4 {
                    inter: i,
                    latent: d,
                    packed: ju.tag == glyd_gpu::pack::FP4R_TAG,
                    act: crate::cpu_fp4::Act::Release { lim: limit.unwrap_or(f32::INFINITY), fp8: x.sim },
                });
                let (u, dd, cache) = crate::tiering::cached_stacks(&ju, &crate::moe_rows::DownHost::Block(jd), cg.places, st, l, x.dev.ordinal, half)?;
                let Down::Block(dd) = dd else { return err("DeepSeek-V4's second matrices in a row layout") };
                tiered = Some(cache);
                (u, dd)
            }
            None => {
                let (mut up, mut down) = (ExpertStack::upload(&ju)?, ExpertStack::upload(&jd)?);
                if let Some((_, n)) = x.tpr {
                    (up.tp_rows, down.tp_rows) = (up.rows * n, down.rows * n);
                }
                (up, down)
            }
        }
    };
    // the shared expert: as released in the FP8 format where the level holds FP8 so and it is stored so (its gate and up stacked, its down), else the bf16 it is or decodes to, packed
    let (sg, su, sd) = (tensor("shared_gate")?, tensor("shared_up")?, tensor("shared_down")?);
    let shared8 = if x.exact8 && [&sg, &su, &sd].iter().all(|w| x.kinds.scales.contains_key(*w)) {
        let f8 = |names: &[&String], rows: usize, cols: usize| -> Result<crate::f8::F8Mat> {
            let (mut q, mut sc) = (Vec::new(), Vec::new());
            for n in names {
                q.extend(read(n)?);
                sc.extend(read(&scale_of(n)?)?.iter().map(|&b| crate::deepseek_v4::e8m0(b)));
            }
            // (a group of GPUs: this GPU's rows, in whole blocks of 128 of the scales)
            if let Some((rank, n)) = x.tpr.filter(|&(_, n)| rows % (128 * n) == 0) {
                let (r0, rn, gc) = (rank * rows / n, rows / n, cols.div_ceil(128));
                let mut m = crate::f8::F8Mat::pack(
                    x.dev,
                    x.lib,
                    &q[r0 * cols..(r0 + rn) * cols],
                    &sc[r0 / 128 * gc..(r0 + rn) / 128 * gc],
                    rn,
                    cols,
                    cfg!(feature = "dev"),
                )?;
                m.tp_rows = rows;
                return Ok(m);
            }
            crate::f8::F8Mat::pack(x.dev, x.lib, &q, &sc, rows, cols, cfg!(feature = "dev"))
        };
        Some((f8(&[&sg, &su], 2 * i, d)?, f8(&[&sd], d, i)?))
    } else {
        None
    };
    let shared = match &shared8 {
        Some(_) => Shared { inter: i, up: PackBuf::shape_only(2 * i, d)?, down: PackBuf::shape_only(d, i)?, gate: None },
        None => {
            let su = [dense(&sg, 0, i)?, dense(&su, 0, i)?].concat();
            let packed = |w: &[u16], rows: usize, cols: usize, name: &str| -> Result<PackBuf> {
                match pack(w, rows, cols, name)? {
                    Mat::Packed(p) => Ok(p),
                    Mat::Dense(..) | Mat::F8(_) => err(format!("{name}: rows of a block layout are wanted")),
                }
            };
            Shared {
                inter: i,
                up: packed(&su, 2 * i, d, &format!("layers.{l}.ffn.shared_experts.w13"))?,
                down: packed(&dense(&sd, 0, d)?, d, i, sd.trim_end_matches(".weight"))?,
                gate: None,
            }
        }
    };
    let cfg = MoeCfg {
        experts: e,
        router_hidden: d,
        hidden: d,
        inter: i,
        router: Router { scoring: Scoring::SqrtSoftplus, top_k: k, renorm: m.router.renorm, scale: m.router.scale, bf16_logits: false },
        act,
    };
    let w = MoeWeights {
        router: upload_bytes(&read(&tensor("router")?)?)?,
        bias: if m.router.bias { Some(upload_bytes(&read(&tensor("choice_bias")?)?)?) } else { None },
        escale: None,
        up,
        down: Down::Block(down),
    };
    let layer = match hbx {
        Some(h) => h.into_layer(cfg, w.router, w.bias, None, Some(shared))?,
        None => match tiered {
            Some(cache) => MoeLayer::new(cfg, w, Some(shared))?.with_cache(cache)?,
            None => MoeLayer::new(cfg, w, Some(shared))?,
        },
    };
    Ok(Box::new(ResidentMoe(match shared8 {
        Some((u, dn)) => layer.with_shared8(u, dn)?,
        None => layer,
    })))
}

/// What the level and the switches make of DeepSeek-V4's products and caches: the large FP8 matrices held FP8 as released where the level is the lossless one
/// (`exact8`; GLYD_V4_F8=0 asks for the bf16 they decode to), from `f8_min` weights (GLYD_V4_F8_MIN); the output groups' products one stack where they are FP8
/// (`ostack`; GLYD_V4_OSTACK=0 asks for one product a group); the release's quantizations of the products' inputs and caches simulated (`sim`; GLYD_V4_SIM=0);
/// the context the compressors' caches hold, in 16s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V4Opts {
    pub exact8: bool,
    pub f8_min: usize,
    pub ostack: bool,
    pub sim: bool,
    pub ctx: usize,
}

impl V4Opts {
    /// The options a load at `layout` with a context of `ctx` tokens runs with.
    pub fn of(layout: Layout, ctx: usize) -> V4Opts {
        V4Opts {
            exact8: matches!(layout, Layout::Tiered) && crate::tune!("GLYD_V4_F8").as_deref() != Some("0"),
            // (the FP8 format where it is the faster: a large matrix, whose fewer bytes outrun its heavier decode; 2026-10-18 on the 4080 SUPER, a token's product read
            // from memory: 33M weights 55-59 us against the bf16 pack's 70, 8M and fewer as fast or slower)
            f8_min: crate::tune!("GLYD_V4_F8_MIN").and_then(|v| v.parse().ok()).unwrap_or(16 << 20),
            ostack: crate::tune!("GLYD_V4_OSTACK").as_deref() != Some("0"),
            sim: crate::tune!("GLYD_V4_SIM").as_deref() != Some("0"),
            ctx: ctx.div_ceil(16) * 16,
        }
    }

    /// Rows `[r0, r0 + rows)` of an FP8 matrix of `cols` columns held FP8 (whole blocks of 128 rows).
    pub(crate) fn f8(&self, rows: usize, cols: usize, r0: usize) -> bool {
        self.exact8 && rows * cols >= self.f8_min && r0 % 128 == 0
    }

    /// The output groups' matrix (`rows` x `cols`, groups of `ol` rows) as one FP8 stack.
    pub(crate) fn stack(&self, rows: usize, cols: usize, ol: usize) -> bool {
        self.ostack && ol % 64 == 0 && self.f8(rows, cols, 0)
    }
}
