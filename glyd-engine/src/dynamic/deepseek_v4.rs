//! DeepSeek-V4 (`DeepseekV4ForCausalLM`, src/deepseek_v4.rs) as a description: the four streams and their mixing around each layer's two blocks, the attention (the query's low rank, the shared key-value
//! head over the window and the compressed cache, the indexer's picks, the sink, the grouped low rank of its output) and the mixture of experts, each product's input quantized to FP8 as the release's are.
//! The weights load exactly: an FP8 matrix as the bf16 it is (packed at the level asked, lossless by default), the FP4 experts as five-bit runs (their codes and power-of-two scales: the same values),
//! the rest as the checkpoint holds it. The arithmetic is kernels/dsv4.cu's, the streams' the streams kind's (kernels/residual.cu); src/dsv4_ref.rs is the CPU reference it is checked against (`dsv4check`).

use super::{Dyn, HeadMix, IdxKeys, IdxPick, IdxScore, ModelDesc, MoeOp, NormW, Op, Opts, ResidentMoe, Ring, Slot, Stage, StreamsMix, V4Dims};
use crate::deepseek_v4::{fp4_runs, fp8_bf16, ConfigV4};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::kern35::Mat;
use crate::kern_nh::MoeBlock;
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeWeights, Router, Scoring, Shared};
use crate::moe_rows::Down;
use crate::runtime::kinds::Rounding;
use crate::weights::{as_bytes, pack_words, upload_bytes, upload_packed, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

// The kernels' names: those the .cu sources give them, in every build (kname!).
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
    /// keys a query reads at most, the indexer's entries at most, rows the attention takes at a time
    pub lmax: usize,
    pub emax: usize,
    pub ra: usize,
}

impl V4Run {
    pub fn new(dev: &Dev, d: &ModelDesc, dims: &V4Dims, ch: usize, ctx: usize) -> Result<V4Run> {
        let lmax = dims.window + dims.topk.max(if dims.all_ratio > 0 { ctx / dims.all_ratio + 1 } else { 0 });
        let emax = ctx / 4 + 1;
        let ra = ch.min(64);
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
            pooled: Buf::zeroed((ch / 4 + 2) * 512 * 4)?,
            gids,
            gplan: Buf::zeroed((2 + 2 * g + ch * g) * 4)?,
            ox: Buf::new(if ch > crate::f8::MOE_TOKENS { crate::f8::MOE_TOKENS * dims.heads * 512 * 2 } else { 16 })?,
            oy: Buf::new(if ch > crate::f8::MOE_TOKENS { crate::f8::MOE_TOKENS * d.width(crate::dynamic::Slot::Mid) * 2 } else { 16 })?,
            lmax,
            emax,
            ra,
        })
    }
}

/// A ratio-4 layer's indexer's pick: every indexer's (`Op::IdxPick`), its entries the indexer's compressor's (state `cache`), its scores the release's bf16 arithmetic, the
/// entries kept as they are.
fn pick(cache: usize, ratio: usize, heads: usize, top_k: usize, wscale: f32) -> Op {
    Op::IdxPick(IdxPick { heads, ratio, top_k, expand: false, tail: false, wscale, score: IdxScore::Release, keys: IdxKeys::State(cache), weighted: true })
}

/// A product's matrix: in the matrices' list, or the FP8 matrices'.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lin {
    Mat(usize),
    F8(usize),
}

impl Lin {
    fn op(self, x: Slot, y: Slot, name: &'static str) -> Op {
        match self {
            Lin::Mat(w) => Op::Linear { w, x, y, name },
            Lin::F8(w) => Op::F8Lin { w, x, y, name },
        }
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

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    let layers = ConfigV4::load(dir)?.layers;
    load_stage(dir, o, Stage::whole(layers))
}

/// Stage `st` of the model (its layers; the embedding where it is the first, the head where the last).
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    load_stage_hb(dir, o, st, None)
}

/// `load_stage` with a hummingbird directory's experts in the expert cache (`caching`: the stage's places and memory, src/hbg_store.rs).
pub fn load_stage_hb(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let c = ConfigV4::load(dir)?;
    if c.head_dim != 512
        || c.rope_dim != 64
        || c.idx_dim != 128
        || c.hc_mult != 4
        || c.hidden % 512 != 0
        || c.q_lora % 128 != 0
        || c.o_lora % 128 != 0
        || c.inter % 128 != 0
    {
        return err(format!("heads of {} ({} rotary), indexer heads of {}, {} streams, widths {} / {} / {} / {}: this engine runs heads of 512 (64 rotary), indexer heads of 128, 4 streams and widths of 128s", c.head_dim, c.rope_dim, c.idx_dim, c.hc_mult, c.hidden, c.q_lora, c.o_lora, c.inter));
    }
    if c.idx_heads > 64 {
        return err(format!("{} indexer heads: this engine runs up to 64", c.idx_heads));
    }
    let dev = Dev::new(o.device)?;
    let ck = io(Checkpoint::open(dir))?;
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let store = caching.map(crate::hbg_store::store_for);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let layout = o.layout.unwrap_or(Layout::Tiered);
    let (d, hd, nh, ch) = (c.hidden, c.head_dim, c.heads, o.chunk.max(1));
    let ctx = o.ctx.div_ceil(16) * 16;
    let read = |name: &str| io(ck.read(name));
    let fp8 = |name: &str| -> Result<(Vec<u16>, usize, usize)> {
        let (_, t) = ck.get(&format!("{name}.weight")).ok_or_else(|| Error(format!("no tensor {name}.weight")))?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        Ok((fp8_bf16(&read(&format!("{name}.weight"))?, &read(&format!("{name}.scale"))?, rows, cols)?, rows, cols))
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
    let small = |name: &str| -> Result<Buf> { upload_bytes(&read(name)?) };
    // the lossless level holds the large FP8 matrices in the library's FP8 format, as released (6.7 bits a weight, read as they are), the rest as the bf16 they decode to, packed lossless; another
    // level packs the bf16 of every one
    let vo = V4Opts::of(layout, o.ctx);
    let exact8 = vo.exact8;
    let lib = crate::ffi::Lib::find()?;
    let mut f8: Vec<crate::f8::F8Mat> = Vec::new();
    // rows [r0, r1) of FP8 matrix `name` (128-row blocks): the FP8 list's index, or the matrices'
    // (`groups`: the rows a stack of that many groups' matrices, each sliced on its own across a group of GPUs)
    let mut fp8_mat = |mats: &mut Vec<Mat>, name: &str, rows: Option<(usize, usize)>, groups: usize| -> Result<Lin> {
        let (_, t) = ck.get(&format!("{name}.weight")).ok_or_else(|| Error(format!("no tensor {name}.weight")))?;
        let (all, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let (r0, r1) = rows.unwrap_or((0, all));
        let (q, sc) = (read(&format!("{name}.weight"))?, read(&format!("{name}.scale"))?);
        let gc = cols.div_ceil(128);
        let what = if rows.is_some() { format!("{name} rows {r0}..{r1}") } else { name.to_string() };
        // (the FP8 format where it is the faster: a large matrix, whose fewer bytes outrun its heavier decode; 2026-10-18 on the 4080 SUPER, a token's product read from memory: 33M weights
        // 55-59 us against the bf16 pack's 70, 8M and fewer as fast or slower; GLYD_V4_F8_MIN: the weights from which, 0 every one)
        if vo.f8(r1 - r0, cols, r0) {
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
            return Ok(Lin::F8(f8.len() - 1));
        }
        let (q, sc) = (&q[r0 * cols..r1 * cols], &sc[r0 / 128 * gc..r1.div_ceil(128) * gc]);
        mats.push(pack(&fp8_bf16(q, sc, r1 - r0, cols)?, r1 - r0, cols, &what)?);
        Ok(Lin::Mat(mats.len() - 1))
    };
    let (mut mats, mut sm, mut moe, mut rings, mut states, mut layers): (Vec<Mat>, Vec<Buf>, Vec<Box<dyn MoeBlock>>, Vec<Ring>, Vec<usize>, Vec<Vec<Op>>) =
        Default::default();
    let put = |sm: &mut Vec<Buf>, b: Buf| {
        sm.push(b);
        sm.len() - 1
    };
    let (ol, hpg) = (c.o_lora, nh / c.o_groups);
    for l in 0..c.layers {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        crate::status::STATUS.beat();
        let (p, a) = (format!("layers.{l}"), format!("layers.{l}.attn"));
        let ratio = c.ratios[l];
        let rope = (ratio != 0) as usize;
        let hc = |sm: &mut Vec<Buf>, b: &str| -> Result<[usize; 3]> {
            Ok([put(sm, small(&format!("{p}.hc_{b}_fn"))?), put(sm, small(&format!("{p}.hc_{b}_base"))?), put(sm, small(&format!("{p}.hc_{b}_scale"))?)])
        };
        let [fa, ba, sa] = hc(&mut sm, "attn")?;
        let attn_norm = put(&mut sm, small(&format!("{p}.attn_norm.weight"))?);
        let q_norm = put(&mut sm, small(&format!("{a}.q_norm.weight"))?);
        let kv_norm = put(&mut sm, small(&format!("{a}.kv_norm.weight"))?);
        let sink = put(&mut sm, small(&format!("{a}.attn_sink"))?);
        let wq_a = fp8_mat(&mut mats, &format!("{a}.wq_a"), None, 1)?;
        let wq_b = fp8_mat(&mut mats, &format!("{a}.wq_b"), None, 1)?;
        let wkv = fp8_mat(&mut mats, &format!("{a}.wkv"), None, 1)?;
        // the groups' matrices: as one FP8 stack where the level holds FP8 as released (every group's product in one launch), else each its own, one after another in their list
        // (the stack where its whole matrix is held FP8: asked for otherwise, it would be packed, found not FP8 and left unused on the GPU)
        let wo_cols = ck.get(&format!("{a}.wo_a.weight")).map(|(_, t)| t.shape[1] as usize).ok_or_else(|| Error(format!("no tensor {a}.wo_a.weight")))?;
        let stack = if vo.stack(c.o_groups * ol, wo_cols, ol) {
            Some(fp8_mat(&mut mats, &format!("{a}.wo_a"), None, c.o_groups)?).filter(|x| matches!(x, Lin::F8(_)))
        } else {
            None
        };
        let (wo_a0, wo_a_f8) = match stack {
            Some(Lin::F8(i)) => (i, true),
            _ => {
                let groups: Vec<Lin> =
                    (0..c.o_groups).map(|g| fp8_mat(&mut mats, &format!("{a}.wo_a"), Some((g * ol, (g + 1) * ol)), 1)).collect::<Result<_>>()?;
                let first = match groups[0] {
                    Lin::F8(i) => (i, true),
                    Lin::Mat(i) => (i, false),
                };
                if groups.iter().enumerate().any(|(g, x)| *x != if first.1 { Lin::F8(first.0 + g) } else { Lin::Mat(first.0 + g) }) {
                    return err(format!("{a}.wo_a: its groups' matrices are not one after another"));
                }
                first
            }
        };
        let wo_b = fp8_mat(&mut mats, &format!("{a}.wo_b"), None, 1)?;
        let ring = rings.len();
        rings.push(Ring::keys(hd, 1, c.window));
        let mut ops = vec![
            Op::StreamsPre { mix: StreamsMix::Hc { f: fa, base: ba, scale: sa }, norm: attn_norm, round: Rounding::Once },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            wq_a.op(Slot::Hq, Slot::Proj, "attention: q down"),
            Op::RmsNorm { x: Slot::Proj, w: q_norm, y: Slot::Qr, weight: NormW::Plain, eps: c.eps },
            Op::Fp8 { x: Slot::Qr, y: Slot::Proj, block: 128 },
            wq_b.op(Slot::Proj, Slot::Qo, "attention: q up"),
            Op::CompQ { rope, heads: nh },
            wkv.op(Slot::Hq, Slot::Kv, "attention: kv"),
            Op::CompKv { norm: kv_norm, rope, ring },
        ];
        // a compressor: its two products stacked (kv's rows, then the gate's), its pooling's offsets, its norm; its cache and state
        let comp = |sm: &mut Vec<Buf>, states: &mut Vec<usize>, q: &str, dd: usize, indexer: bool| -> Result<Op> {
            let coff = 1 + (ratio == 4) as usize;
            let mut wb = read(&format!("{q}.wkv.weight"))?;
            wb.extend(read(&format!("{q}.wgate.weight"))?);
            let w = put(sm, upload_bytes(&wb)?);
            let ape = put(sm, small(&format!("{q}.ape"))?);
            let norm = put(sm, small(&format!("{q}.norm.weight"))?);
            let cache = states.len();
            states.push((ctx / ratio + 2) * dd * 2);
            states.push(coff * ratio * coff * dd * 4);
            states.push(coff * ratio * coff * dd * 4);
            Ok(Op::Compress { w, ape, norm, d: dd, ratio, rope, cache, state: cache + 1, indexer })
        };
        let mut cache = None;
        if ratio == 4 {
            let iq = fp8_mat(&mut mats, &format!("{a}.indexer.wq_b"), None, 1)?;
            let wp: Vec<u16> = read(&format!("{a}.indexer.weights_proj.weight"))?.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let iw = mats.len();
            mats.push(pack(&wp, c.idx_heads, d, &format!("{a}.indexer.weights_proj"))?);
            let ic = comp(&mut sm, &mut states, &format!("{a}.indexer.compressor"), c.idx_dim, true)?;
            let Op::Compress { cache: icache, .. } = ic else { unreachable!() };
            let wscale = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
            ops.extend([
                iq.op(Slot::Proj, Slot::Iq, "indexer: q"),
                Op::IdxQ { rope, heads: c.idx_heads },
                ic,
                Op::Linear { w: iw, x: Slot::H, y: Slot::Iw, name: "indexer: weights" },
                pick(icache, ratio, c.idx_heads, c.idx_topk, wscale),
            ]);
        }
        if ratio > 0 {
            let mc = comp(&mut sm, &mut states, &format!("{a}.compressor"), hd, false)?;
            if let Op::Compress { cache: k, .. } = mc {
                cache = Some(k);
            }
            ops.push(mc);
        }
        ops.extend([
            Op::CompAttn { ring, cache, ratio, sink, rope, heads: nh, hpg },
            if stack.is_some() {
                Op::OGroupLoraStack { w: wo_a0, groups: c.o_groups, ol, y: Slot::M, fp8: true }
            } else {
                Op::OGroupLora { w: wo_a0, in_f8: wo_a_f8, groups: c.o_groups, hpg, ol, y: Slot::M, fp8: true }
            },
            wo_b.op(Slot::M, Slot::O, "attention: o"),
            Op::StreamsPost { round: Rounding::Once, gated: false },
        ]);
        // the mixture of experts
        let [ff, bf, sf] = hc(&mut sm, "ffn")?;
        let ffn_norm = put(&mut sm, small(&format!("{p}.ffn_norm.weight"))?);
        let f = format!("{p}.ffn");
        let (e, k, i) = (c.experts, c.top_k, c.inter);
        // each expert's two matrices in the smallest layout that holds the layer's experts exactly (one pack for the stack, so one layout for all of them): the FP4 layout with packed scales (4.07
        // bits a weight), else the FP4 layout as released (4.25), else five-bit runs (5.5; GLYD_V4_FP4=0 asks for them); packed on all the cores (an expert's pack is its rows of the stacked matrix's:
        // block layouts have fixed strides), then joined
        // (a group of GPUs: the FP4 layouts alone, each expert's rows sliced)
        // (the host's pairs of the cache: the release's SwiGLU with its clamp, weighed after the second product, where the GPU weighs the activation and rounds it to FP8)
        let act = Act::SwiGlu { limit: (c.swiglu_limit > 0.0).then_some(c.swiglu_limit) };
        let hbx =
            crate::hbg_store::hb_layer(hb.as_ref(), l, tpr.is_some(), caching.zip(store.as_ref()), act, crate::cpu_hbg::Finish::Rotated, dev.ordinal, c.top_k)?;
        let mut tiered = None;
        let (up, down) = if hbx.is_some() {
            (ExpertStack::shape_only(e, 2 * i, d), ExpertStack::shape_only(e, d, i))
        } else {
            let kinds: &[u8] = if tpr.is_some() {
                &[2, 1]
            } else if crate::tune!("GLYD_V4_FP4").as_deref() != Some("0") {
                &[2, 1, 0]
            } else {
                &[0]
            };
            let one = |x: usize, kind: u8| -> Result<Option<(glyd_gpu::pack::Tiered, glyd_gpu::pack::Tiered)>> {
                let raw = |wn: &str| -> Result<(Vec<u8>, Vec<u8>)> {
                    let q = format!("{f}.experts.{x}.{wn}");
                    Ok((read(&format!("{q}.weight"))?, read(&format!("{q}.scale"))?))
                };
                if kind > 0 {
                    let pack = if kind == 2 { glyd_gpu::pack::pack_fp4r_checked } else { glyd_gpu::pack::pack_fp4_checked };
                    let ((mut uq, mut ue), (gq, ge), (dq, de)) = (raw("w1")?, raw("w3")?, raw("w2")?);
                    // (an expert's first matrix: w1's rows, the gate, then w3's, the up)
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
                let runs = |wn: &str, rows: usize, cols: usize| -> Result<(Vec<i8>, Vec<u16>)> {
                    let (q, s) = raw(wn)?;
                    fp4_runs(&q, &s, rows, cols)
                };
                let ((mut uc, mut us), (gc, gs)) = (runs("w1", i, d)?, runs("w3", i, d)?);
                uc.extend(gc);
                us.extend(gs);
                let (dc, ds) = runs("w2", d, i)?;
                let bad = || Error(format!("{f}.experts.{x}: not five-bit runs"));
                Ok(Some((
                    glyd_gpu::pack::pack_int5_runs_checked(&uc, &us, 2 * i, d).ok_or_else(bad)?,
                    glyd_gpu::pack::pack_int5_runs_checked(&dc, &ds, d, i).ok_or_else(bad)?,
                )))
            };
            let (mut ups, mut downs) = (Vec::with_capacity(e), Vec::with_capacity(e));
            for &kind in kinds {
                (ups.clear(), downs.clear());
                let mut all = true;
                crate::weights::ordered(e, threads, 2 * threads, &|x| one(x, kind), |_, p| {
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
                    "{f}: experts the FP4 layouts do not hold exactly{}",
                    if tpr.is_some() { " (across GPUs, the five-bit runs are not taken)" } else { "" }
                ));
            }
            let (ju, jd) = (Joined::of(&ups.iter().collect::<Vec<_>>())?, Joined::of(&downs.iter().collect::<Vec<_>>())?);
            drop((ups, downs));
            match caching.zip(store.as_ref()) {
                // (the expert cache, R12: places for `places` experts a layer, every expert in the computer's memory, the misses copied in)
                Some((cg, st)) => {
                    if tpr.is_some() || cg.ram.is_some() {
                        return err("DeepSeek-V4's experts in the expert cache: on one GPU a stage, every expert in the computer's memory (the disk holds hummingbird directories' experts alone)");
                    }
                    // (the host computes the misses the plan gives it where the experts are in an exact FP4 layout, as the GPU's act_w does: src/cpu_fp4.rs)
                    let fp4 = [glyd_gpu::pack::FP4_TAG, glyd_gpu::pack::FP4R_TAG].contains(&ju.tag);
                    let half = (fp4 && st.wants_cpu()).then_some(crate::offload::HostHalf::Fp4 {
                        inter: i,
                        latent: d,
                        packed: ju.tag == glyd_gpu::pack::FP4R_TAG,
                        act: crate::cpu_fp4::Act::Release { lim: if c.swiglu_limit > 0.0 { c.swiglu_limit } else { f32::INFINITY }, fp8: vo.sim },
                    });
                    let (u, d, cache) = crate::tiering::cached_stacks(&ju, &jd, cg.places, st, l, dev.ordinal, half)?;
                    tiered = Some(cache);
                    (u, d)
                }
                None => {
                    let (mut up, mut down) = (ExpertStack::upload(&ju)?, ExpertStack::upload(&jd)?);
                    if let Some((_, n)) = tpr {
                        (up.tp_rows, down.tp_rows) = (up.rows * n, down.rows * n);
                    }
                    (up, down)
                }
            }
        };
        // the shared expert: as released in the FP8 format where the level holds FP8 so (its gate and up stacked, its down), else the bf16 it decodes to, packed
        let shared8 = if exact8 {
            let f8 = |names: &[&str], rows: usize, cols: usize| -> Result<crate::f8::F8Mat> {
                let (mut q, mut sc) = (Vec::new(), Vec::new());
                for n in names {
                    q.extend(read(&format!("{f}.shared_experts.{n}.weight"))?);
                    sc.extend(read(&format!("{f}.shared_experts.{n}.scale"))?.iter().map(|&b| crate::deepseek_v4::e8m0(b)));
                }
                // (a group of GPUs: this GPU's rows, in whole blocks of 128 of the scales)
                if let Some((rank, n)) = tpr.filter(|&(_, n)| rows % (128 * n) == 0) {
                    let (r0, rn, gc) = (rank * rows / n, rows / n, cols.div_ceil(128));
                    let mut m = crate::f8::F8Mat::pack(
                        &dev,
                        &lib,
                        &q[r0 * cols..(r0 + rn) * cols],
                        &sc[r0 / 128 * gc..(r0 + rn) / 128 * gc],
                        rn,
                        cols,
                        cfg!(feature = "dev"),
                    )?;
                    m.tp_rows = rows;
                    return Ok(m);
                }
                crate::f8::F8Mat::pack(&dev, &lib, &q, &sc, rows, cols, cfg!(feature = "dev"))
            };
            Some((f8(&["w1", "w3"], 2 * i, d)?, f8(&["w2"], d, i)?))
        } else {
            None
        };
        let shared = match &shared8 {
            Some(_) => Shared { inter: i, up: PackBuf::shape_only(2 * i, d)?, down: PackBuf::shape_only(d, i)?, gate: None },
            None => {
                let (w1, _, _) = fp8(&format!("{f}.shared_experts.w1"))?;
                let (w3, _, _) = fp8(&format!("{f}.shared_experts.w3"))?;
                let (w2, _, _) = fp8(&format!("{f}.shared_experts.w2"))?;
                let su = [w1, w3].concat();
                let packed = |w: &[u16], rows: usize, cols: usize, name: &str| -> Result<PackBuf> {
                    match pack(w, rows, cols, name)? {
                        Mat::Packed(p) => Ok(p),
                        Mat::Dense(..) => err(format!("{name}: rows of a block layout are wanted")),
                    }
                };
                Shared {
                    inter: i,
                    up: packed(&su, 2 * i, d, &format!("{f}.shared_experts.w13"))?,
                    down: packed(&w2, d, i, &format!("{f}.shared_experts.w2"))?,
                    gate: None,
                }
            }
        };
        let hash = l < c.hash_layers;
        let cfg = MoeCfg {
            experts: e,
            router_hidden: d,
            hidden: d,
            inter: i,
            router: Router { scoring: Scoring::SqrtSoftplus, top_k: k, renorm: true, scale: c.route_scale, bf16_logits: false },
            act: Act::SwiGlu { limit: (c.swiglu_limit > 0.0).then_some(c.swiglu_limit) },
        };
        let w = MoeWeights {
            router: small(&format!("{f}.gate.weight"))?,
            bias: if hash { None } else { Some(small(&format!("{f}.gate.bias"))?) },
            escale: None,
            up,
            down: Down::Block(down),
        };
        let block = moe.len();
        let layer = match hbx {
            Some(x) => x.into_layer(cfg, w.router, w.bias, None, Some(shared))?,
            None => match tiered {
                Some(cache) => MoeLayer::new(cfg, w, Some(shared))?.with_cache(cache)?,
                None => MoeLayer::new(cfg, w, Some(shared))?,
            },
        };
        moe.push(Box::new(ResidentMoe(match shared8 {
            Some((u, dn)) => layer.with_shared8(u, dn)?,
            None => layer,
        })));
        let hash = if hash { Some(put(&mut sm, small(&format!("{f}.gate.tid2eid"))?)) } else { None };
        ops.extend([
            Op::StreamsPre { mix: StreamsMix::Hc { f: ff, base: bf, scale: sf }, norm: ffn_norm, round: Rounding::Once },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            Op::Moe(v4_moe(block, hash, vo.sim)),
            Op::StreamsPost { round: Rounding::Once, gated: false },
        ]);
        layers.push(ops);
    }
    // the ends: the embedding where this stage is the first, the head (its mixing, the norm, the LM head) where the last; another stage's are empty places
    let placeholder = || -> Result<Mat> { Ok(Mat::Dense(Buf::new(2)?, 0, 0)) };
    let words = |name: &str| -> Result<Vec<u16>> { Ok(read(name)?.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()) };
    let embed = mats.len();
    // (the embedding whole on every GPU: a token's row is read from it)
    mats.push(if st.first { pack_as(&words("embed.weight")?, c.vocab, d, "embed", false)? } else { placeholder()? });
    let lm = mats.len();
    mats.push(if st.last { pack(&words("head.weight")?, c.vocab, d, "head")? } else { placeholder()? });
    if st.last {
        let f = put(&mut sm, small("hc_head_fn")?);
        let b = put(&mut sm, small("hc_head_base")?);
        let s = put(&mut sm, small("hc_head_scale")?);
        let norm = put(&mut sm, small("norm.weight")?);
        layers[c.layers - 1].push(Op::StreamsHead { mix: HeadMix::Weighted([f, b, s]), norm, round: Rounding::Once });
    }
    let mp = ctx + ch + 16;
    let ropes = vec![
        tables(&crate::deepseek_v4::freqs(c.rope_dim, c.rope_theta, None), mp)?,
        tables(&crate::deepseek_v4::freqs(c.rope_dim, c.compress_theta, Some(c.yarn)), mp)?,
    ];
    let mut widths = [0usize; super::SLOTS];
    let wmax = if c.ratios.contains(&4) { 2 * hd } else { hd };
    for (s, w) in [
        (Slot::X, d),
        (Slot::S, 4 * d),
        (Slot::H, d),
        (Slot::Hq, d),
        (Slot::Proj, c.q_lora),
        (Slot::Qr, c.q_lora),
        (Slot::Qo, nh * hd),
        (Slot::Kv, hd),
        (Slot::Cp, 4 * wmax),
        (Slot::Iq, c.idx_heads * c.idx_dim),
        (Slot::Iw, c.idx_heads),
        (Slot::Att, nh * hd),
        (Slot::Mid, c.o_groups * ol),
        (Slot::M, c.o_groups * ol),
        (Slot::O, d),
    ] {
        widths[s as usize] = w;
    }
    let all_ratio = c.ratios.iter().copied().filter(|&r| r != 0 && r != 4).min().unwrap_or(0);
    let desc = ModelDesc {
        family: "DeepSeek-V4",
        hidden: d,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
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
        eos: c.eos.clone(),
        post: None,
        hc: (c.hc_eps, c.hc_iters as i32),
        logit_mult: 0.0,
        v4: Some(V4Dims { heads: nh, idx_heads: c.idx_heads, topk: c.idx_topk, window: c.window, all_ratio, sim: vo.sim, o_groups: c.o_groups }),
    };
    drop(fp8_mat);
    // (the loader's own operations, as it indexed what it loaded, held to the layer kinds' description of the config: one model, or it is not loaded)
    let desc = super::lower::agree(lowered(dir, &c, &st, &vo)?, &desc, &dir.join("config.json").display().to_string())?;
    Dyn::from_loaded_f8(dev, desc, mats, f8, sm, moe, ropes, layout, st, o)
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

/// A DeepSeek-V4 layer's experts (block `block`; the first layers' table of experts by token, small int64 [vocab, k], `hash`): the router reads H, the experts the FP8-rounded
/// Hq, each pair's weight in its activation before the second product (rounded to FP8 where `fp8`), into O.
pub(crate) fn v4_moe(block: usize, hash: Option<usize>, fp8: bool) -> MoeOp {
    MoeOp { x: Slot::Hq, hash, weigh_act: true, fp8, ..MoeOp::of(block) }
}

/// DeepSeek-V4's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::deepseek_v4`), held to `describe`.
pub fn lowered(dir: &Path, c: &ConfigV4, st: &Stage, vo: &V4Opts) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let lowered = super::lower::deepseek_v4(&kinds, c.eos.clone(), c.max_pos, st, vo)?;
    super::lower::agree(lowered, &describe(c, st, vo)?, &label)
}

/// The executor's description of DeepSeek-V4 for stage `st` with the options `vo`, as the loader indexes what it loads (`v4_desc`).
pub fn describe(c: &ConfigV4, st: &Stage, vo: &V4Opts) -> Result<ModelDesc> {
    if c.inter % 128 != 0 {
        return err(format!("experts of width {}: this engine runs widths of 128s", c.inter));
    }
    let s = V4Shape {
        d: c.hidden,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        eos: c.eos.clone(),
        nh: c.heads,
        hd: c.head_dim,
        rope_dim: c.rope_dim,
        q_lora: c.q_lora,
        ol: c.o_lora,
        og: c.o_groups,
        wo_cols: c.heads * c.head_dim / c.o_groups.max(1),
        idx: (c.idx_heads, c.idx_dim, c.idx_topk),
        window: c.window,
        ratios: c.ratios.clone(),
        hash: (0..c.layers).map(|l| l < c.hash_layers).collect(),
        streams: c.hc_mult,
        hc: (c.hc_eps, c.hc_iters),
        round: Rounding::Once,
    };
    v4_desc(&s, st, vo)
}

/// DeepSeek-V4's widths and layers as the executor takes them: the width, vocabulary, norms' epsilon, context and end tokens; the heads, their size and rotary
/// part; the queries' low rank, the output's groups (rank, count, columns); the indexer's heads, their size and its choices; the window, each layer's compression
/// ratio and whether its experts are hash-routed; the streams, their mix's epsilon and iterations.
pub(crate) struct V4Shape {
    pub d: usize,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    pub eos: Vec<i64>,
    pub nh: usize,
    pub hd: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub ol: usize,
    pub og: usize,
    pub wo_cols: usize,
    pub idx: (usize, usize, usize),
    pub window: usize,
    pub ratios: Vec<usize>,
    pub hash: Vec<bool>,
    pub streams: usize,
    pub hc: (f32, usize),
    /// The streams' arithmetic (`StreamsP::rounding`: the release's, `Once`).
    pub round: Rounding,
}

/// The executor's DeepSeek-V4 for stage `st`. Each layer's small tensors: the attention's stream mix (projection, base, scale), its norm, the queries' and keys' norms,
/// the sink; a ratio-4 layer's indexer compressor's (its two products stacked, its offsets, its norm), a compressing layer's compressor's; the experts' stream mix and
/// norm, and a hash-routed layer's token-to-expert table; the last stage's head mix and final norm after the last layer's. Its matrices, the FP8 ones in their own list
/// where `vo` holds them FP8: the queries' down and up products, the keys' and values', the output groups' (one stack, or one a group), the output's; a ratio-4 layer's
/// indexer queries' and weights' products; then the embedding and the head. A layer's experts are their block's.
pub(crate) fn v4_desc(s: &V4Shape, st: &Stage, vo: &V4Opts) -> Result<ModelDesc> {
    let (d, nh, hd, ol, og) = (s.d, s.nh, s.hd, s.ol, s.og);
    let (ih, idim, itop) = s.idx;
    if hd != 512 || s.rope_dim != 64 || idim != 128 || s.streams != 4 || d % 512 != 0 || s.q_lora % 128 != 0 || ol % 128 != 0 {
        return err(format!(
            "heads of {hd} ({} rotary), indexer heads of {idim}, {} streams, widths {d} / {} / {ol}: this engine runs heads of 512 (64 rotary), indexer heads of 128, 4 streams and widths of 128s",
            s.rope_dim, s.streams, s.q_lora
        ));
    }
    if ih > 64 {
        return err(format!("{ih} indexer heads: this engine runs up to 64"));
    }
    let hpg = nh / og;
    let layers_n = s.ratios.len();
    #[derive(Default)]
    struct Count {
        mats: usize,
        f8: usize,
        small: usize,
    }
    let lin = |n: &mut Count, rows: usize, cols: usize, r0: usize| -> Lin {
        if vo.f8(rows, cols, r0) {
            n.f8 += 1;
            Lin::F8(n.f8 - 1)
        } else {
            n.mats += 1;
            Lin::Mat(n.mats - 1)
        }
    };
    let small = |n: &mut Count, k: usize| -> usize {
        n.small += k;
        n.small - k
    };
    let (mut n, mut n_moe, mut rings, mut states, mut layers) = (Count::default(), 0, Vec::new(), Vec::new(), Vec::new());
    for l in 0..layers_n {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let ratio = s.ratios[l];
        let rope = (ratio != 0) as usize;
        let a = small(&mut n, 7);
        let (fa, ba, sa, attn_norm, q_norm, kv_norm, sink) = (a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6);
        let wq_a = lin(&mut n, s.q_lora, d, 0);
        let wq_b = lin(&mut n, nh * hd, s.q_lora, 0);
        let wkv = lin(&mut n, hd, d, 0);
        let stack = vo.stack(og * ol, s.wo_cols, ol);
        let (wo_a0, wo_a_f8) = if stack {
            n.f8 += 1;
            (n.f8 - 1, true)
        } else {
            let first = lin(&mut n, ol, s.wo_cols, 0);
            for g in 1..og {
                lin(&mut n, ol, s.wo_cols, g * ol);
            }
            match first {
                Lin::F8(i) => (i, true),
                Lin::Mat(i) => (i, false),
            }
        };
        let wo_b = lin(&mut n, d, og * ol, 0);
        let ring = rings.len();
        rings.push(Ring::keys(hd, 1, s.window));
        let mut ops = vec![
            Op::StreamsPre { mix: StreamsMix::Hc { f: fa, base: ba, scale: sa }, norm: attn_norm, round: s.round },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            wq_a.op(Slot::Hq, Slot::Proj, "attention: q down"),
            Op::RmsNorm { x: Slot::Proj, w: q_norm, y: Slot::Qr, weight: NormW::Plain, eps: s.eps },
            Op::Fp8 { x: Slot::Qr, y: Slot::Proj, block: 128 },
            wq_b.op(Slot::Proj, Slot::Qo, "attention: q up"),
            Op::CompQ { rope, heads: nh },
            wkv.op(Slot::Hq, Slot::Kv, "attention: kv"),
            Op::CompKv { norm: kv_norm, rope, ring },
        ];
        let comp = |n: &mut Count, states: &mut Vec<usize>, dd: usize, indexer: bool| -> Op {
            let coff = 1 + (ratio == 4) as usize;
            let w = small(n, 3);
            let cache = states.len();
            states.push((vo.ctx / ratio + 2) * dd * 2);
            states.push(coff * ratio * coff * dd * 4);
            states.push(coff * ratio * coff * dd * 4);
            Op::Compress { w, ape: w + 1, norm: w + 2, d: dd, ratio, rope, cache, state: cache + 1, indexer }
        };
        if ratio == 4 {
            let iq = lin(&mut n, ih * idim, s.q_lora, 0);
            let iw = n.mats;
            n.mats += 1;
            let icache = states.len();
            let ic = comp(&mut n, &mut states, idim, true);
            let wscale = (idim as f32).powf(-0.5) * (ih as f32).powf(-0.5);
            ops.extend([
                iq.op(Slot::Proj, Slot::Iq, "indexer: q"),
                Op::IdxQ { rope, heads: ih },
                ic,
                Op::Linear { w: iw, x: Slot::H, y: Slot::Iw, name: "indexer: weights" },
                pick(icache, ratio, ih, itop, wscale),
            ]);
        }
        let mut cache = None;
        if ratio > 0 {
            cache = Some(states.len());
            ops.push(comp(&mut n, &mut states, hd, false));
        }
        ops.extend([
            Op::CompAttn { ring, cache, ratio, sink, rope, heads: nh, hpg },
            if stack {
                Op::OGroupLoraStack { w: wo_a0, groups: og, ol, y: Slot::M, fp8: true }
            } else {
                Op::OGroupLora { w: wo_a0, in_f8: wo_a_f8, groups: og, hpg, ol, y: Slot::M, fp8: true }
            },
            wo_b.op(Slot::M, Slot::O, "attention: o"),
            Op::StreamsPost { round: s.round, gated: false },
        ]);
        let f = small(&mut n, 4);
        let block = n_moe;
        n_moe += 1;
        let hash = s.hash[l].then(|| small(&mut n, 1));
        ops.extend([
            Op::StreamsPre { mix: StreamsMix::Hc { f, base: f + 1, scale: f + 2 }, norm: f + 3, round: s.round },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            Op::Moe(v4_moe(block, hash, vo.sim)),
            Op::StreamsPost { round: s.round, gated: false },
        ]);
        layers.push(ops);
    }
    let (embed, lm) = (n.mats, n.mats + 1);
    if st.last {
        let h = small(&mut n, 4);
        layers.last_mut().ok_or_else(|| Error("a model of no layers".into()))?.push(Op::StreamsHead {
            mix: HeadMix::Weighted([h, h + 1, h + 2]),
            norm: h + 3,
            round: s.round,
        });
    }
    let mut widths = [0usize; super::SLOTS];
    let wmax = if s.ratios.contains(&4) { 2 * hd } else { hd };
    for (sl, w) in [
        (Slot::X, d),
        (Slot::S, 4 * d),
        (Slot::H, d),
        (Slot::Hq, d),
        (Slot::Proj, s.q_lora),
        (Slot::Qr, s.q_lora),
        (Slot::Qo, nh * hd),
        (Slot::Kv, hd),
        (Slot::Cp, 4 * wmax),
        (Slot::Iq, ih * idim),
        (Slot::Iw, ih),
        (Slot::Att, nh * hd),
        (Slot::Mid, og * ol),
        (Slot::M, og * ol),
        (Slot::O, d),
    ] {
        widths[sl as usize] = w;
    }
    let all_ratio = s.ratios.iter().copied().filter(|&r| r != 0 && r != 4).min().unwrap_or(0);
    Ok(ModelDesc {
        family: "DeepSeek-V4",
        hidden: d,
        vocab: s.vocab,
        eps: s.eps,
        max_pos: s.max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
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
        eos: s.eos.clone(),
        post: None,
        hc: (s.hc.0, s.hc.1 as i32),
        logit_mult: 0.0,
        v4: Some(V4Dims { heads: nh, idx_heads: ih, topk: itop, window: s.window, all_ratio, sim: vo.sim, o_groups: og }),
    })
}
