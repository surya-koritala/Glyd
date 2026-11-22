//! DeepSeek-V4 (`DeepseekV4ForCausalLM`, src/deepseek_v4.rs) as a description: the four streams and their mixing around each layer's two blocks, the attention (the query's low rank, the shared key-value
//! head over the window and the compressed cache, the indexer's picks, the sink, the grouped low rank of its output) and the mixture of experts, each product's input quantized to FP8 as the release's are.
//! The weights load exactly: an FP8 matrix as the bf16 it is (packed at the level asked, lossless by default), the FP4 experts as five-bit runs (their codes and power-of-two scales: the same values),
//! the rest as the checkpoint holds it. The arithmetic is kernels/dsv4.cu's; src/dsv4_ref.rs is the CPU reference it is checked against (`dsv4check`).

use super::{Dyn, ModelDesc, Op, Opts, Ring, Slot, Stage, V4Dims};
use crate::deepseek_v4::{fp4_runs, fp8_bf16, ConfigV4};
use crate::dev::{Buf, Dev, Kernel};
use crate::ffi::{err, Error, Result};
use crate::kern35::Mat;
use crate::kern_nh::{MoeBlock, NhHost};
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeWeights, Router, Scoring, Shared};
use crate::moe_rows::Down;
use crate::weights::{as_bytes, pack_words, upload_bytes, upload_packed, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// kernels/dsv4.cu's kernels (and glm.cu's streams).
pub(crate) struct KernV4 {
    pub streams: Kernel,
    pub hc_proj: Kernel,
    pub hc_fin: Kernel,
    pub hc_post: Kernel,
    pub hc_head: Kernel,
    pub fp8: Kernel,
    pub q_prep: Kernel,
    pub kv_prep: Kernel,
    pub lin32: Kernel,
    pub lin32_1: Kernel,
    pub comp_pool: Kernel,
    pub comp_out: Kernel,
    pub idx_q: Kernel,
    pub idx_score: Kernel,
    pub topk: Kernel,
    pub attn_scores: Kernel,
    pub attn_soft: Kernel,
    pub attn_out: Kernel,
    pub ogather: Kernel,
}

impl KernV4 {
    fn load(dev: &Dev) -> Result<KernV4> {
        Ok(KernV4 {
            streams: dev.kernel(kname!("glm_streams"))?,
            hc_proj: dev.kernel(kname!("v4_hc_proj"))?,
            hc_fin: dev.kernel(kname!("v4_hc_fin"))?,
            hc_post: dev.kernel(kname!("v4_hc_post"))?,
            hc_head: dev.kernel(kname!("v4_hc_head"))?,
            fp8: dev.kernel(kname!("v4_fp8"))?,
            q_prep: dev.kernel(kname!("v4_q_prep"))?,
            kv_prep: dev.kernel(kname!("v4_kv_prep"))?,
            lin32: dev.kernel(kname!("v4_lin32"))?,
            lin32_1: dev.kernel(kname!("v4_lin32_1"))?,
            comp_pool: dev.kernel(kname!("v4_comp_pool"))?,
            comp_out: dev.kernel(kname!("v4_comp_out"))?,
            idx_q: dev.kernel(kname!("v4_idx_q"))?,
            idx_score: dev.kernel(kname!("v4_idx_score"))?,
            topk: dev.kernel(kname!("v4_topk"))?,
            attn_scores: dev.kernel(kname!("v4_attn_scores"))?,
            attn_soft: dev.kernel(kname!("v4_attn_soft"))?,
            attn_out: dev.kernel(kname!("v4_attn_out"))?,
            ogather: dev.kernel(kname!("v4_ogather"))?,
        })
    }
}

/// The executor's DeepSeek-V4 buffers: the mixing weights a block keeps for its post, the mixing projection's partial sums, the attention's scores and softmax sums (`ra` rows at a time), the indexer's
/// scores, picks and their counts, a compressor's pooled entries, the tokens (the routing by token), the experts' weights of one.
pub(crate) struct V4Run {
    pub k: KernV4,
    pub dims: V4Dims,
    pub hcp: Buf,
    pub part: Buf,
    pub sc: Buf,
    pub den: Buf,
    pub isc: Buf,
    pub sel: Buf,
    pub cnt: Buf,
    pub pooled: Buf,
    pub tok: Buf,
    pub ones: Buf,
    /// the grouped low rank's pairs' groups (int64 [chunk, groups]: pair t groups + g is group g) and their plan
    pub gids: Buf,
    pub gplan: Buf,
    /// a part of a prompt's chunk of more than the grouped product's 64 rows (`Op::V4OLoraStack`): its rows of Att and of Mid, group by group, one after
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
        let ones = Buf::new(ch * dims.experts_k * 4)?;
        ones.write(0, &vec![1f32; ch * dims.experts_k])?;
        let g = dims.o_groups.max(1);
        let gids = Buf::new(ch * g * 8)?;
        gids.write(0, &(0..ch * g).map(|j| (j % g) as i64).collect::<Vec<i64>>())?;
        Ok(V4Run {
            k: KernV4::load(dev)?,
            dims: dims.clone(),
            hcp: Buf::zeroed(ch * 24 * 4)?,
            part: Buf::zeroed(ch * (4 * d.hidden / 256).max(1) * 25 * 4)?,
            sc: Buf::zeroed(ra * dims.heads * lmax * 4)?,
            den: Buf::zeroed(ra * dims.heads * 4)?,
            isc: Buf::zeroed(ra * emax * 4)?,
            sel: Buf::zeroed(ch * dims.topk.max(1) * 4)?,
            cnt: Buf::zeroed(ch * 4)?,
            pooled: Buf::zeroed((ch / 4 + 2) * 512 * 4)?,
            tok: Buf::zeroed(ch * 4)?,
            ones,
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

/// A layer's experts on the dynamic engine: the core's layer, run as the release computes it (`MoeLayer::forward_release`).
pub struct V4Moe(pub MoeLayer);

impl MoeBlock for V4Moe {
    unsafe fn forward(&self, _: &dyn NhHost, _: u64, _: usize, _: u64) -> Result<()> {
        err("DeepSeek-V4's experts run on the dynamic engine")
    }
    fn graphable(&self) -> bool {
        true
    }
    fn bytes(&self) -> usize {
        self.0.weights_bytes()
    }
    fn resident(&self) -> Option<(&MoeLayer, Option<(&Mat, &Mat)>)> {
        Some((&self.0, None))
    }
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
    let exact8 = matches!(layout, Layout::Tiered) && crate::tune!("GLYD_V4_F8").as_deref() != Some("0");
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
        let big = (r1 - r0) * cols >= crate::tune!("GLYD_V4_F8_MIN").and_then(|v| v.parse().ok()).unwrap_or(16 << 20);
        if exact8 && big && r0 % 128 == 0 {
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
        let stack = if exact8 && ol % 64 == 0 && crate::tune!("GLYD_V4_OSTACK").as_deref() != Some("0") {
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
        rings.push(Ring { d: hd, nk: 1, window: c.window });
        let mut ops = vec![
            Op::V4Pre { f: fa, base: ba, scale: sa, norm: attn_norm },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            wq_a.op(Slot::Hq, Slot::Proj, "attention: q down"),
            Op::RmsNorm { x: Slot::Proj, w: q_norm, y: Slot::Qr, plus1: false },
            Op::Fp8 { x: Slot::Qr, y: Slot::Proj, block: 128 },
            wq_b.op(Slot::Proj, Slot::Qo, "attention: q up"),
            Op::V4Q { rope, heads: nh },
            wkv.op(Slot::Hq, Slot::Kv, "attention: kv"),
            Op::V4Kv { norm: kv_norm, rope, ring },
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
            Ok(Op::V4Comp { w, ape, norm, d: dd, ratio, rope, cache, state: cache + 1, indexer })
        };
        let mut cache = None;
        if ratio == 4 {
            let iq = fp8_mat(&mut mats, &format!("{a}.indexer.wq_b"), None, 1)?;
            let wp: Vec<u16> = read(&format!("{a}.indexer.weights_proj.weight"))?.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let iw = mats.len();
            mats.push(pack(&wp, c.idx_heads, d, &format!("{a}.indexer.weights_proj"))?);
            let ic = comp(&mut sm, &mut states, &format!("{a}.indexer.compressor"), c.idx_dim, true)?;
            let Op::V4Comp { cache: icache, .. } = ic else { unreachable!() };
            let wscale = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
            ops.extend([
                iq.op(Slot::Proj, Slot::Iq, "indexer: q"),
                Op::V4IdxQ { rope, heads: c.idx_heads },
                ic,
                Op::Linear { w: iw, x: Slot::H, y: Slot::Iw, name: "indexer: weights" },
                Op::V4IdxPick { cache: icache, ratio, heads: c.idx_heads, topk: c.idx_topk, wscale },
            ]);
        }
        if ratio > 0 {
            let mc = comp(&mut sm, &mut states, &format!("{a}.compressor"), hd, false)?;
            if let Op::V4Comp { cache: k, .. } = mc {
                cache = Some(k);
            }
            ops.push(mc);
        }
        ops.extend([
            Op::V4Attn { ring, cache, ratio, sink, rope, heads: nh, hpg },
            if stack.is_some() {
                Op::V4OLoraStack { w: wo_a0, groups: c.o_groups, ol, y: Slot::M, fp8: true }
            } else {
                Op::V4OLora { w: wo_a0, in_f8: wo_a_f8, groups: c.o_groups, hpg, ol, y: Slot::M, fp8: true }
            },
            wo_b.op(Slot::M, Slot::O, "attention: o"),
            Op::V4Post,
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
        let mut up = ExpertStack::upload(&Joined::of(&ups.iter().collect::<Vec<_>>())?)?;
        let mut down = ExpertStack::upload(&Joined::of(&downs.iter().collect::<Vec<_>>())?)?;
        drop((ups, downs));
        if let Some((_, n)) = tpr {
            (up.tp_rows, down.tp_rows) = (up.rows * n, down.rows * n);
        }
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
        let layer = MoeLayer::new(cfg, w, Some(shared))?;
        moe.push(Box::new(V4Moe(match shared8 {
            Some((u, dn)) => layer.with_shared8(u, dn)?,
            None => layer,
        })));
        let hash = if hash { Some(put(&mut sm, small(&format!("{f}.gate.tid2eid"))?)) } else { None };
        ops.extend([
            Op::V4Pre { f: ff, base: bf, scale: sf, norm: ffn_norm },
            Op::Fp8 { x: Slot::H, y: Slot::Hq, block: 128 },
            Op::V4Moe { block, hash },
            Op::V4Post,
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
        layers[c.layers - 1].push(Op::V4Head { f, base: b, scale: s, norm });
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
        first: Op::V4Streams,
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
        hc: (0.0, 0),
        logit_mult: 0.0,
        v4: Some(V4Dims {
            hc_eps: c.hc_eps,
            iters: c.hc_iters as i32,
            heads: nh,
            idx_heads: c.idx_heads,
            topk: c.idx_topk,
            window: c.window,
            experts_k: c.top_k,
            all_ratio,
            sim: crate::tune!("GLYD_V4_SIM").as_deref() != Some("0"),
            o_groups: c.o_groups,
        }),
    };
    drop(fp8_mat);
    Dyn::from_loaded_f8(dev, desc, mats, f8, sm, moe, ropes, layout, st, o)
}
