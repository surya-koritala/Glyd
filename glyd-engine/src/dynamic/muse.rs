//! Muse Spark's text model (src/muse.rs) as a description: each layer's attention (q, k, v and the gate in one product; heads of 64 normed, the rotary on the window's layers; the output normed and gated),
//! its experts in the latent (a projection in, a centered norm, the experts with each output normed, the weighted sum, a projection out), the input-scaled norms and the residual gates; the normed
//! embedding, the final norm, the soft-cap of the logits times the output multiplier. bf16 as the checkpoint has it, packed at the level asked (lossless by default); the routers fp32, as they are. The
//! arithmetic is kernels/muse.cu's; src/muse_ref.rs is the CPU reference it is checked against (`musecheck`).

use super::{Dyn, Loaded, ModelDesc, Op, Opts, Ring, Slot, Stage};
use crate::dev::{Buf, Dev, Kernel};
use crate::ffi::{err, Error, Result};
use crate::model35::Mat;
use crate::model_nh::{MoeBlock, NhHost};
use crate::moe::{Act, ExpertStack, MoeCfg, MoeLayer, MoeWeights, Router};
use crate::moe_rows::Down;
use crate::muse::{plus_one, residual_gate, ConfigMuse};
use crate::weights::{as_bytes, pack_words, upload_bytes, upload_packed, Layout};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// kernels/muse.cu's kernels the executor launches (the experts' are moe.rs's).
pub struct KernMuse {
    pub qk: Kernel,
    pub attn: Kernel,
    pub ogate: Kernel,
    pub rms_in: Kernel,
    pub rgate: Kernel,
    pub softcap: Kernel,
}

impl KernMuse {
    pub fn load(dev: &Dev) -> Result<KernMuse> {
        Ok(KernMuse {
            qk: dev.kernel(kname!("muse_qk"))?,
            attn: dev.kernel(kname!("muse_attn"))?,
            ogate: dev.kernel(kname!("muse_ogate"))?,
            rms_in: dev.kernel(kname!("muse_rms_in"))?,
            rgate: dev.kernel(kname!("muse_rgate"))?,
            softcap: dev.kernel(kname!("muse_softcap"))?,
        })
    }
}

/// A layer's experts on the dynamic engine (`MoeLayer::forward_muse`, its router fp32 in `MoeWeights::router`).
struct MuseMoe(MoeLayer);

impl MoeBlock for MuseMoe {
    unsafe fn forward(&self, _: &dyn NhHost, _: u64, _: usize, _: u64) -> Result<()> {
        err("Muse Spark's experts run on the dynamic engine")
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

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

fn words(b: &[u8]) -> Vec<u16> {
    b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn values(w: &[u16]) -> Vec<f32> {
    w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

fn bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
}

pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_stage(dir, o, Stage::whole(ConfigMuse::load(dir)?.layers))
}

/// Stage `st` of the model (its layers; the embedding where it is the first, the final norm and the head where the last).
pub fn load_stage(dir: &Path, o: &Opts, st: Stage) -> Result<Dyn> {
    let c = ConfigMuse::load(dir)?;
    if c.head_dim != 64 || c.hidden > 8192 || c.latent > 4096 || c.top_k > 32 {
        return err(format!("heads of {}, a width of {}, a latent of {}, {} experts a token: this engine runs heads of 64, widths up to 8192, latents up to 4096 and up to 32 experts a token", c.head_dim, c.hidden, c.latent, c.top_k));
    }
    let dev = Dev::new(o.device)?;
    let ck = io(Checkpoint::open(dir))?;
    let layout = o.layout.unwrap_or(Layout::Tiered);
    let (d, nq, nk, hd, lat, inter, e) = (c.hidden, c.heads, c.kv_heads, c.head_dim, c.latent, c.inter, c.experts);
    let read = |n: &str| io(ck.read(n));
    // (every layer across a group of GPUs: this GPU's slice of a big matrix's rows, of each expert's: src/dynamic/tp.rs; the embedding whole)
    let sl = super::tp::slicer();
    let pack_as = |w: &[u16], rows: usize, cols: usize, name: &str, slice: bool| -> Result<Mat> {
        if let Some(t) = sl.filter(|t| slice && t.slices(rows, cols)) {
            let (r0, rn) = t.rows(rows);
            let mut pk = upload_packed(pack_words(&w[r0 * cols..(r0 + rn) * cols], rn, cols, layout, name)?)?;
            pk.tp_rows = rows;
            return Ok(Mat::Packed(pk));
        }
        Ok(Mat::Packed(upload_packed(pack_words(w, rows, cols, layout, name)?)?))
    };
    let pack = |w: &[u16], rows: usize, cols: usize, name: &str| pack_as(w, rows, cols, name, true);
    let small_bits = |v: &[u16]| upload_bytes(as_bytes(v));
    let mut l_ = Loaded::default();
    let (mut rings, mut layers) = (Vec::new(), Vec::new());
    let put = |v: &mut Vec<Buf>, b: Buf| {
        v.push(b);
        v.len() - 1
    };
    let ones = put(&mut l_.small, small_bits(&vec![0x3F80u16; d.max(lat)])?);
    let row = (2 * nq + 2 * nk) * hd;
    for l in 0..c.layers {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        crate::status::STATUS.beat();
        let p = format!("{}layers.{l}", c.prefix);
        let (a, m) = (format!("{p}.self_attn"), format!("{p}.mlp"));
        let vals = |n: &str| -> Result<Vec<f32>> { Ok(values(&words(&read(n)?))) };
        let centered = |small: &mut Vec<Buf>, n: &str| -> Result<usize> { Ok(put(small, small_bits(&bf16_bits(&plus_one(&vals(n)?)))?)) };
        let gate = |small: &mut Vec<Buf>, n: &str| -> Result<(usize, usize)> {
            let (al, be) = residual_gate(&vals(n)?, c.gate_temp);
            Ok((put(small, small_bits(&bf16_bits(&al))?), put(small, small_bits(&bf16_bits(&be))?)))
        };
        let w_in = centered(&mut l_.small, &format!("{p}.input_layernorm.weight"))?;
        let (aa, ba) = gate(&mut l_.small, &format!("{p}.post_attention_residual_gate.gate"))?;
        let w_ff = centered(&mut l_.small, &format!("{p}.pre_feedforward_layernorm.weight"))?;
        let w_pre = centered(&mut l_.small, &format!("{m}.pre_expert_norm.weight"))?;
        let w_post_e = centered(&mut l_.small, &format!("{m}.experts.post_expert_norm.weight"))?;
        let w_post_ff = centered(&mut l_.small, &format!("{p}.post_feedforward_layernorm.weight"))?;
        let (af, bfg) = gate(&mut l_.small, &format!("{p}.post_feedforward_residual_gate.gate"))?;
        // q, k, v and the gate as one product (Mix's row: q | k | v | gate)
        let mut qkvg = Vec::with_capacity(row * d);
        for n in ["q_proj", "k_proj", "v_proj", "gate_proj"] {
            qkvg.extend(words(&read(&format!("{a}.{n}.weight"))?));
        }
        let m0 = l_.mats.len();
        l_.mats.push(pack(&qkvg, row, d, &format!("{a}.qkvg"))?);
        drop(qkvg);
        l_.mats.push(pack(&words(&read(&format!("{a}.o_proj.weight"))?), d, nq * hd, &format!("{a}.o_proj"))?);
        l_.mats.push(pack(&words(&read(&format!("{m}.pre_expert_proj.weight"))?), lat, d, &format!("{m}.pre_expert_proj"))?);
        l_.mats.push(pack(&words(&read(&format!("{m}.post_expert_proj.weight"))?), d, lat, &format!("{m}.post_expert_proj"))?);
        // the experts: each stacked matrix packed as one (gate and up [E 2I, latent], down [E latent, I]); the router fp32 as it is
        let stack = |n: &str, rows: usize, cols: usize| -> Result<ExpertStack> {
            let w = words(&read(&format!("{m}.experts.{n}"))?);
            let Some(t) = sl else { return ExpertStack::lossless(upload_packed(pack_words(&w, e * rows, cols, Layout::Tiered, &format!("{m}.experts.{n}"))?)?, e) };
            // (a group of GPUs: this GPU's rows of each expert's, stacked)
            if rows % (64 * t.n) != 0 {
                return err(format!("{m}.experts.{n}: experts of {rows} rows across {} GPUs: each GPU's rows in 64s are wanted", t.n));
            }
            let (r0, rn) = t.rows(rows);
            let mine: Vec<u16> = (0..e).flat_map(|x| w[(x * rows + r0) * cols..(x * rows + r0 + rn) * cols].iter().copied()).collect();
            let mut st = ExpertStack::lossless(upload_packed(pack_words(&mine, e * rn, cols, Layout::Tiered, &format!("{m}.experts.{n}"))?)?, e)?;
            st.tp_rows = rows;
            Ok(st)
        };
        let (up, down) = (stack("gate_up_proj", 2 * inter, lat)?, stack("down_proj", lat, inter)?);
        let router = upload_bytes(&read(&format!("{m}.gate.weight"))?)?;
        let bias = upload_bytes(&read(&format!("{m}.gate.e_score_correction_bias"))?)?;
        // (the clamp of an infinite limit: the activation is the kernel's, so the products are not fused)
        let cfg = MoeCfg { experts: e, router_hidden: d, hidden: lat, inter, router: Router::sigmoid(c.top_k, c.route_scale), act: Act::SwiGlu { limit: Some(f32::INFINITY) } };
        let block = l_.moe.len();
        l_.moe.push(Box::new(MuseMoe(MoeLayer::new(cfg, MoeWeights { router, bias: Some(bias), escale: None, up, down: Down::Block(down) }, None)?)));
        let ring = rings.len();
        rings.push(Ring { d: hd, nk, window: if c.sliding[l] { c.window } else { 0 } });
        let mut ops = vec![
            Op::MuseRms { x: Slot::X, w: w_in, y: Slot::H, eps: c.eps },
            Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "attention: q, k, v and gate product" },
            Op::MuseQk { stride: row, nq, nk, rope: c.sliding[l], ring, scale: c.qk_scale },
            Op::MuseAttn { ring, nq, nk, window: if c.sliding[l] { c.window } else { 0 } },
            Op::MuseOGate { stride: row, goff: (nq + 2 * nk) * hd, nq },
            Op::Linear { w: m0 + 1, x: Slot::Core, y: Slot::O, name: "attention: o product" },
            Op::MuseRms { x: Slot::O, w: ones, y: Slot::O, eps: c.post_eps },
            Op::MuseGate { alpha: aa, beta: ba },
            Op::MuseRms { x: Slot::X, w: w_ff, y: Slot::H, eps: c.eps },
            Op::Linear { w: m0 + 2, x: Slot::H, y: Slot::Proj, name: "experts: latent in" },
            Op::MuseRms { x: Slot::Proj, w: w_pre, y: Slot::Xc, eps: c.eps },
            Op::MuseMoe { block, post_w: w_post_e, eps: c.post_eps },
            Op::Linear { w: m0 + 3, x: Slot::M, y: Slot::O, name: "experts: latent out" },
            Op::MuseRmsIn { x: Slot::O, w: w_post_ff, y: Slot::O, eps: c.post_eps },
            Op::MuseGate { alpha: af, beta: bfg },
        ];
        if st.last && l + 1 == c.layers {
            let norm = put(&mut l_.small, upload_bytes(&read(&format!("{}norm.weight", c.prefix))?)?);
            ops.push(Op::MuseRms { x: Slot::X, w: norm, y: Slot::H, eps: c.eps });
        }
        layers.push(ops);
    }
    let empty = || -> Result<Mat> { Ok(Mat::Dense(Buf::new(2)?, 0, 0)) };
    let embed = l_.mats.len();
    l_.mats.push(if st.first { pack_as(&words(&read(&format!("{}embed_tokens.weight", c.prefix))?), c.vocab, d, "embed", false)? } else { empty()? });
    let lm = l_.mats.len();
    l_.mats.push(if st.last { pack(&words(&read("lm_head.weight")?), c.vocab, d, "lm_head")? } else { empty()? });
    // the rotary's tables: cos and sin [positions, 32] fp32, the angle pos x f in fp32, its cosine and sine in f64
    let mp = o.ctx.div_ceil(16) * 16 + o.chunk.max(1) + 16;
    let f = c.freqs();
    let (mut cs, mut sn) = (Vec::with_capacity(mp * f.len()), Vec::with_capacity(mp * f.len()));
    for p in 0..mp {
        for &fr in &f {
            let (s_, c_) = ((p as f32 * fr) as f64).sin_cos();
            cs.push(c_ as f32);
            sn.push(s_ as f32);
        }
    }
    let ropes = vec![(upload_bytes(as_bytes(&cs))?, upload_bytes(as_bytes(&sn))?)];
    let mut widths = [0usize; super::SLOTS];
    for (s, w) in [(Slot::X, d), (Slot::H, d), (Slot::Mix, row), (Slot::Qo, nq * hd), (Slot::Att, nq * hd), (Slot::Core, nq * hd), (Slot::O, d), (Slot::Proj, lat), (Slot::Xc, lat), (Slot::M, lat)] {
        widths[s as usize] = w;
    }
    let desc = ModelDesc {
        family: "Muse Spark",
        hidden: d,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        ropes: Vec::new(),
        rings,
        // (the embedding's row normed without a weight, in place)
        first: Op::MuseRms { x: Slot::X, w: ones, y: Slot::X, eps: c.eps },
        layers,
        states: Vec::new(),
        final_norm: None,
        decoded: false,
        skinny: false,
        embed,
        lm,
        softcap: c.softcap,
        eos: c.eos.clone(),
        post: None,
        v4: None,
        hc: (0.0, 0),
        logit_mult: c.out_mult,
    };
    Dyn::from_parts(dev, desc, l_, ropes, layout, st, o)
}
