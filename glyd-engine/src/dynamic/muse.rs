//! Muse Spark's text model (src/muse.rs) as a description: each layer's attention (q, k, v and the gate in one product; heads of 64 normed, the rotary on the window's layers; the output normed and gated),
//! its experts in the latent (a projection in, a centered norm, the experts with each output normed, the weighted sum, a projection out), the input-scaled norms and the residual gates; the normed
//! embedding, the final norm, the soft-cap of the logits times the output multiplier. bf16 as the checkpoint has it, packed at the level asked (lossless by default); the routers fp32, as they are. The
//! arithmetic is the kinds' (the attention's kernels/attn.cu, the norms' and residual gates' kernels/residual.cu, the experts' moe.rs) and kernels/muse.cu's; src/muse_ref.rs is the CPU reference it is checked against (`musecheck`).

use super::attn::{AttnOp, GateAt, QkNorm, RopeAt};
use super::{Dyn, Loaded, ModelDesc, MoeOp, NormW, Op, Opts, ResidentMoe, Ring, Slot, Stage};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::kern35::Mat;
use crate::moe::{Act, ExpertStack, MoeCfg, MoeLayer, MoeWeights, Router};
use crate::moe_rows::Down;
use crate::muse::{plus_one, residual_gate, ConfigMuse};
use crate::weights::{as_bytes, pack_words, upload_bytes, upload_packed, Layout};
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// kernels/muse.cu's kernel the executor launches for the head (the experts' are moe.rs's, the attention's the attention kind's, the norms' and residual gates' the residual kinds').
pub struct KernMuse {
    pub(crate) softcap: crate::kt::muse_softcap,
}

impl KernMuse {
    pub fn load(dev: &Dev) -> Result<KernMuse> {
        Ok(KernMuse { softcap: dev.typed()? })
    }
}

/// Layer `l`'s attention (heads of 64, ring `ring`): q, k, v and the gate in Mix's rows; q and k normed without a weight, q times the scale (the qk scale over
/// 64^1/2) before the rotary on the window's layers, each with transformers' bf16 rounding points (`AttnRounding::Sdpa`); the scores fp32 and not scaled again;
/// the output normed without a weight and times sigmoid(gate), in Att.
fn attention(c: &ConfigMuse, l: usize, ring: usize) -> Op {
    let (nq, nk, hd) = (c.heads, c.kv_heads, c.head_dim);
    Op::Attn(AttnOp {
        src: Slot::Mix,
        stride: (2 * nq + 2 * nk) * hd,
        q_step: hd,
        k_off: nq * hd,
        v_off: Some((nq + nk) * hd),
        d: hd,
        nq,
        nk,
        writes: true,
        ring,
        window: if c.sliding[l] { c.window } else { 0 },
        norm: QkNorm::Plain,
        v_norm: false,
        rope: c.sliding[l].then_some(RopeAt { table: 0, span: hd, pairs: hd / 2 }),
        q_scale: c.qk_scale / (hd as f32).sqrt(),
        scale: 1.0,
        softcap: 0.0,
        gate: Some(GateAt { off: (nq + 2 * nk) * hd, step: hd, norm: true }),
        round: true,
        sbf16: false,
        picked: None,
    })
}

/// A layer's experts in the latent (block `block`): the router reads H by its fp32 weights, its logits times `mult` (the output multiplier), the experts the normed latent
/// Xc, each expert's output normed (input-scaled by the small `post_w`, eps) before it is weighed, their sum into M (`MoeLayer::forward_normed`).
pub(crate) fn latent_moe(block: usize, post_w: usize, eps: f32, mult: f32) -> MoeOp {
    MoeOp { x: Slot::Xc, y: Slot::M, logit_scale: mult, expert_norm: Some((post_w, eps)), ..MoeOp::of(block) }
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
    load_stage_hb(dir, o, st, None)
}

/// `load_stage` with its experts in the expert cache (`hc`: the stage's places and memory): a hummingbird directory's (src/hbg_store.rs), or the release's lossless experts (src/tiering.rs `cached_tiered`).
pub fn load_stage_hb(dir: &Path, o: &Opts, st: Stage, hc: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let c = ConfigMuse::load(dir)?;
    if c.head_dim != 64 || c.hidden > 8192 || c.latent > 4096 || c.top_k > 32 {
        return err(format!("heads of {}, a width of {}, a latent of {}, {} experts a token: this engine runs heads of 64, widths up to 8192, latents up to 4096 and up to 32 experts a token", c.head_dim, c.hidden, c.latent, c.top_k));
    }
    let dev = Dev::new(o.device)?;
    let ck = io(Checkpoint::open(dir))?;
    // (a hummingbird directory, src/hb_pack.rs: the routed experts' codebook rows in place of the release's)
    let hb = crate::hb_pack::HbDir::open(dir)?;
    let store = hc.map(crate::hbg_store::store_for);
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
    // (the stage's last layers' matrices in the computer's memory where the cache says so, src/dense_ram.rs: each copied into one slot before it runs)
    let mut ram = crate::dense_ram::RamBuilder::new(crate::dense_ram::Placer::last(st.layers.0, st.layers.1, hc.map_or(0, |x| x.dense)));
    if sl.is_some() && hc.is_some_and(|x| x.dense > 0) {
        return err("Muse Spark's dense layers in the computer's memory: a stage on one GPU, not sliced across GPUs");
    }
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
        let mut mat = |mats: &mut Vec<Mat>, w: &[u16], rows: usize, cols: usize, name: &str| -> Result<()> {
            if ram.in_ram(l) {
                ram.push(l, mats, pack_words(w, rows, cols, layout, name)?).map(|_| ())
            } else {
                mats.push(pack(w, rows, cols, name)?);
                Ok(())
            }
        };
        mat(&mut l_.mats, &qkvg, row, d, &format!("{a}.qkvg"))?;
        drop(qkvg);
        mat(&mut l_.mats, &words(&read(&format!("{a}.o_proj.weight"))?), d, nq * hd, &format!("{a}.o_proj"))?;
        mat(&mut l_.mats, &words(&read(&format!("{m}.pre_expert_proj.weight"))?), lat, d, &format!("{m}.pre_expert_proj"))?;
        mat(&mut l_.mats, &words(&read(&format!("{m}.post_expert_proj.weight"))?), d, lat, &format!("{m}.post_expert_proj"))?;
        // the experts: each stacked matrix packed as one (gate and up [E 2I, latent], down [E latent, I]); the router fp32 as it is
        let stack = |n: &str, rows: usize, cols: usize| -> Result<ExpertStack> {
            let w = words(&read(&format!("{m}.experts.{n}"))?);
            let Some(t) = sl else {
                return ExpertStack::lossless(upload_packed(pack_words(&w, e * rows, cols, Layout::Tiered, &format!("{m}.experts.{n}"))?)?, e);
            };
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
        let router = upload_bytes(&read(&format!("{m}.gate.weight"))?)?;
        let bias = upload_bytes(&read(&format!("{m}.gate.e_score_correction_bias"))?)?;
        // (the clamp of an infinite limit: the activation is the kernel's, so the products are not fused)
        let cfg = MoeCfg {
            experts: e,
            router_hidden: d,
            hidden: lat,
            inter,
            router: Router::sigmoid(c.top_k, c.route_scale),
            act: Act::SwiGlu { limit: Some(f32::INFINITY) },
        };
        let block = l_.moe.len();
        // (the host's pairs of the cache normed as the sum norms each expert's output: the post-expert norm's input scale as the GPU holds it, bf16)
        let post: Vec<f32> = if hc.is_some() && hb.is_some() {
            plus_one(&vals(&format!("{m}.experts.post_expert_norm.weight"))?).iter().map(|&v| crate::hb_codec::f32_of(crate::hb_codec::bf16(v))).collect()
        } else {
            Vec::new()
        };
        let finish = crate::cpu_hbg::Finish::Normed { signs: Vec::new(), rot: 0, w: post, eps: c.post_eps };
        let layer = match crate::hbg_store::hb_layer(hb.as_ref(), l, sl.is_some(), hc.zip(store.as_ref()), cfg.act, finish, dev.ordinal, cfg.router.top_k)? {
            Some(x) => x.into_layer(cfg, router, Some(bias), None, None)?,
            // (R12: the release's experts in the expert cache, every one in the computer's memory: each stack's one lossless pack in the tiered layout, whatever the level of the rest, cut per expert)
            None if hb.is_none() && hc.is_some() => {
                let (Some(cg), Some(store)) = (hc, store.as_ref()) else { unreachable!("a stage's cache has its store") };
                if sl.is_some() || cg.ram.is_some() {
                    return err("Muse Spark's experts in the expert cache: a layer on one GPU, every expert in the computer's memory (the disk holds hummingbird directories' experts alone)");
                }
                let places = |n: &str, rows: usize, cols: usize| -> Result<crate::tiering::TieredPlaces> {
                    let w = words(&read(&format!("{m}.experts.{n}"))?);
                    let p = glyd_gpu::pack::pack_tiered_checked(&w, e * rows, cols)
                        .ok_or_else(|| Error(format!("{m}.experts.{n}: decoded to other bits than its weights")))?;
                    crate::tiering::TieredPlaces::of(&p, e)
                };
                let (pu, pd) = (places("gate_up_proj", 2 * inter, lat)?, places("down_proj", lat, inter)?);
                let (up, down, cache) = crate::tiering::cached_tiered(&pu, &pd, cg.places, store, l, dev.ordinal)?;
                MoeLayer::new(cfg, MoeWeights { router, bias: Some(bias), escale: None, up, down: Down::Block(down) }, None)?.with_cache(cache)?
            }
            None => {
                let (up, down) = (stack("gate_up_proj", 2 * inter, lat)?, stack("down_proj", lat, inter)?);
                MoeLayer::new(cfg, MoeWeights { router, bias: Some(bias), escale: None, up, down: Down::Block(down) }, None)?
            }
        };
        l_.moe.push(Box::new(ResidentMoe(layer)));
        let ring = rings.len();
        rings.push(Ring { d: hd, nk, window: if c.sliding[l] { c.window } else { 0 }, values: true });
        let mut ops = vec![
            Op::RmsNorm { x: Slot::X, w: w_in, y: Slot::H, weight: NormW::Plain, eps: c.eps },
            Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "attention: q, k, v and gate product" },
            attention(&c, l, ring),
            Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
            Op::RmsNorm { x: Slot::O, w: ones, y: Slot::O, weight: NormW::Plain, eps: c.post_eps },
            Op::GatedResidual { alpha: aa, beta: ba },
            Op::RmsNorm { x: Slot::X, w: w_ff, y: Slot::H, weight: NormW::Plain, eps: c.eps },
            Op::Linear { w: m0 + 2, x: Slot::H, y: Slot::Proj, name: "experts: latent in" },
            Op::RmsNorm { x: Slot::Proj, w: w_pre, y: Slot::Xc, weight: NormW::Plain, eps: c.eps },
            Op::Moe(latent_moe(block, w_post_e, c.post_eps, c.out_mult)),
            Op::Linear { w: m0 + 3, x: Slot::M, y: Slot::O, name: "experts: latent out" },
            Op::RmsNorm { x: Slot::O, w: w_post_ff, y: Slot::O, weight: NormW::Input, eps: c.post_eps },
            Op::GatedResidual { alpha: af, beta: bfg },
        ];
        if st.last && l + 1 == c.layers {
            let norm = put(&mut l_.small, upload_bytes(&read(&format!("{}norm.weight", c.prefix))?)?);
            ops.push(Op::RmsNorm { x: Slot::X, w: norm, y: Slot::H, weight: NormW::Plain, eps: c.eps });
        }
        layers.push(ops);
    }
    let empty = || -> Result<Mat> { Ok(Mat::Dense(Buf::new(2)?, 0, 0)) };
    let embed = l_.mats.len();
    l_.mats.push(if st.first { pack_as(&words(&read(&format!("{}embed_tokens.weight", c.prefix))?), c.vocab, d, "embed", false)? } else { empty()? });
    let lm = l_.mats.len();
    l_.mats.push(if st.last { pack(&words(&read("lm_head.weight")?), c.vocab, d, "lm_head")? } else { empty()? });
    // the rotary's tables: cos and sin [positions, 32] bf16, the angle pos x f in fp32, its cosine and sine in f64 rounded to fp32 then to bf16 (transformers' cos and
    // sin in the activations' type)
    let mp = o.ctx.div_ceil(16) * 16 + o.chunk.max(1) + 16;
    let f = c.freqs();
    let (mut cs, mut sn) = (Vec::with_capacity(mp * f.len()), Vec::with_capacity(mp * f.len()));
    for p in 0..mp {
        for &fr in &f {
            let (s_, c_) = ((p as f32 * fr) as f64).sin_cos();
            cs.push(crate::gemma4::bf16_bits(c_ as f32));
            sn.push(crate::gemma4::bf16_bits(s_ as f32));
        }
    }
    let ropes = vec![(upload_bytes(as_bytes(&cs))?, upload_bytes(as_bytes(&sn))?)];
    let mut widths = [0usize; super::SLOTS];
    for (s, w) in [
        (Slot::X, d),
        (Slot::H, d),
        (Slot::Mix, row),
        (Slot::Qo, nq * hd),
        (Slot::Att, nq * hd),
        (Slot::O, d),
        (Slot::Proj, lat),
        (Slot::Xc, lat),
        (Slot::M, lat),
    ] {
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
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        // (the embedding's row normed without a weight, in place)
        first: Op::RmsNorm { x: Slot::X, w: ones, y: Slot::X, weight: NormW::Plain, eps: c.eps },
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
    l_.ram = ram.finish(&mut l_.mats)?;
    // (the loader's own operations, as it indexed what it loaded, held to the layer kinds' description of the config: one model, or it is not loaded)
    let desc = super::lower::agree(lowered(dir, &c, &st)?, &desc, &dir.join("config.json").display().to_string())?;
    Dyn::from_parts(dev, desc, l_, ropes, layout, st, o)
}

/// Muse Spark's executor description for stage `st` from the config.json in `dir` (R3a): its layer kinds lowered (`lower::muse`), held to `describe`.
pub fn lowered(dir: &Path, c: &ConfigMuse, st: &Stage) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let lowered = super::lower::muse(&kinds, c.eos.clone(), c.max_pos, st)?;
    super::lower::agree(lowered, &describe(c, st)?, &label)
}

/// The executor's description of Muse Spark for stage `st`, as the loader indexes what it loads: the small tensors a weightless norm's ones first, then each layer's
/// nine (its input norm, the attention's residual gate's two, the feed-forward norm, the experts' pre and post norms, the post-feed-forward norm, its residual
/// gate's two), the final norm after the last layer's; each layer's four matrices (q, k, v and the gate; o; the latent in and out), then the embedding and the head;
/// a block of experts a layer.
pub fn describe(c: &ConfigMuse, st: &Stage) -> Result<ModelDesc> {
    if c.head_dim != 64 || c.hidden > 8192 || c.latent > 4096 || c.top_k > 32 {
        return err(format!("heads of {}, a width of {}, a latent of {}, {} experts a token: this engine runs heads of 64, widths up to 8192, latents up to 4096 and up to 32 experts a token", c.head_dim, c.hidden, c.latent, c.top_k));
    }
    let (d, nq, nk, hd, lat) = (c.hidden, c.heads, c.kv_heads, c.head_dim, c.latent);
    let row = (2 * nq + 2 * nk) * hd;
    let ones = 0;
    let (mut n_small, mut n_mats, mut n_moe, mut rings, mut layers) = (1, 0, 0, Vec::new(), Vec::new());
    for l in 0..c.layers {
        if !(st.layers.0..st.layers.1).contains(&l) {
            layers.push(Vec::new());
            continue;
        }
        let s0 = n_small;
        let (w_in, aa, ba, w_ff, w_pre, w_post_e, w_post_ff, af, bfg) = (s0, s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5, s0 + 6, s0 + 7, s0 + 8);
        n_small += 9;
        let m0 = n_mats;
        n_mats += 4;
        let block = n_moe;
        n_moe += 1;
        let ring = rings.len();
        let window = if c.sliding[l] { c.window } else { 0 };
        rings.push(Ring { d: hd, nk, window, values: true });
        let mut ops = vec![
            Op::RmsNorm { x: Slot::X, w: w_in, y: Slot::H, weight: NormW::Plain, eps: c.eps },
            Op::Linear { w: m0, x: Slot::H, y: Slot::Mix, name: "attention: q, k, v and gate product" },
            attention(c, l, ring),
            Op::Linear { w: m0 + 1, x: Slot::Att, y: Slot::O, name: "attention: o product" },
            Op::RmsNorm { x: Slot::O, w: ones, y: Slot::O, weight: NormW::Plain, eps: c.post_eps },
            Op::GatedResidual { alpha: aa, beta: ba },
            Op::RmsNorm { x: Slot::X, w: w_ff, y: Slot::H, weight: NormW::Plain, eps: c.eps },
            Op::Linear { w: m0 + 2, x: Slot::H, y: Slot::Proj, name: "experts: latent in" },
            Op::RmsNorm { x: Slot::Proj, w: w_pre, y: Slot::Xc, weight: NormW::Plain, eps: c.eps },
            Op::Moe(latent_moe(block, w_post_e, c.post_eps, c.out_mult)),
            Op::Linear { w: m0 + 3, x: Slot::M, y: Slot::O, name: "experts: latent out" },
            Op::RmsNorm { x: Slot::O, w: w_post_ff, y: Slot::O, weight: NormW::Input, eps: c.post_eps },
            Op::GatedResidual { alpha: af, beta: bfg },
        ];
        if st.last && l + 1 == c.layers {
            ops.push(Op::RmsNorm { x: Slot::X, w: n_small, y: Slot::H, weight: NormW::Plain, eps: c.eps });
            n_small += 1;
        }
        layers.push(ops);
    }
    let (embed, lm) = (n_mats, n_mats + 1);
    let mut widths = [0usize; super::SLOTS];
    for (s, w) in [
        (Slot::X, d),
        (Slot::H, d),
        (Slot::Mix, row),
        (Slot::Qo, nq * hd),
        (Slot::Att, nq * hd),
        (Slot::O, d),
        (Slot::Proj, lat),
        (Slot::Xc, lat),
        (Slot::M, lat),
    ] {
        widths[s as usize] = w;
    }
    Ok(ModelDesc {
        family: "Muse Spark",
        hidden: d,
        vocab: c.vocab,
        eps: c.eps,
        max_pos: c.max_pos,
        widths,
        mats: Vec::new(),
        small: Vec::new(),
        forms: Vec::new(),
        ropes: Vec::new(),
        rings,
        first: Op::RmsNorm { x: Slot::X, w: ones, y: Slot::X, weight: NormW::Plain, eps: c.eps },
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
    })
}
