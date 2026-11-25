//! GLM-5.3-Flash's kernel table and weights, which the executor's description of it loads by (src/dynamic/glm.rs) and Condor's engine (src/model_qwen4x.rs) shares: the
//! kernels of kernels/glm.cu and kernels/mla.cu with typed launches (`KernGlm`: Kimi delta attention, the residual streams, the latent attention's), the layers' weights
//! (`WeightsGlm`: a layer's norms, mixing weights and Kimi delta or dense MLP matrices, packed as they load, and the blocks a caller makes of its mixture-of-experts layers),
//! and a latent attention layer's (`MlaWeights`). The config and tensor names are src/glm.rs's. (GLM's own development engine, which these came out of, is gone: R13 #234.)

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result, Stream, Words};
use crate::glm::{ConfigGlm, GlmKind};
use crate::kern35::Mat;
use crate::launch;
use crate::moe::MoeLayer;
use crate::qwen35::Pack35;
use crate::runtime::kinds::Rounding;
use crate::weights::{as_bytes, choose_layout, ordered, read_pack, upload_bytes, upload_packed, v7_manifest, Layout, PackBuf};
use glyd_gpu::json::Value;
use glyd_gpu::pack::{unpack_int4, unpack_int5, unpack_int6, Tiered, INT4_TAG, INT5_TAG, INT6_TAG};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::path::Path;

/// A Kimi delta head's channels (and the state's side), the residual streams (kernels/residual.cu's), the tokens of a chunk of the chunked delta rule and its dynamic shared memory in bytes (kernels/glm.cu).
pub const GM_D: usize = 128;
pub const GM_HC: usize = crate::dynamic::residual::HC_N;
pub const GM_L: usize = 32;
pub const GM_CHUNK_SMEM: u32 = 99_072;
/// The floats of a row of the mixing projection's results: the collapse weights, the spreading weights, the mixing matrix (4 + 4 + 16).
pub const GM_NP: usize = crate::dynamic::residual::HC_NP;

/// GLM's kernels with typed launches.
pub struct KernGlm {
    /// The residual streams' kernels (transformers' bf16 rounding, `Rounding::Twice`).
    hc: crate::dynamic::residual::KernRes,
    conv: crate::kt::conv_silu,
    conv_w32: crate::kt::conv_silu_w32,
    rec: crate::kt::glm_kda_rec,
    chunk: crate::kt::glm_kda_chunk,
    out: crate::kt::glm_kda_out,
    out_w32: crate::kt::glm_kda_out_w32,
    small: crate::kt::small_linear,
    // the latent attention's (kernels/mla.cu, the layer kind's: this engine's caches are rooms of their own, read with no block table)
    rms2: crate::kt::rms2_rows,
    layernorm: crate::kt::layernorm_rows,
    scatter: crate::kt::scatter_rows,
    absorb: crate::kt::mla_absorb,
    kp_keys: crate::kt::idx_pool,
    idx_score: crate::kt::idx_score,
    select: crate::kt::idx_pick,
    mla_attn: crate::kt::mla_attn,
    /// The shared memory a block of `mla_attn` may take here (the GPU's most, opted in to at load).
    mla_smem: u32,
    mla_merge: crate::kt::mla_merge,
    expand_v: crate::kt::mla_expand_v,
    /// The streams' split projection's partial sums, for up to HC_FEW rows.
    hc_part: Buf,
}

/// Rows at most that the mixing weights' split projection takes (a decode step, the rows of a verified draft).
pub const HC_FEW: u32 = crate::dynamic::residual::HC_FEW;

/// The dynamic shared memory `mla_attn` (kernels/mla.cu) may take on `dev`: the GPU's most a block, less a KiB for the kernel's own (its block sums' 256 bytes).
pub fn mla_smem(dev: &Dev) -> Result<u32> {
    Ok(dev.smem_optin()?.saturating_sub(1 << 10))
}

/// The most positions a pass of the latent attention (scores over `ks` channels) holds where a block has `smem` bytes of shared memory: `mla_attn_with`'s
/// tile, a multiple of 16.
pub fn mla_positions(smem: u32, ks: i32) -> usize {
    ((smem as usize / 4).saturating_sub(ks as usize) / 2) / 16 * 16
}

impl KernGlm {
    pub fn load(dev: &Dev) -> Result<KernGlm> {
        let chunk = dev.typed::<crate::kt::glm_kda_chunk>()?;
        chunk.allow_smem(GM_CHUNK_SMEM)?;
        // (the dense attention's scores, two floats a position, are in shared memory: as much as the GPU gives a block)
        let (mla_attn, mla_smem) = (dev.typed::<crate::kt::mla_attn>()?, mla_smem(dev)?);
        mla_attn.allow_smem(mla_smem)?;
        Ok(KernGlm {
            hc: crate::dynamic::residual::KernRes::load(dev)?,
            conv: dev.typed()?,
            conv_w32: dev.typed()?,
            rec: dev.typed()?,
            chunk,
            out: dev.typed()?,
            out_w32: dev.typed()?,
            small: dev.typed()?,
            rms2: dev.typed()?,
            layernorm: dev.typed()?,
            scatter: dev.typed()?,
            absorb: dev.typed()?,
            kp_keys: dev.typed()?,
            idx_score: dev.typed()?,
            select: dev.typed()?,
            mla_attn,
            mla_smem,
            mla_merge: dev.typed()?,
            expand_v: dev.typed()?,
            hc_part: Buf::zeroed(crate::dynamic::residual::part_words(Rounding::Twice, 0, 0)? * 4)?,
        })
    }

    /// x [n, 4, d] <- the embedding rows e [n, d] four times.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/glm.cu and kernels/residual.cu).
    pub unsafe fn streams(&self, s: Stream, e: u64, x: u64, n: u32, d: i32) -> Result<()> {
        self.hc.streams(s, e, x, n, d, 4)
    }

    /// The mixing weights of the streams x [n, 4, d] (fn [24, 4d] bf16, base [24] and scale [3] fp32) into hcp [n, 24] fp32, and the collapse normed with the weight nw [d] into h [n, d].
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn hc_mix(&self, s: Stream, x: u64, hc: &Hc, nw: u64, h: u64, hcp: u64, n: u32, d: i32, eps: f32, hc_eps: f32, iters: i32) -> Result<()> {
        let w = (hc.proj.ptr, hc.base.ptr, hc.scale.ptr);
        self.hc.pre(s, x, w, nw, h, hcp, self.hc_part.ptr, n, d, (eps, hc_eps, iters), Rounding::Twice)
    }

    /// The streams x [n, 4, d] after a block whose output is o [n, d]: updated in place by the weights hcp [n, 24] holds.
    ///
    /// # Safety
    /// As `streams`.
    pub unsafe fn hc_post(&self, s: Stream, x: u64, o: u64, hcp: u64, n: u32, d: i32) -> Result<()> {
        self.hc.post(s, x, o, hcp, n, d, Rounding::Twice)
    }

    /// out [rows, d] = the final norm (weight w [d]) of the mean of each row of x [rows, 4, d].
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn hc_head(&self, s: Stream, x: u64, w: u64, out: u64, rows: u32, d: i32, eps: f32) -> Result<()> {
        self.hc.head(s, x, None, 0, w, out, rows, d, (eps, 0.0), Rounding::Twice)
    }

    /// The causal convolution and SiLU of n rows (the first c channels of rows xs apart), the bf16 state [3, c] read and written (qwen35.cu's `conv_silu`; `w32`: the weights fp32, `conv_silu_w32`).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, w32: bool, st: u64, y: u64, n: i32, c: i32) -> Result<()> {
        launch!(if w32 { self.conv_w32.sig() } else { self.conv.sig() }, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, st, y, n, c)
    }

    /// The delta rule of n rows for `heads` heads: token by token (a decode step: n = 1) or, `chunked`, in chunks of GM_L tokens. xc [n, 3 heads 128] the convolution's output, fg [n, heads 128] and bb [n, heads] the gates'
    /// products, dtb and alog fp32, state fp32 [heads, 128, 128] read and written back, core [n, heads 128] the outputs.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta(
        &self,
        s: Stream,
        chunked: bool,
        xc: u64,
        fg: u64,
        dtb: u64,
        alog: u64,
        bb: u64,
        state: u64,
        core: u64,
        n: i32,
        heads: i32,
        bound: f32,
    ) -> Result<()> {
        if chunked {
            launch!(self.chunk, [heads as u32, 2, 1], [256, 1, 1], smem GM_CHUNK_SMEM, s; xc, fg, dtb, alog, bb, state, core, n, heads, bound)
        } else {
            launch!(self.rec, [heads as u32, 1, 1], [128, 1, 1], s; xc, fg, dtb, alog, bb, state, core, n, heads, bound)
        }
    }

    /// att [n, heads 128] = the gated norm of core (weight w [128], bf16 or, `w32`, fp32; gate og [n, heads 128]).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_out(&self, s: Stream, core: u64, og: u64, w: u64, w32: bool, att: u64, n: u32, heads: i32, eps: f32) -> Result<()> {
        launch!(if w32 { self.out_w32.sig() } else { self.out.sig() }, [n, heads as u32, 1], [128, 1, 1], s; core, og, w, att, heads, eps)
    }

    /// n rows of x (rows `xs` apart, `k` wide) normed as transformers' RMSNorm does (weight w [k]) into y (rows `ys` apart); with `dst` (the rows' positions, int32 on the device) row i goes to y's row dst[i], else to row i.
    ///
    /// # Safety
    /// As `streams`; k at most 4,096.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rms2(&self, s: Stream, x: u64, xs: i32, w: u64, y: u64, ys: i32, dst: u64, k: i32, eps: f32, n: u32) -> Result<()> {
        if k <= 0 || k > 4096 {
            return err(format!("a norm of {k} values: not supported"));
        }
        launch!(self.rms2, [n, 1, 1], [256, 1, 1], s; x, xs, w, y, ys, dst, 0u64, k, eps)
    }

    /// The same with a LayerNorm (weight w, bias b [k], k at most 256).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn layernorm(&self, s: Stream, x: u64, xs: i32, w: u64, b: u64, y: u64, ys: i32, dst: u64, k: i32, eps: f32, n: u32) -> Result<()> {
        if k <= 0 || k > 256 {
            return err(format!("a LayerNorm of {k} values: not supported"));
        }
        launch!(self.layernorm, [n, 1, 1], [128, 1, 1], s; x, xs, w, b, y, ys, dst, 0u64, k, eps)
    }

    /// Row pos[i] of dst (rows `ds` apart) = row i of src (rows `ss` apart), `k` values each.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scatter(&self, s: Stream, src: u64, ss: i32, dst: u64, ds: i32, pos: u64, k: i32, n: u32) -> Result<()> {
        launch!(self.scatter, [n, 1, 1], [256, 1, 1], s; src, ss, dst, ds, pos, 0u64, k)
    }

    /// qa [n, heads, kv_rank] fp32 = each head's Wk^T q for q [n, heads nope] (rows `qs` apart), from kv_b_proj `wkv` [heads (nope + vd), kv_rank].
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn absorb(&self, s: Stream, q: u64, qs: i32, wkv: u64, qa: u64, n: i32, heads: i32, nope: i32, vd: i32, kr: i32) -> Result<()> {
        self.absorb_with(s, q, qs, nope, wkv, qa, n, heads, nope, vd, kr, kr)
    }

    /// As `absorb`, a head's query `qhs` wide in q and `ks` wide in qa (its channels nope .. nope + ks - kr copied after the absorbed kr: Kimi-K3's key part the heads share).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn absorb_with(
        &self,
        s: Stream,
        q: u64,
        qs: i32,
        qhs: i32,
        wkv: u64,
        qa: u64,
        n: i32,
        heads: i32,
        nope: i32,
        vd: i32,
        kr: i32,
        ks: i32,
    ) -> Result<()> {
        if nope <= 0 || nope > 256 || kr <= 0 || kr > 512 || ks < kr || qhs < nope + ks - kr {
            return err(format!("latent attention with heads of {nope} ({qhs} wide), latents of {kr} and scores over {ks}: this engine runs heads up to 256 and latents up to 512"));
        }
        launch!(self.absorb, [heads as u32, (n as u32).div_ceil(8), 1], [256, 1, 1], s; q, qs, qhs, wkv, qa, n, heads, nope, vd, kr, ks)
    }

    /// The groups of keys the n rows (consecutive positions from pos[0]) complete: their keys into pk [., d].
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kp_keys(&self, s: Stream, ik: u64, gs: u64, ape: u64, pk: u64, pos: u64, n: i32, d: i32, kp: i32) -> Result<()> {
        if kp <= 0 || kp > 8 {
            return err(format!("key groups of {kp} tokens: this engine runs groups up to 8"));
        }
        launch!(self.kp_keys, [(n as u32).div_ceil(kp as u32) + 1, 1, 1], [128, 1, 1], s; ik, d, gs, d, ape, pk, d, 1i32, 0i32, 0u64, pos, n, d, kp)
    }

    /// scores [n, pmax] fp32 for the rows whose pools are more than `ksel`: the indexer's heads (`ih` of `d` channels, queries iq with rows `iqs` apart, head weights wts with rows `wsr` apart) against the pool keys pk.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn idx_score(
        &self,
        s: Stream,
        iq: u64,
        iqs: i32,
        wts: u64,
        wsr: i32,
        pk: u64,
        pos: u64,
        scores: u64,
        pmax: i32,
        ih: i32,
        d: i32,
        kp: i32,
        ksel: i32,
        scale: f32,
        wscale: f32,
        n: u32,
    ) -> Result<()> {
        let smem = ((ih * (d + 1) + ih + 8 * d) * 4) as u32;
        launch!(self.idx_score, [(pmax as u32).div_ceil(8), n, 1], [256, 1, 1], smem smem, s; iq, iqs, wts, wsr, pk, d, 1i32, 0i32, 0u64, pos, scores, pmax, ih, d, kp, ksel, scale, wscale)
    }

    /// The tokens each of n rows attends to: sel [n, tmax] int32 (the first cnt[i] of a row), from the scores.
    ///
    /// # Safety
    /// As `streams`; ksel at most 1,024.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn select(
        &self,
        s: Stream,
        scores: u64,
        pmax: i32,
        pos: u64,
        sel: u64,
        cnt: u64,
        tmax: i32,
        kp: i32,
        ksel: i32,
        tail: bool,
        n: u32,
    ) -> Result<()> {
        if ksel <= 0 || ksel > 1024 {
            return err(format!("an indexer that picks {ksel} groups of keys: this engine runs up to 1,024"));
        }
        launch!(self.select, [n, 1, 1], [1024, 1, 1], s; scores, pmax, pos, kp, ksel, kp, tail as i32, sel, tmax, cnt)
    }

    /// olat [n, heads, kv_rank] fp32: each row's and head's softmax attention over the tokens sel/cnt picked, of the latents lat [., kv_rank]. A few rows (at most `HC_FEW`) cut their tokens into 8 runs over blocks and merge
    /// them (`part`: [n, heads, 8, kv_rank + 2] fp32); more rows are parallel enough by themselves.
    ///
    /// # Safety
    /// As `streams`; kv_rank a multiple of 64 up to 512.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mla_attn(
        &self,
        s: Stream,
        qa: u64,
        lat: u64,
        sel: u64,
        cnt: u64,
        olat: u64,
        part: u64,
        tmax: i32,
        heads: i32,
        kr: i32,
        scale: f32,
        n: u32,
    ) -> Result<()> {
        self.mla_attn_with(s, qa, lat, sel, cnt, olat, part, tmax, heads, kr, kr, scale, n)
    }

    /// As `mla_attn`, the scores over `ks` channels (qa's heads and lat's rows ks wide), the values the first kr; `sel` 0: every position up to the row's (cnt = pos + 1).
    ///
    /// # Safety
    /// As `streams`; kv_rank and ks multiples of 64, kv_rank up to 512.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mla_attn_with(
        &self,
        s: Stream,
        qa: u64,
        lat: u64,
        sel: u64,
        cnt: u64,
        olat: u64,
        part: u64,
        tmax: i32,
        heads: i32,
        kr: i32,
        ks: i32,
        scale: f32,
        n: u32,
    ) -> Result<()> {
        if ks < kr || ks % 64 != 0 {
            return err(format!("latent attention of scores over {ks} channels and latents of {kr}: this engine does not run that shape"));
        }
        let splits = if n <= HC_FEW { 8 } else { 1 };
        // (a block's tokens a tile at a time: as many as its shared memory holds, two words a token besides the query's KS)
        let mut tile = ((tmax + splits - 1) / splits).min(mla_positions(self.mla_smem, ks) as i32).max(1);
        if tile < 64 && tile < (tmax + splits - 1) / splits {
            return err(format!(
                "latent attention with scores over {ks} channels: {} bytes of shared memory a block hold too few tokens a pass",
                self.mla_smem
            ));
        }
        // (a development build's GLYD_MLA_TILE: smaller tiles, the passes of a long context on a short one)
        if let Some(t) = crate::tune!("GLYD_MLA_TILE").and_then(|v| v.parse::<i32>().ok()).filter(|&t| t > 0) {
            tile = tile.min(t);
        }
        let smem = ((ks + 2 * tile) * 4) as u32;
        launch!(self.mla_attn, [heads as u32, n, splits as u32], [256, 1, 1], smem smem, s; qa, lat, ks, 0u64, sel, cnt, olat, part, tmax, heads, kr, ks, scale, splits, tile)?;
        if splits > 1 {
            launch!(self.mla_merge, [heads as u32, n, 1], [256, 1, 1], s; part, olat, heads, kr, splits)?;
        }
        Ok(())
    }

    /// ov [n, heads vd] bf16 = olat times each head's value rows of kv_b_proj `wkv`.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn expand_v(&self, s: Stream, olat: u64, wkv: u64, ov: u64, n: i32, heads: i32, nope: i32, vd: i32, kr: i32) -> Result<()> {
        let smem = (4 * kr * 4) as u32;
        launch!(self.expand_v, [heads as u32, (n as u32).div_ceil(4), 1], [256, 1, 1], smem smem, s; olat, wkv, ov, n, heads, nope, vd, kr)
    }

    /// y [m, rows] = x [m, cols] W^T for a bf16 W the library does not pack.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn small_linear(&self, s: Stream, x: u64, xs: i32, w: u64, y: u64, ys: i32, m: u32, rows: i32, cols: i32) -> Result<()> {
        launch!(self.small, [m, (rows.max(1) as u32).div_ceil(8), 1], [256, 1, 1], s; x, xs, w, y, ys, rows, cols)
    }
}

/// A layer's mixing weights: the projection of the flattened streams [24, 4 hidden] (bf16), the base [24] and the three scales (fp32).
pub struct Hc {
    pub proj: Buf,
    pub base: Buf,
    pub scale: Buf,
}

/// A layer's buffers, by kind: its matrices (Kimi delta attention: qkv, o, f_a, f_b, g_a, g_b, b; then a dense MLP's gate and up, down; a latent attention layer's are `MlaWeights`, a mixture of experts' its
/// layer) and small tensors; a mixture-of-experts layer's experts' layer, where the loader was given a maker of it.
pub struct LayerGlm {
    pub kind: GlmKind,
    /// Its index among the layers of its kind (a Kimi delta layer's state, a latent attention layer's cache).
    pub idx: usize,
    /// The norms before the two blocks.
    pub norm_a: Buf,
    pub norm_f: Buf,
    /// The mixing weights of the token mixer's block and of the feed-forward's.
    pub hc: [Hc; 2],
    pub mats: Vec<Mat>,
    /// Kimi delta attention: the three convolutions' weights as one [3 heads 128, 4], A_log, dt_bias (fp32), the gated norm's weight.
    pub small: Vec<Buf>,
    pub moe: Option<Box<MoeLayer>>,
}

/// The model's weights in GPU memory. A layer not asked for (`only`) is None; so are the embedding, the LM head and the final norm then.
pub struct WeightsGlm {
    pub cfg: ConfigGlm,
    pub layout: Layout,
    pub layers: Vec<Option<LayerGlm>>,
    pub embed: Option<PackBuf>,
    pub head: Option<PackBuf>,
    pub norm_f: Option<Buf>,
    /// Device bytes of the weights.
    pub bytes: usize,
}

/// A tensor's bytes as fp32 (a checkpoint's fp32 parameters, which a glyd-v7 directory may hold as bf16: widened exactly).
pub(crate) fn f32_bytes(ck: &Checkpoint, name: &str) -> Result<Vec<u8>> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let bytes = ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?;
    match t.dtype.as_str() {
        "F32" => Ok(bytes),
        "BF16" => Ok(bytes.as_chunks::<2>().0.iter().flat_map(|b| ((u16::from_le_bytes(*b) as u32) << 16).to_le_bytes()).collect()),
        other => err(format!("{name} is {other}, where F32 or BF16 is wanted")),
    }
}

/// The matrices `WeightsGlm::load` holds for layer `l`, in the order of `LayerGlm::mats`: a Kimi delta layer's seven (q, k and v as one, o, the forget gate's two, the output gate's two, the input gate's) and, where the
/// feed-forward is a dense MLP, its gate and up as one and its down. A latent attention layer's are its block's; a mixture-of-experts layer's (the shared expert's two and the experts) are its block's.
pub(crate) fn layer_matrices(cfg: &ConfigGlm, l: usize) -> Vec<Pack35> {
    let packs = cfg.layer_packs(l);
    let (attn, mlp) = packs.split_at(packs.len() - 2);
    attn.iter().filter(|_| cfg.kinds[l] == GlmKind::Kda).chain(mlp.iter().filter(|_| !cfg.moe[l])).cloned().collect()
}

/// Makes layer `l`'s experts' layer (src/moe_glm.rs `layers_cached`): the layer's number, the config, the checkpoint, the layout of the packs.
pub type Make<'a> = &'a dyn Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<MoeLayer>>;

impl WeightsGlm {
    /// A Hub-format directory (config.json, the safetensors: FP8 with block scales or bf16) or a glyd-v7 directory (`glyd-engine pack MODEL OUT --layout i6|i5`): the layers of `only` (all, with the embedding, the head and the
    /// final norm, where None) checked against the weight map, their matrices packed here by `threads` workers in `layout` (None: the GPU's) and copied to the GPU as they are done. A glyd-v7 directory is at one level
    /// (`layout` None, or the same): the packs it holds saved are read as they are, the rest (a matrix the packer does not take) as the Hub's. A latent attention layer holds its norms and mixing weights (its own weights are
    /// `MlaWeights`); a mixture-of-experts layer the same and `moe`'s experts' layer, which gets the checkpoint and the layout and loads what it needs of the layer's tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dir: &Path,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        only: Option<&[usize]>,
        moe: Option<Make<'_>>,
    ) -> Result<WeightsGlm> {
        WeightsGlm::load_ends(dir, threads, layout, cc, total_mem, only, None, moe)
    }

    /// `load`, with the ends asked for apart from the layers: `ends` (the embedding; the LM head and the final norm) for a stage of a model split across GPUs, None for `load`'s (with the whole model, or with no
    /// layer: the MTP head's).
    #[allow(clippy::too_many_arguments)]
    pub fn load_ends(
        dir: &Path,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        only: Option<&[usize]>,
        ends: Option<(bool, bool)>,
        moe: Option<Make<'_>>,
    ) -> Result<WeightsGlm> {
        let cfg = ConfigGlm::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let have: Vec<(String, String, Vec<u64>)> = ck
            .names()
            .map(|n| {
                let (_, t) = ck.get(n).expect("a tensor the checkpoint names");
                (n.to_string(), t.dtype.clone(), t.shape.clone())
            })
            .collect();
        let v7 = v7_manifest(dir)?;
        if let (Some((got, _)), Some(asked)) = (&v7, layout) {
            if *got != asked {
                return err(format!("{}: packed in the {got:?} layout, where {asked:?} is asked", dir.display()));
            }
        }
        let mut wanted: Vec<usize> = only.map_or_else(|| (0..cfg.layers).collect(), |l| l.to_vec());
        wanted.sort_unstable();
        wanted.dedup();
        let held = if v7.is_some() { cfg.saved(&ck) } else { Default::default() };
        match only {
            None => {
                cfg.check(&have, &held)?;
            }
            Some(l) => cfg.check_layers(&have, l, &held)?,
        }
        // the matrices: packed where the packer takes them, else as the checkpoint has them (as the bf16 an FP8 matrix dequantises to). The blocks load their own (a latent attention layer's, the experts and the shared
        // expert of a mixture-of-experts layer): here a layer's Kimi delta matrices and a dense MLP's.
        let mut items: Vec<Pack35> = Vec::new();
        let mut dense: Vec<(String, Vec<String>, usize, usize)> = Vec::new();
        for &l in &wanted {
            for p in layer_matrices(&cfg, l) {
                if p.packable() {
                    items.push(p);
                } else {
                    dense.push((p.key.clone(), p.parts.iter().map(|x| x.0.clone()).collect(), p.rows(), p.cols));
                }
            }
        }
        // (the embedding and the LM head: of a whole model, and of an engine with no layer of its own, which is the MTP head's alone; or a stage's ends)
        let (first, last) = ends.unwrap_or_else(|| {
            let all = only.is_none() || wanted.is_empty();
            (all, all)
        });
        items.extend(cfg.packs().into_iter().filter(|p| (p.key == "embed" && (first || (last && cfg.tie))) || (p.key == "lm_head" && last)));
        let matrix_bytes: u64 = items.iter().map(|p| (p.rows() * p.cols * 2) as u64).sum();
        let layout = match &v7 {
            Some((l, _)) => *l,
            None => choose_layout(layout, cc, total_mem, matrix_bytes, 0).0,
        };
        let mut packed: HashMap<String, PackBuf> = HashMap::new();
        // (a GPU of a group: its slice of the rows of each big matrix but the embedding, src/dynamic/tp.rs)
        let sl = crate::dynamic::tp::slicer();
        let cut: Vec<Option<(usize, usize)>> =
            items.iter().map(|p| sl.filter(|s| p.key != "embed" && s.slices(p.rows(), p.cols)).map(|s| s.rows(p.rows()))).collect();
        ordered(items.len(), threads, 2 * threads.max(1), &|k| crate::weights::read_pack_rows(&ck, v7.as_ref(), &items[k], layout, cut[k]), |k, p| {
            crate::status::STATUS.beat(); // (a sign of life for the program that started the engine)
            let mut b = upload_packed(p)?;
            if cut[k].is_some() {
                b.tp_rows = items[k].rows();
            }
            packed.insert(items[k].key.clone(), b);
            Ok(())
        })?;
        let read = |name: &str| -> Result<Buf> { upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?) };
        let read_f32 = |name: &str| -> Result<Buf> { upload_bytes(&f32_bytes(&ck, name)?) };
        let schedule = cfg.schedule();
        let mut layers: Vec<Option<LayerGlm>> = (0..cfg.layers).map(|_| None).collect();
        let mut bytes = 0usize;
        for &l in &wanted {
            crate::status::STATUS.beat(); // (a mixture-of-experts layer's block takes seconds to read)
            let (kind, idx) = schedule[l];
            let mut mats = Vec::new();
            for p in layer_matrices(&cfg, l) {
                mats.push(match packed.remove(&p.key) {
                    Some(b) => Mat::Packed(b),
                    None => {
                        let (_, members, rows, cols) = dense.iter().find(|d| d.0 == p.key).expect("a matrix kept as it is");
                        let mut w = vec![0u16; rows * cols];
                        let mut at = 0;
                        for (m, r) in members.iter().zip(&p.parts) {
                            ck.read_bf16(m, &mut w[at..at + r.1 * cols]).map_err(|e| Error(format!("{m}: {e}")))?;
                            at += r.1 * cols;
                        }
                        Mat::Dense(upload_bytes(crate::weights::as_bytes(&w))?, *rows, *cols)
                    }
                });
            }
            // (layer_kept's order: the two norms, the mixing weights of the two blocks, then Kimi delta attention's three convolutions, A_log, dt_bias and gated norm)
            let kept = cfg.layer_kept(l);
            let (norm_a, norm_f) = (read(&kept[0].name)?, read(&kept[1].name)?);
            let hc = |i: usize| -> Result<Hc> {
                Ok(Hc { proj: read(&kept[2 + 3 * i].name)?, base: read_f32(&kept[3 + 3 * i].name)?, scale: read_f32(&kept[4 + 3 * i].name)? })
            };
            let hc = [hc(0)?, hc(1)?];
            let small = if kind == GlmKind::Kda {
                let mut conv = Vec::new();
                for t in &kept[8..11] {
                    conv.extend(ck.read(&t.name).map_err(|e| Error(format!("{}: {e}", t.name)))?);
                }
                vec![upload_bytes(&conv)?, read_f32(&kept[11].name)?, read_f32(&kept[12].name)?, read(&kept[13].name)?]
            } else {
                Vec::new()
            };
            let moe_block = match &moe {
                Some(make) if cfg.moe[l] => Some(make(l, &cfg, &ck, layout)?),
                _ => None,
            };
            bytes += norm_a.bytes
                + norm_f.bytes
                + hc.iter().map(|h| h.proj.bytes + h.base.bytes + h.scale.bytes).sum::<usize>()
                + mats.iter().map(Mat::bytes).sum::<usize>()
                + small.iter().map(|b| b.bytes).sum::<usize>();
            bytes += moe_block.as_ref().map_or(0, |b| b.weights_bytes());
            layers[l] = Some(LayerGlm { kind, idx, norm_a, norm_f, hc, mats, small, moe: moe_block });
        }
        let embed = if first || (last && cfg.tie) { Some(packed.remove("embed").ok_or_else(|| Error("the embedding is not packed".into()))?) } else { None };
        let head = if last && !cfg.tie { Some(packed.remove("lm_head").ok_or_else(|| Error("the LM head is not packed".into()))?) } else { None };
        let norm_f = if last { Some(read(&format!("{}norm.weight", cfg.prefix))?) } else { None };
        bytes += embed.as_ref().map_or(0, |b| b.bytes) + head.as_ref().map_or(0, |h| h.bytes) + norm_f.as_ref().map_or(0, |b| b.bytes);
        if let Some(k) = packed.keys().next() {
            return err(format!("a pack the forward pass does not take: {k}"));
        }
        Ok(WeightsGlm { cfg, layout, layers, embed, head, norm_f, bytes })
    }

    pub fn lm_head(&self) -> Option<&PackBuf> {
        self.head.as_ref().or(self.embed.as_ref())
    }

    /// Whether every layer, the embedding, the head and the final norm are here.
    pub fn whole(&self) -> bool {
        self.layers.iter().all(Option::is_some) && self.embed.is_some() && self.norm_f.is_some()
    }
}

/// A latent attention layer's weights on the GPU: the library's products (q's two, the key-value latent's, o; packed as they load), kv_b_proj as the bf16 it is (the
/// absorbed form multiplies by its rows), the two norms' weights and the indexer's tensors as the checkpoint has them. The executor's latent attention (src/dynamic/glm.rs:
/// `Op::Mla`'s matrices and small tensors) loads them here.
pub(crate) struct MlaWeights {
    pub q_a: Mat,
    pub q_b: Mat,
    pub kv_a: Mat,
    pub o: Mat,
    pub q_a_ln: Buf,
    pub kv_a_ln: Buf,
    pub kv_b: Buf,
    pub wq_b: Buf,
    pub wk: Buf,
    pub k_norm_w: Buf,
    pub k_norm_b: Buf,
    pub wproj: Buf,
    pub ape: Buf,
    pub ix_gate: Buf,
}

/// A matrix of the layer as the engine multiplies by it: packed in `layout` (from the FP8 or bf16 checkpoint or a glyd-v7 directory's saved pack) where the library packs it, else as the bf16 it is.
fn load_mat(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, layout: Layout) -> Result<Mat> {
    if pk.packable() {
        // (a GPU of a group: its slice of the rows where the matrix is a big one, src/dynamic/tp.rs)
        let cut = crate::dynamic::tp::slicer().filter(|s| s.slices(pk.rows(), pk.cols)).map(|s| s.rows(pk.rows()));
        let mut b = upload_packed(crate::weights::read_pack_rows(ck, v7, pk, layout, cut)?)?;
        if cut.is_some() {
            b.tp_rows = pk.rows();
        }
        return Ok(Mat::Packed(b));
    }
    let (rows, cols) = (pk.rows(), pk.cols);
    let mut w = vec![0u16; rows * cols];
    let mut at = 0;
    for (name, r) in &pk.parts {
        ck.read_bf16(name, &mut w[at..at + r * cols]).map_err(|e| Error(format!("{name}: {e}")))?;
        at += r * cols;
    }
    Ok(Mat::Dense(upload_bytes(as_bytes(&w))?, rows, cols))
}

/// A tensor of the checkpoint, bf16 as it is, on the GPU.
fn tensor(ck: &Checkpoint, name: &str, elements: usize) -> Result<Buf> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    if t.dtype != "BF16" || t.shape.iter().product::<u64>() as usize != elements {
        return err(format!("{name} is {} {:?}, where {elements} BF16 values are wanted", t.dtype, t.shape));
    }
    upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?)
}

impl MlaWeights {
    /// Layer `l`'s of the model `c` from the checkpoint `ck`, its library matrices packed in `layout`. `kv_b_proj` is the checkpoint's bf16 (from a glyd-v7 directory, which holds it packed, the pack decoded: the
    /// layer's numbers are then the pack's).
    pub(crate) fn load(l: usize, c: &ConfigGlm, ck: &Checkpoint, layout: Layout) -> Result<MlaWeights> {
        if !c.has_layer(l) || c.kind_of(l) != crate::glm::GlmKind::Mla {
            return err(format!("layer {l} is not a latent attention layer"));
        }
        let (m, ix) = (&c.mla, &c.indexer);
        let v7 = v7_manifest(&ck.dir)?;
        if let Some((got, _)) = &v7 {
            if *got != layout {
                return err(format!("{}: packed in the {got:?} layout, where {layout:?} is asked", ck.dir.display()));
            }
        }
        let packs = c.layer_packs(l);
        let find =
            |suffix: &str| -> Result<&Pack35> { packs.iter().find(|p| p.key.ends_with(suffix)).ok_or_else(|| Error(format!("layer {l}: no pack {suffix}"))) };
        let a = format!("{}layers.{l}.self_attn", c.prefix);
        let (q_a, q_b, kv_a, o) = (
            load_mat(ck, v7.as_ref(), find(".q_a_proj")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".q_b_proj")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".kv_a_proj_with_mqa")?, layout)?,
            load_mat(ck, v7.as_ref(), find(".o_proj")?, layout)?,
        );
        // kv_b_proj: the checkpoint's bf16; where the directory holds it as a pack (a glyd-v7 one), that pack decoded
        let kv_b_name = format!("{a}.kv_b_proj.weight");
        let kv_b = if ck.get(&kv_b_name).is_some() {
            let mut w = vec![0u16; m.heads * (m.nope + m.v) * m.kv_rank];
            ck.read_bf16(&kv_b_name, &mut w).map_err(|e| Error(format!("{kv_b_name}: {e}")))?;
            upload_bytes(as_bytes(&w))?
        } else {
            let pk = find(".kv_b_proj")?;
            upload_bytes(as_bytes(&unpacked(ck, v7.as_ref(), pk, layout)?))?
        };
        let (h, qr) = (c.hidden, m.q_rank);
        let ixn = format!("{a}.indexer");
        Ok(MlaWeights {
            q_a,
            q_b,
            kv_a,
            o,
            q_a_ln: tensor(ck, &format!("{a}.q_a_layernorm.weight"), qr)?,
            kv_a_ln: tensor(ck, &format!("{a}.kv_a_layernorm.weight"), m.kv_rank)?,
            kv_b,
            wq_b: tensor(ck, &format!("{ixn}.wq_b.weight"), ix.heads * ix.dim * qr)?,
            wk: tensor(ck, &format!("{ixn}.wk.weight"), ix.dim * h)?,
            k_norm_w: tensor(ck, &format!("{ixn}.k_norm.weight"), ix.dim)?,
            k_norm_b: tensor(ck, &format!("{ixn}.k_norm.bias"), ix.dim)?,
            wproj: tensor(ck, &format!("{ixn}.weights_proj.weight"), ix.heads * h)?,
            ape: tensor(ck, &format!("{ixn}.index_kpool_compress_ape"), ix.kpool * ix.dim)?,
            ix_gate: tensor(ck, &format!("{ixn}.index_kpool_compress_gate"), ix.dim * h)?,
        })
    }
}

/// A pack of `ck` in the `level` layout (a glyd-v7 directory's saved one where `v7` is its manifest and it holds the pack, else packed from the Hub-format tensors) decoded to bf16 rows, as the library decodes it on the GPU.
pub(crate) fn unpacked(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout) -> Result<Vec<u16>> {
    let p = read_pack(ck, v7, pk, level)?;
    let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: not in a block layout", pk.key)) };
    let t = Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers };
    let mut w = vec![0u16; p.rows * p.cols];
    let ok = match tiers[0] >> 24 {
        INT6_TAG => unpack_int6(&t, &mut w),
        INT5_TAG => unpack_int5(&t, &mut w),
        INT4_TAG => unpack_int4(&t, &mut w),
        tag => return err(format!("{}: a layout of tag {tag}", pk.key)),
    };
    if ok {
        Ok(w)
    } else {
        err(format!("{}: its buffers do not decode", pk.key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = include_str!("../kernels/glm.cu");
    const MLA_SRC: &str = include_str!("../kernels/mla.cu");

    /// The constants this file names are the kernels' (`#define`s of kernels/glm.cu), and the shared memory of the chunked rule is what its arrays take.
    #[test]
    fn the_kernels_constants_are_the_sources() {
        let define = |n: &str| -> Option<usize> {
            SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!((define("GM_D"), define("GM_L")), (Some(GM_D), Some(GM_L)));
        let (bv, ks, ls) = (define("GM_BV").unwrap(), define("GM_KS").unwrap(), define("GM_LS").unwrap());
        assert_eq!((3 * GM_L * ks + GM_D * bv + GM_L * bv + 2 * GM_L * ls + GM_L) * 4, GM_CHUNK_SMEM as usize);
        // (it fits the 99 KB a block of an Ampere, Ada or Blackwell consumer card may ask for)
        const { assert!(GM_CHUNK_SMEM <= 101_376) };
        // the latent attention's (kernels/mla.cu, the layer kind's): the limits the launches hold to and the rows a block takes are the kernels' defines
        let define_mla = |n: &str| -> Option<usize> {
            MLA_SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!(
            (define_mla("GM_NOPE_MAX"), define_mla("GM_KSEL_MAX"), define_mla("GM_IDX_W"), define_mla("GM_RB"), define_mla("GM_VB")),
            (Some(256), Some(1024), Some(8), Some(8), Some(4))
        );
        for name in ["rms2_rows", "layernorm_rows", "scatter_rows", "mla_absorb", "idx_pool", "idx_score", "idx_pick", "mla_attn", "mla_expand_v"] {
            assert!(
                MLA_SRC.contains(&format!("extern \"C\" __global__ void {name}(")) || MLA_SRC.contains(&format!(") {name}(")),
                "no kernel {name} in kernels/mla.cu"
            );
        }
        for name in ["glm_kda_rec", "glm_kda_chunk", "glm_kda_out", "glm_kda_out_w32"] {
            assert!(
                SRC.contains(&format!("extern \"C\" __global__ void {name}(")) || SRC.contains(&format!(") {name}(")),
                "no kernel {name} in kernels/glm.cu"
            );
        }
    }

    /// The real model's numbers are ones the kernels run.
    #[test]
    fn the_real_config_is_one_the_engine_runs() {
        let c = crate::glm::tests::parse();
        assert!(
            c.kda.dim == GM_D
                && c.kda.conv_kernel == 4
                && c.hc.mult == GM_HC
                && c.hidden.is_multiple_of(8)
                && c.hidden <= 16 * 512
                && c.inter.is_multiple_of(8)
        );
        assert_eq!(c.hc_rows(), GM_NP);
        // the qkv product's rows are the convolutions' channels, and its state the delta rule's: 34 layers of 4 MiB
        assert_eq!((3 * c.qkv(), c.layers_of(GlmKind::Kda) * c.kda.heads * GM_D * GM_D * 4), (24576, 34 * 4 * 1024 * 1024));
    }

    /// The random tiny model `tools/glm_ref.py tiny` builds (its config is the fixture the script's selftest holds it to) is a model these kernels run: the real model's pattern in eight layers (three of Kimi delta attention with a
    /// dense MLP, a latent attention layer, three of Kimi delta attention, a latent attention layer; experts from the fourth), the numbers of its kernels, every matrix of its layers in whole blocks of the three block layouts.
    #[test]
    fn the_random_tiny_model_is_one_the_engine_takes() {
        use crate::glm::{IndexerDims, MlaDims};
        let c = ConfigGlm::parse(include_str!("../tests/fixtures/glm5_next_toy_config.json"), "toy", None).unwrap();
        let (k, m) = (GlmKind::Kda, GlmKind::Mla);
        assert_eq!(c.kinds, [k, k, k, m, k, k, k, m]);
        assert_eq!(c.moe, [false, false, false, true, true, true, true, true]);
        assert!(
            c.kda.dim == GM_D
                && c.kda.conv_kernel == 4
                && c.hc.mult == GM_HC
                && c.hidden.is_multiple_of(8)
                && c.hidden <= 16 * 512
                && c.inter.is_multiple_of(8)
                && c.kda.heads == 2
        );
        assert_eq!(
            (c.hidden, c.inter, c.experts.experts, c.experts.top_k, c.experts.inter, c.experts.shared_inter, c.vocab),
            (256, 384, 16, 4, 192, 192, 2048)
        );
        // latent attention: 2 heads of 64 with latents of 64 and a query latent of 64; an indexer of 4 heads of 32 channels that picks 2 pools of 4 tokens (so that a prompt of 12 tokens or more prunes)
        assert_eq!(c.mla, MlaDims { heads: 2, q_rank: 64, kv_rank: 64, nope: 64, v: 64 });
        assert_eq!(c.indexer, IndexerDims { heads: 4, dim: 32, topk: 8, kpool: 4, tail: true });
        crate::moe_glm::moe_cfg(&c).check().unwrap();
        for l in 0..c.layers {
            assert!(c.layer_packs(l).iter().filter(|p| !p.key.ends_with(".b_proj")).all(Pack35::packable), "layer {l}");
            // (the input gate's product has 2 rows, one for a head: the engine multiplies by it as a small dense matrix; the latent attention has none)
            assert_eq!(c.layer_packs(l).iter().find(|p| p.key.ends_with(".b_proj")).map(Pack35::packable), (c.kinds[l] == k).then_some(false), "layer {l}");
        }
        // (the shared expert's and the experts' matrices of a Kimi delta layer and of a latent attention layer)
        for (l, attn) in [(4, 7), (3, 5)] {
            for p in c.layer_packs(l).into_iter().skip(attn).chain(c.expert_packs(l)) {
                for tag in [glyd_gpu::pack::INT6_TAG, glyd_gpu::pack::INT5_TAG, glyd_gpu::pack::INT4_TAG] {
                    assert!(crate::moe::expert_strides(tag, p.rows(), p.cols).is_ok(), "{} is {} x {}", p.key, p.rows(), p.cols);
                }
            }
        }
        // the FP8 blocks divide the dense MLP's matrices (as the real model's divide its own), which transformers' dequantisation wants
        let [bm, bn] = c.fp8_block.unwrap();
        assert!(c.layer_packs(0).iter().skip(7).all(|p| p.parts.iter().all(|x| x.1 % bm == 0) && p.cols % bn == 0));
        // the kernels' limits: heads and latents the latent attention kernels take
        assert!(
            c.mla.nope <= 256
                && c.mla.kv_rank <= 512
                && c.mla.kv_rank.is_multiple_of(64)
                && c.indexer.dim <= 256
                && c.indexer.kpool <= 8
                && c.indexer.topk / c.indexer.kpool <= 1024
        );
    }

    /// What the loader holds of each layer: nine matrices for a Kimi delta layer with a dense MLP (the delta attention's seven, the MLP's two), seven where the feed-forward is a block of experts, none for a latent attention
    /// layer (its block's); and in the order `delta` and `dense_mlp` index them by.
    #[test]
    fn the_matrices_a_layer_holds() {
        let c = crate::glm::tests::parse();
        let keys = |l: usize| -> Vec<String> {
            layer_matrices(&c, l).iter().map(|p| p.key.rsplit("layers.").next().unwrap().split_once('.').unwrap().1.to_string()).collect()
        };
        assert_eq!(
            keys(1),
            [
                "self_attn.q_proj",
                "self_attn.o_proj",
                "self_attn.f_a_proj",
                "self_attn.f_b_proj",
                "self_attn.g_a_proj",
                "self_attn.g_b_proj",
                "self_attn.b_proj",
                "mlp.gate_proj",
                "mlp.down_proj"
            ]
        );
        assert_eq!(keys(4), keys(1)[..7]);
        assert_eq!(keys(44), keys(1)[..7]);
        assert!(layer_matrices(&c, 3).is_empty() && layer_matrices(&c, 43).is_empty());
        assert_eq!((0..c.layers).map(|l| layer_matrices(&c, l).len()).sum::<usize>(), 3 * 9 + 31 * 7);
        // (rows and columns: the products' shapes, as `delta` multiplies by them)
        let shapes: Vec<(usize, usize)> = layer_matrices(&c, 0).iter().map(|p| (p.rows(), p.cols)).collect();
        assert_eq!(shapes, [(24576, 4096), (4096, 8192), (128, 4096), (8192, 128), (128, 4096), (8192, 128), (64, 4096), (24576, 4096), (4096, 12288)]);
    }

    /// `layer_kept`'s order is the one the loader reads it in: the two norms, the mixing weights of the two blocks, then a Kimi delta layer's three convolutions, A_log, dt_bias and gated norm.
    #[test]
    fn the_kept_tensors_come_in_the_order_the_loader_reads() {
        let c = crate::glm::tests::parse();
        let names: Vec<String> = c.layer_kept(4).into_iter().map(|t| t.name.rsplit("layers.4.").next().unwrap().to_string()).collect();
        assert_eq!(
            names[..14],
            [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
                "self_attn.A_log",
                "self_attn.dt_bias",
                "self_attn.o_norm.weight"
            ]
        );
        // a layer's packs: the attention's, then the MLP's two; a Kimi delta layer's attention has seven
        assert_eq!((c.layer_packs(4).len(), c.layer_packs(3).len()), (9, 7));
        assert!(c.layer_packs(4)[7].key.ends_with("shared_experts.gate_proj") && c.layer_packs(0)[7].key.ends_with("mlp.gate_proj"));
    }

    /// An fp32 parameter is read as fp32 from an fp32 tensor and, widened exactly, from a bf16 one (a glyd-v7 directory may hold it so); any other dtype is refused.
    #[test]
    fn an_fp32_parameter_is_read_from_fp32_or_bf16() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-f32-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let w = [0x3f80u16, 0xbf00, 0x3dcd];
        let bf16: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let widened: Vec<u8> = w.iter().flat_map(|&x| ((x as u32) << 16).to_le_bytes()).collect();
        crate::pack::write_safetensors(
            &dir.join("model.safetensors"),
            &[],
            &[("a", "F32", vec![3], &widened[..]), ("b", "BF16", vec![3], &bf16[..]), ("c", "F16", vec![3], &bf16[..])],
        )
        .unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        assert_eq!(f32_bytes(&ck, "a").unwrap(), widened);
        assert_eq!(f32_bytes(&ck, "b").unwrap(), widened);
        assert!(f32_bytes(&ck, "c").unwrap_err().msg.contains("F32 or BF16"));
        assert!(f32_bytes(&ck, "d").unwrap_err().msg.contains("no tensor d"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
