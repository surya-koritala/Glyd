//! GLM-5.3-Flash's and Kimi-K3's kernels (kernels/glm.cu, kernels/mla.cu) with typed launches: Kimi delta attention, the residual streams, the latent attention's (`KernGlm`),
//! and the streams' mixing weights a launch takes (`Hc`), which the executor and Condor run.

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Result, Stream};
use crate::launch;
use crate::runtime::kinds::Rounding;

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
        self.hc.streams(s, e, x, n, d)
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
        launch!(self.select, [n, 1, 1], [1024, 1, 1], s; scores, pmax, pos, kp, ksel, kp, tail as i32, sel, tmax, cnt, 0u64, 0i32)
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
