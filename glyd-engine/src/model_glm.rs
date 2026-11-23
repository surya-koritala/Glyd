//! FROZEN (R13, #234): the engine here (`EngineGlm`, with src/glm_serve.rs, src/glm_cb.rs, src/mtp_glm.rs, src/draft_loop.rs and the `glm*` development commands of src/kcheck_glm.rs) stays as it is until the dynamic route has what only it has (the experts tiered through the codes pack, and MTP drafting); no new features here. Its blocks (`KernGlm`, `WeightsGlm`, `MlaBlock`, `MoeBlock`, `GlmHost`, src/mla_glm.rs, src/moe_glm.rs) are the dynamic route's for GLM, Kimi and DeepSeek, and move under src/runtime/ when the engine goes.
//! GLM-5.3-Flash on the GPU, one sequence: four residual streams per token, each layer two blocks (`x = mix(x, block(norm(collapse(x))))`, the mixing weights learned from the streams: mHC), the first a token mixer (Kimi delta
//! attention, or the latent attention over the keys an indexer picks, src/mla_glm.rs: `MlaBlock` is the slot it plugs in by, and `NoMla` the stand-in for a model loaded without one) and the second a dense MLP (layers 0 to 2) or a mixture of experts
//! (`MoeBlock`, src/moe_glm.rs: the block on the mixture-of-experts core; `NoMoe` the stand-in for a model loaded without one), then the mean of the streams, the final norm and the LM head. The Kimi delta layers and the
//! streams' kernels are this file and kernels/glm.cu; a prompt in chunks and a decode step as one CUDA graph, greedy tokens. Kimi delta attention is the Gated DeltaNet of kernels/qwen35.cu with a decay for each key channel
//! (g = -5 sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias))), three convolutions side by side (q, k and v: `conv_silu`, the engine's), l2-normed q and k, and a sigmoid-gated norm of its outputs.
//! The weights are a glyd-v7 directory's (`glyd-engine pack MODEL OUT --layout i6|i5`) or a Hub-format directory's (FP8 with 128 x 128 block scales or bf16: a matrix is packed as it loads, from the bf16 it dequantises to).
//! The layers' math: tools/glm_ref.py; the config and tensor names: src/glm.rs.

use crate::dev::{copy_d2d_async, copy_d2h_async, copy_h2d_async, Buf, Dev, Graph, Pinned};
use crate::ffi::{err, Error, Lib, Result, Stream};
use crate::glm::{ConfigGlm, GlmKind};
use crate::kern::embed_plan;
use crate::kern::Kern;
use crate::kern35::Mat;
use crate::launch;
use crate::moe::{Act, MoeKern, MoeScratch};
use crate::qwen35::Pack35;
use crate::runtime::kinds::Rounding;
use crate::sched::Sample;
use crate::weights::{choose_layout, ordered, upload_bytes, upload_packed, v7_manifest, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::HashMap;
use std::path::Path;

// The kernels' names: those the .cu sources give them, in every build (kname!).
/// A development build's timeline marker: `mark!(engine, "stage")` records an event on the stream (also while a CUDA graph is captured) when `EngineGlm::prof` is on, so that `glmprof` can say how long each stage of a layer
/// takes in a graph's replay. Not compiled into a release build.
macro_rules! mark {
    ($e:expr, $label:literal) => {
        #[cfg(feature = "dev")]
        $e.mark($label);
    };
}
pub(crate) use mark;

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
    // the latent attention's
    rms2: crate::kt::glm_rms2,
    layernorm: crate::kt::glm_layernorm,
    scatter: crate::kt::glm_scatter,
    absorb: crate::kt::glm_absorb,
    kp_keys: crate::kt::glm_kp_keys,
    idx_score: crate::kt::glm_idx_score,
    select: crate::kt::glm_select,
    mla_attn: crate::kt::glm_mla_attn,
    /// The shared memory a block of `mla_attn` may take here (the GPU's most, opted in to at load).
    mla_smem: u32,
    mla_merge: crate::kt::glm_mla_merge,
    expand_v: crate::kt::glm_expand_v,
    /// The streams' split projection's partial sums, for up to HC_FEW rows.
    hc_part: Buf,
}

/// Rows at most that the mixing weights' split projection takes (a decode step, the rows of a verified draft).
pub const HC_FEW: u32 = crate::dynamic::residual::HC_FEW;

/// The dynamic shared memory `glm_mla_attn` may take on `dev`: the GPU's most a block, less a KiB for the kernel's own (its block sums' 256 bytes).
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
        let (mla_attn, mla_smem) = (dev.typed::<crate::kt::glm_mla_attn>()?, mla_smem(dev)?);
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
        launch!(self.rms2, [n, 1, 1], [256, 1, 1], s; x, xs, w, y, ys, dst, k, eps)
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
        launch!(self.layernorm, [n, 1, 1], [128, 1, 1], s; x, xs, w, b, y, ys, dst, k, eps)
    }

    /// Row pos[i] of dst (rows `ds` apart) = row i of src (rows `ss` apart), `k` values each.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scatter(&self, s: Stream, src: u64, ss: i32, dst: u64, ds: i32, pos: u64, k: i32, n: u32) -> Result<()> {
        launch!(self.scatter, [n, 1, 1], [256, 1, 1], s; src, ss, dst, ds, pos, k)
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
        launch!(self.kp_keys, [(n as u32).div_ceil(kp as u32) + 1, 1, 1], [128, 1, 1], s; ik, gs, ape, pk, pos, n, d, kp)
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
        launch!(self.idx_score, [(pmax as u32).div_ceil(8), n, 1], [256, 1, 1], smem smem, s; iq, iqs, wts, wsr, pk, pos, scores, pmax, ih, d, kp, ksel, scale, wscale)
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
        launch!(self.select, [n, 1, 1], [1024, 1, 1], s; scores, pmax, pos, sel, cnt, tmax, kp, ksel, tail as i32)
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
        launch!(self.mla_attn, [heads as u32, n, splits as u32], [256, 1, 1], smem smem, s; qa, lat, sel, cnt, olat, part, tmax, heads, kr, ks, scale, splits, tile)?;
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

/// A mixture-of-experts layer's block, which the engine's layers call and does not itself hold: the router, the experts and the shared expert (src/moe_glm.rs). `h` [n, hidden] bf16 is the layer's normed input, `out`
/// [n, hidden] bf16 the block's output (the engine spreads it over the streams).
pub trait MoeBlock {
    /// # Safety
    /// Live device buffers of those sizes, used on the engine's stream.
    unsafe fn forward(&self, e: &EngineGlm, h: u64, n: usize, out: u64) -> Result<()>;
    /// The core's layer where the block is one, all its experts on the GPU (the dynamic engine runs it itself).
    fn layer(&self) -> Option<&crate::moe::MoeLayer> {
        None
    }
    /// Whether `forward` can be captured in a CUDA graph: no host read or copy that depends on the tokens.
    fn graphable(&self) -> bool {
        false
    }
    /// Device bytes it holds.
    fn bytes(&self) -> usize {
        0
    }
    /// The (rows, columns, whether 12-bit) of every pack it multiplies through `EngineGlm::lin`, so that the engine sizes the library's workspace and its decoded-matrix scratch for them.
    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        Vec::new()
    }
    /// Whether a decode step (one row) can be run in three parts, `front`, `host` and `back`, the GPU's two in CUDA graphs (`EngineGlm::step`'s segmented decode): the experts are served by a host whose wait is one a layer.
    fn segmented(&self) -> bool {
        false
    }
    /// The GPU work up to the host's wait: the router, the rotation of the input, the experts' numbers, the choices on their way to the host (and what does not need them, the shared expert). Captured in a graph.
    ///
    /// # Safety
    /// As `forward`; `n` is 1.
    unsafe fn front(&self, _: &EngineGlm, _: u64, _: usize) -> Result<()> {
        err("this block does not run in parts")
    }
    /// Its buffers right for a step in parts (before a capture: a capture may not reallocate or wait).
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn prepare(&self, _: &EngineGlm) -> Result<()> {
        Ok(())
    }
    /// The host's part: the wait for the choices, the plan and the copies, the products of the experts in places, the host's pairs and their sums sent up. Not captured.
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn host(&self, _: &EngineGlm) -> Result<()> {
        err("this block does not run in parts")
    }
    /// The GPU work after the host's part: the sum of the pairs and the shared expert's addition into `out`. Captured in a graph.
    ///
    /// # Safety
    /// As `forward`; `n` is 1.
    unsafe fn back(&self, _: &EngineGlm, _: u64, _: usize) -> Result<()> {
        err("this block does not run in parts")
    }
}

/// What a mixture-of-experts layer is where no block was given to `EngineGlm::new` (src/moe_glm.rs makes one): it says so where it is run.
pub struct NoMoe;

impl MoeBlock for NoMoe {
    unsafe fn forward(&self, _: &EngineGlm, _: u64, _: usize, _: u64) -> Result<()> {
        err("a mixture-of-experts layer with no block of experts to run it: the engine was loaded without one")
    }
}

/// Makes layer `l`'s block of experts: the layer's number, the config, the checkpoint, the layout of the other packs.
pub type MoeFactory<'a> = &'a dyn Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>>;

/// What a GLM block asks of the engine it runs in (this one, or the dynamic engine): GLM's kernels, the stream, the products, the context and chunk it was made for, the latent attention's scratch.
pub trait GlmHost {
    fn kg(&self) -> &KernGlm;
    fn stream(&self) -> Stream;
    fn sync(&self) -> Result<()>;
    /// y [m, rows] = x [m, cols] W^T.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()>;
    fn ctx(&self) -> usize;
    fn chunk(&self) -> usize;
    fn hidden(&self) -> usize;
    fn mla_scratch(&self) -> &std::cell::OnceCell<crate::mla_glm::MlaScratch>;
    /// A development build's stage marker (a profile's).
    fn mark(&self, _what: &'static str) {}
    /// The sequences a call's n rows belong to, in order: each one's first row, its rows, and which sequence it is (its caches: 0 to sched::MAX_USERS - 1). One sequence, 0, where the engine runs one at a time; several
    /// in a step of several sequences (src/dynamic/rows.rs).
    fn seq_rows(&self, n: usize) -> Vec<(usize, usize, usize)> {
        vec![(0, n, 0)]
    }
}

impl GlmHost for EngineGlm {
    fn kg(&self) -> &KernGlm {
        &self.k
    }
    fn stream(&self) -> Stream {
        self.dev.stream
    }
    fn sync(&self) -> Result<()> {
        self.dev.sync()
    }
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        EngineGlm::lin(self, w, x, m, y)
    }
    fn ctx(&self) -> usize {
        self.ctx
    }
    fn chunk(&self) -> usize {
        self.chunk
    }
    fn hidden(&self) -> usize {
        self.w.cfg.hidden
    }
    fn mla_scratch(&self) -> &std::cell::OnceCell<crate::mla_glm::MlaScratch> {
        &self.mla_scratch
    }
    #[cfg(feature = "dev")]
    fn mark(&self, what: &'static str) {
        EngineGlm::mark(self, what)
    }
}

/// A latent attention layer's mixer (the next round): `h` [n, hidden] is the normed input, `pos` the rows' positions (int32 on the device) with `start` tokens before them, `out` [n, hidden] the mixer's output.
pub trait MlaBlock {
    /// # Safety
    /// Live device buffers of those sizes, used on the engine's stream.
    unsafe fn forward(&self, e: &dyn GlmHost, h: u64, n: usize, pos: u64, start: usize, out: u64) -> Result<()>;
    fn graphable(&self) -> bool {
        false
    }
    fn bytes(&self) -> usize {
        0
    }
    /// The caches one sequence takes at a context of `ctx` tokens (made at its first call).
    fn seq_bytes(&self, _ctx: usize) -> usize {
        0
    }
    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        Vec::new()
    }
    /// A development check's view after `forward` on n rows at positions `pos0..`: the stages it keeps, named, each the bytes of those n rows (the checks append the calls').
    fn stages(&self, _e: &dyn GlmHost, _n: usize, _pos0: usize) -> Result<Vec<(&'static str, Vec<u8>)>> {
        Ok(Vec::new())
    }
    /// After the last call of a case, the tokens in the sequence: what the whole sequence left (the indexer's pool keys).
    fn finish(&self, _e: &dyn GlmHost, _total: usize) -> Result<Vec<(&'static str, Vec<u8>)>> {
        Ok(Vec::new())
    }
    /// A development check asks the block to keep every row's stages of a call that it runs a block of rows at a time (what `stages` then shows of a call of more rows than a block) from now on, or not.
    fn collect_stages(&self, _on: bool) {}
}

/// What a latent attention layer is where no block was given to `EngineGlm::new` (src/mla_glm.rs makes one): it holds the layer's place and refuses to run.
pub struct NoMla;

impl MlaBlock for NoMla {
    unsafe fn forward(&self, _: &dyn GlmHost, _: u64, _: usize, _: u64, _: usize, _: u64) -> Result<()> {
        err("a latent attention layer (11 of the 45) with no block to run it: the engine was loaded without one (src/mla_glm.rs makes it)")
    }
}

/// Makes layer `l`'s latent attention block: the layer's number, the config, the checkpoint, the layout of the packs.
pub type MlaFactory<'a> = &'a dyn Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<dyn MlaBlock>>;

/// A layer's mixing weights: the projection of the flattened streams [24, 4 hidden] (bf16), the base [24] and the three scales (fp32).
pub struct Hc {
    pub proj: Buf,
    pub base: Buf,
    pub scale: Buf,
}

/// A layer's buffers, by kind: its matrices (Kimi delta attention: qkv, o, f_a, f_b, g_a, g_b, b; then a dense MLP's gate and up, down; a latent attention layer's and a mixture of experts' are their blocks') and small tensors.
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
    pub mla: Option<Box<dyn MlaBlock>>,
    pub moe: Option<Box<dyn MoeBlock>>,
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

impl WeightsGlm {
    /// A Hub-format directory (config.json, the safetensors: FP8 with block scales or bf16) or a glyd-v7 directory (`glyd-engine pack MODEL OUT --layout i6|i5`): the layers of `only` (all, with the embedding, the head and the
    /// final norm, where None) checked against the weight map, their matrices packed here by `threads` workers in `layout` (None: the GPU's) and copied to the GPU as they are done. A glyd-v7 directory is at one level
    /// (`layout` None, or the same): the packs it holds saved are read as they are, the rest (a matrix the packer does not take) as the Hub's. A latent attention layer holds its norms and mixing weights and its block, which
    /// is `mla`'s (else `NoMla`); a mixture-of-experts layer the same with `moe`'s (else `NoMoe`): a block gets the checkpoint and the layout and loads what it needs of the layer's tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dir: &Path,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        only: Option<&[usize]>,
        moe: Option<MoeFactory<'_>>,
        mla: Option<MlaFactory<'_>>,
    ) -> Result<WeightsGlm> {
        WeightsGlm::load_ends(dir, threads, layout, cc, total_mem, only, None, moe, mla)
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
        moe: Option<MoeFactory<'_>>,
        mla: Option<MlaFactory<'_>>,
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
            let mla_block: Option<Box<dyn MlaBlock>> = if kind == GlmKind::Mla {
                Some(match &mla {
                    Some(make) => make(l, &cfg, &ck, layout)?,
                    None => Box::new(NoMla),
                })
            } else {
                None
            };
            let moe_block: Option<Box<dyn MoeBlock>> = if cfg.moe[l] {
                Some(match &moe {
                    Some(make) => make(l, &cfg, &ck, layout)?,
                    None => Box::new(NoMoe),
                })
            } else {
                None
            };
            bytes += norm_a.bytes
                + norm_f.bytes
                + hc.iter().map(|h| h.proj.bytes + h.base.bytes + h.scale.bytes).sum::<usize>()
                + mats.iter().map(Mat::bytes).sum::<usize>()
                + small.iter().map(|b| b.bytes).sum::<usize>();
            bytes += mla_block.as_ref().map_or(0, |b| b.bytes()) + moe_block.as_ref().map_or(0, |b| b.bytes());
            layers[l] = Some(LayerGlm { kind, idx, norm_a, norm_f, hc, mats, small, mla: mla_block, moe: moe_block });
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

/// How a run is set up.
pub struct OptsGlm {
    /// Positions a sequence may reach (prompt and reply).
    pub ctx: usize,
    /// Tokens a prompt chunk takes at most.
    pub chunk: usize,
    pub graph: bool,
    pub device: i32,
    pub layout: Option<Layout>,
    /// A prompt's delta rule in chunks of GM_L tokens (else one token after another, as a decode step).
    pub chunked: bool,
    /// Load just these layers (a development check's); none: the model.
    pub only: Option<Vec<usize>>,
    /// Load the MTP head too (`EngineGlm::mtp`).
    pub mtp: bool,
    /// Room to verify a draft of up to `SPEC_ROWS - 1` tokens in one pass and to rewind the delta rules' states to where the draft was right (`EngineGlm::verify`, `commit`).
    pub spec: bool,
}

impl Default for OptsGlm {
    fn default() -> OptsGlm {
        OptsGlm { ctx: 4096, chunk: 2048, graph: true, device: 0, layout: None, chunked: false, only: None, mtp: false, spec: false }
    }
}

pub struct EngineGlm {
    pub dev: Dev,
    pub lib: Lib,
    pub(crate) kern: Kern,
    pub(crate) k: KernGlm,
    pub(crate) kmoe: MoeKern,
    pub w: WeightsGlm,
    chunk: usize,
    ctx: usize,
    pub(crate) chunked: bool,
    // activations, `chunk` rows each: the streams x [4, hidden], the normed block input h, the block's output o, and Kimi delta attention's
    pub(crate) x: Buf,
    pub(crate) h: Buf,
    pub(crate) o: Buf,
    pub(crate) mix: Buf,
    pub(crate) xc: Buf,
    pub(crate) fg: Buf,
    pub(crate) og: Buf,
    pub(crate) low: Buf,
    pub(crate) bb: Buf,
    pub(crate) core: Buf,
    pub(crate) att: Buf,
    /// The mixing weights [rows, 24] fp32 of the token mixer's block and of the feed-forward's.
    pub(crate) hcp: [Buf; 2],
    hlast: Buf,
    logits: Buf,
    scratch: Buf,
    embed_groups: Buf,
    embed_pairs: Buf,
    pub(crate) pos: Buf,
    pub(crate) st: Buf,
    tok: Buf,
    val: Buf,
    // the products
    ws: Buf,
    done: Buf,
    blas: Option<crate::ffi::Blas>,
    own: bool,
    scratch_w: Buf,
    gpu_code: i64,
    // what a sequence holds: the delta rules' states (fp32 [layers of Kimi delta attention, heads, 128, 128]) and the convolutions' (bf16 [layers, 3, 3 heads 128])
    pub ssm: Buf,
    pub convst: Buf,
    graph: Option<Graph>,
    /// The decode step in parts (`forward_segmented`): the graphs of the stretches of GPU work between the host's waits, made at the first step (a layer's, by the layer's number + 1; 0 the stretch before the first), and whether
    /// a step is run so (a development session may turn it off).
    segs: std::cell::RefCell<Vec<Option<Graph>>>,
    pub segmented: std::cell::Cell<bool>,
    /// The tensor-core products for 1 to 8 rows of the Int4, Int5 and Int6 layouts (src/skinny.rs), used for a decode step's and a verification's products where `skinny_on`.
    skinny: crate::skinny::SkinnyKern,
    pub skinny_on: std::cell::Cell<bool>,
    host: Pinned,
    /// Tokens in the sequence.
    pub n: usize,
    /// Device bytes of the activations and workspaces.
    pub work_bytes: usize,
    /// The mixture-of-experts layers' scratch, one for all of them: the first block that runs makes it (src/moe_glm.rs).
    pub(crate) moe_scratch: std::cell::OnceCell<MoeScratch>,
    /// The latent attention layers' scratch, one for all of them (src/mla_glm.rs): the first block that runs makes it.
    pub(crate) mla_scratch: std::cell::OnceCell<crate::mla_glm::MlaScratch>,
    /// A development check's look at each layer after it ran (layer, rows, tokens before them), in a pass that is not a graph's: the experts the routers chose and the tokens the indexers picked, to teacher-force a reference.
    pub trace: Option<Trace>,
    /// What a verified draft keeps (when asked for in `OptsGlm`), and whether the pass in flight is saving its Kimi delta inputs.
    pub(crate) spec: Option<SpecBufs>,
    /// The rows of the verified pass that `commit` has not yet settled (none: no pass is waiting).
    pending: std::cell::Cell<Option<usize>>,
    /// The MTP head (src/mtp_glm.rs), where `OptsGlm::mtp` asked for it, and the final normed hidden states of the rows of a prompt chunk it is run on.
    pub mtp: Option<crate::mtp_glm::MtpHead>,
    hall: Option<Buf>,
    save: std::cell::Cell<bool>,
    /// The stage markers recorded while it is Some (`mark!`): (stage, event), in order.
    #[cfg(feature = "dev")]
    pub prof: std::cell::RefCell<Option<Vec<(&'static str, crate::dev::Ev)>>>,
}

/// Rows a verified draft takes at most: the last token and up to seven guesses.
pub const SPEC_ROWS: usize = HC_FEW as usize;

/// What verifying a draft in one pass keeps to be able to rewind the Kimi delta layers to where it was right: their states and convolutions' states as they were before the pass, and per layer the pass's rows of the
/// convolution's input (`mix`) and output (`xc`) and of the two gates (`fg`, `bb`), from which the recurrence is run again over the rows kept. (The latent attention's caches need no rewinding: a position is written before
/// it is read, and the pool keys are made again where a later call completes a pool.) Also the pass's rows of the final norm, the logits, and the argmax.
pub(crate) struct SpecBufs {
    ssm0: Buf,
    conv0: Buf,
    save: Buf,
    per_layer: usize,
    hl: Buf,
    lg: Buf,
    tk: Buf,
    vl: Buf,
}

impl SpecBufs {
    /// Layer `idx`'s saved (mix, xc, fg, bb) for SPEC_ROWS rows.
    fn slot(&self, idx: usize, qd: usize) -> (u64, u64, u64, u64) {
        let base = self.save.ptr + (idx * self.per_layer) as u64;
        let (mix, xc) = (SPEC_ROWS * 3 * qd * 2, SPEC_ROWS * 3 * qd * 2);
        (base, base + mix as u64, base + (mix + xc) as u64, base + (mix + xc + SPEC_ROWS * qd * 2) as u64)
    }
}

/// How a speculative generation went: passes made, guesses made and guesses that were right.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub passes: usize,
    pub drafted: usize,
    pub accepted: usize,
}

/// A development check's look at each layer after it ran: (the engine, the layer, its rows, the tokens before them).
pub type Trace = Box<dyn Fn(&EngineGlm, usize, usize, usize)>;

/// 64-row groups of the embedding decoded at once for a prompt chunk.
const EMBED_GROUPS: usize = 32;

impl EngineGlm {
    pub fn new(dir: &Path, o: &OptsGlm, moe: Option<MoeFactory<'_>>, mla: Option<MlaFactory<'_>>) -> Result<EngineGlm> {
        let dev = Dev::new(o.device)?;
        let lib = Lib::find()?;
        let kern = Kern::load(&dev)?;
        let k = KernGlm::load(&dev)?;
        let kmoe = MoeKern::load(&dev)?;
        let skinny = crate::skinny::SkinnyKern::load(&dev)?;
        let (free0, _) = dev.mem_info()?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let w = WeightsGlm::load(dir, threads, o.layout, dev.cc_pair(), dev.total_mem as u64, o.only.as_deref(), moe, mla)?;
        let mtp = if o.mtp && w.cfg.mtp_layers > 0 { Some(crate::mtp_glm::MtpHead::load(&w.cfg, dir, w.layout, moe, mla)?) } else { None };
        let c = &w.cfg;
        if c.kda.dim != GM_D || c.kda.conv_kernel != 4 || c.hc.mult != GM_HC || !c.hidden.is_multiple_of(8) || c.hidden > 16 * 480 || !c.inter.is_multiple_of(8)
        {
            return err(format!(
                "Kimi delta heads of {} channels with a convolution of {}, {} residual streams, a width of {} and an MLP of {}: this engine runs heads of {GM_D} channels, a convolution of 4, {GM_HC} streams, a width divisible by 8 up to 7,680",
                c.kda.dim, c.kda.conv_kernel, c.hc.mult, c.hidden, c.inter
            ));
        }
        let (hid, qd) = (c.hidden, c.qkv());
        let ch = o.chunk.max(1);
        let act = |cols: usize, rows: usize| Buf::new(rows * cols * 2);
        let x = act(GM_HC * hid, ch)?;
        let (h, o_) = (act(hid, ch)?, act(hid, ch)?);
        let (mix, xc) = (act((3 * qd).max(2 * c.inter), ch)?, act((3 * qd).max(c.inter), ch)?);
        let (fg, og, low, bb, core, att) = (act(qd, ch)?, act(qd, ch)?, act(GM_D, ch)?, act(c.kda.heads, ch)?, act(qd, ch)?, act(qd, ch)?);
        let hcp = [Buf::zeroed(ch * GM_NP * 4)?, Buf::zeroed(ch * GM_NP * 4)?];
        let (hlast, logits) = (act(hid, 1)?, act(c.vocab, 1)?);
        let scratch = Buf::new(64 * hid * 2)?;
        let embed_groups = Buf::new(EMBED_GROUPS * 64 * hid * 2)?;
        let embed_pairs = Buf::new(ch * 8)?;
        let (pos, st) = (Buf::zeroed(4 * ch)?, Buf::zeroed(16)?);
        let (tok, val) = (Buf::zeroed(4)?, Buf::zeroed(4)?);
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let blas = if ch > 64 && (want == "blas" || (want.is_empty() && dev.cc != (8, 9))) { crate::ffi::Blas::find() } else { None };
        let own = want == "own" || (want.is_empty() && blas.is_none() && matches!(dev.cc.0, 8 | 9));
        let gpu_code = lib.gpu()? as i64;
        let mut shapes: Vec<(i64, i64, bool)> = w
            .layers
            .iter()
            .flatten()
            .flat_map(|l| l.mats.iter())
            .filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None })
            .collect();
        shapes.extend(w.layers.iter().flatten().filter_map(|l| l.moe.as_ref()).flat_map(|b| b.pack_shapes()));
        shapes.extend(w.layers.iter().flatten().filter_map(|l| l.mla.as_ref()).flat_map(|b| b.pack_shapes()));
        shapes.extend(mtp.iter().flat_map(|m| m.pack_shapes()));
        let scratch_w = if blas.is_some() || own { Buf::new(shapes.iter().map(|s| (s.0 * s.1) as usize * 2).max().unwrap_or(16))? } else { Buf::new(16)? };
        shapes.sort_unstable();
        shapes.dedup();
        let mut need = 0usize;
        for m_ in 1..=ch as i64 {
            for &(oo, kk, twelve) in &shapes {
                need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
            }
        }
        if let Some(head) = w.lm_head() {
            for m_ in 1..=if o.spec { SPEC_ROWS as i64 } else { 1 } {
                need = need.max(lib.linear_workspace(head.pack.is_twelve(), head.pack.rows, head.pack.cols, m_)?);
            }
        }
        let ws = Buf::zeroed(need)?;
        let done = Buf::zeroed(4 << 18)?;
        let ctx = o.ctx.div_ceil(16) * 16;
        let ssm = Buf::zeroed(c.layers_of(GlmKind::Kda) * c.kda.heads * GM_D * GM_D * 4)?;
        let convst = Buf::zeroed(c.layers_of(GlmKind::Kda) * 3 * 3 * qd * 2)?;
        let host = Pinned::new(4096)?;
        let spec = if o.spec && c.layers_of(GlmKind::Kda) > 0 {
            let kl = c.layers_of(GlmKind::Kda);
            let per_layer = SPEC_ROWS * (3 * qd + 3 * qd + qd + c.kda.heads) * 2;
            Some(SpecBufs {
                ssm0: Buf::zeroed(ssm.bytes)?,
                conv0: Buf::zeroed(convst.bytes)?,
                save: Buf::zeroed(kl * per_layer)?,
                per_layer,
                hl: Buf::zeroed(SPEC_ROWS * hid * 2)?,
                lg: Buf::zeroed(SPEC_ROWS * c.vocab * 2)?,
                tk: Buf::zeroed(SPEC_ROWS * 4)?,
                vl: Buf::zeroed(SPEC_ROWS * 4)?,
            })
        } else {
            None
        };
        let used = free0.saturating_sub(dev.mem_info()?.0);
        let work_bytes = used.saturating_sub(w.bytes + ssm.bytes + convst.bytes);
        let mut e = EngineGlm {
            dev,
            lib,
            kern,
            k,
            kmoe,
            w,
            chunk: ch,
            ctx,
            chunked: o.chunked,
            x,
            h,
            o: o_,
            mix,
            xc,
            fg,
            og,
            low,
            bb,
            core,
            att,
            hcp,
            hlast,
            logits,
            scratch,
            embed_groups,
            embed_pairs,
            pos,
            st,
            tok,
            val,
            ws,
            done,
            blas,
            own,
            scratch_w,
            gpu_code,
            ssm,
            convst,
            graph: None,
            segs: std::cell::RefCell::new(Vec::new()),
            segmented: std::cell::Cell::new(false),
            skinny,
            skinny_on: std::cell::Cell::new(false),
            host,
            n: 0,
            work_bytes,
            moe_scratch: std::cell::OnceCell::new(),
            mla_scratch: std::cell::OnceCell::new(),
            trace: None,
            spec,
            pending: std::cell::Cell::new(None),
            hall: if mtp.is_some() { Some(Buf::zeroed(ch * hid * 2)?) } else { None },
            mtp,
            save: std::cell::Cell::new(false),
            #[cfg(feature = "dev")]
            prof: std::cell::RefCell::new(None),
        };
        let graphable = e.w.layers.iter().flatten().all(|l| l.moe.as_ref().is_none_or(|b| b.graphable()) && l.mla.as_ref().is_none_or(|b| b.graphable()));
        if o.graph && e.w.whole() && graphable {
            e.capture_graph()?;
        }
        Ok(e)
    }

    /// Records a marker event on the stream while `prof` is on.
    #[cfg(feature = "dev")]
    pub(crate) fn mark(&self, what: &'static str) {
        crate::trace::mark(self.dev.stream, what);
        if let Some(v) = self.prof.borrow_mut().as_mut() {
            if let Ok(ev) = crate::dev::Ev::new() {
                if ev.record_in_graph(self.dev.stream).is_ok() {
                    v.push((what, ev));
                }
            }
        }
    }

    pub fn describe(&self) -> String {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        format!(
            "{} (sm_{}{}): weights {:.3} GiB (layout {:?}), states {:.3} GiB, workspaces {:.3} GiB, {:.2} of {:.2} GiB in use",
            self.dev.name,
            self.dev.cc.0,
            self.dev.cc.1,
            gib(self.w.bytes),
            self.w.layout,
            gib(self.ssm.bytes + self.convst.bytes),
            gib(self.work_bytes),
            gib(total - free),
            gib(total)
        )
    }

    /// Last forward pass's BF16 logits, widened exactly for development scoring.
    pub fn logits(&self) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut words = vec![0u16; self.w.cfg.vocab];
        self.logits.read(0, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

    /// The last forward pass's final normed hidden state of its last row (bf16 bits), which the LM head multiplied.
    pub fn hidden(&self) -> Result<Vec<u16>> {
        self.dev.sync()?;
        let mut v = vec![0u16; self.w.cfg.hidden];
        self.hlast.read(0, &mut v)?;
        Ok(v)
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// Whether the engine was made with room for a verified draft (`OptsGlm::spec`).
    pub fn has_draft_room(&self) -> bool {
        self.spec.is_some()
    }

    /// The positions a sequence may reach.
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// A new sequence: the states zeroed.
    pub fn reset(&mut self) -> Result<()> {
        self.n = 0;
        self.pending.set(None);
        self.ssm.fill_async(0, self.dev.stream)?;
        self.convst.fill_async(0, self.dev.stream)?;
        self.dev.sync()
    }

    /// Whether a product of `m` rows by `p` decodes the matrix and multiplies by a dense product (the library's route says DECODE or AHEAD), as Qwen3.5's engine's.
    unsafe fn decoded_way(&self, p: &PackBuf, m: usize) -> Result<bool> {
        if (self.blas.is_none() && !self.own) || m <= 64 || p.pack.cols % 64 != 0 || p.pack.rows % 128 != 0 {
            return Ok(false);
        }
        let route = self.lib.route(self.gpu_code, p.pack.is_twelve(), p.pack.rows, p.pack.cols, m as i64)?;
        Ok(route == 0 || route == 5)
    }

    pub(crate) unsafe fn lin_packed(&self, p: &PackBuf, x: u64, m: usize, y: u64) -> Result<()> {
        if self.skinny_on.get() && (1..=8).contains(&m) && p.pack.rows >= 2048 && crate::skinny::SkinnyKern::takes(&p.pack) {
            return self.skinny.gemv(self.dev.stream, &p.pack, x, y, m);
        }
        if self.decoded_way(p, m)? {
            self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, self.dev.stream)?;
            if let Some(b) = &self.blas {
                return b.gemm_nt(self.dev.stream, self.scratch_w.ptr, x, y, p.pack.rows as i32, p.pack.cols as i32, m as i32);
            }
            let tile = self.kern.pick_tile(m, p.pack.rows as usize, p.pack.cols as usize);
            return self.kern.gemm(self.dev.stream, tile, x, self.scratch_w.ptr, y, m as i32, p.pack.rows as i32, p.pack.cols as i32);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, self.dev.stream)
    }

    /// Y [m, rows] = X [m, cols] W^T, X and Y dense.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    pub unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        match w {
            Mat::Packed(p) => self.lin_packed(p, x, m, y),
            Mat::Dense(b, rows, cols) => self.k.small_linear(self.dev.stream, x, *cols as i32, b.ptr, y, *rows as i32, m as u32, *rows as i32, *cols as i32),
        }
    }

    /// Token rows into `dst` (bf16 [tokens, hidden]; as the Qwen3.5 engine's `embed`, whose rows go to its stream).
    unsafe fn embed(&self, tokens: &[u32]) -> Result<()> {
        self.embed_into(tokens, self.o.ptr)
    }

    /// The embedding rows of `tokens` into `dst` (bf16 [tokens, hidden]).
    ///
    /// # Safety
    /// A live device range of that size, used on the engine's stream.
    pub(crate) unsafe fn embed_into(&self, tokens: &[u32], dst: u64) -> Result<()> {
        let hid = self.w.cfg.hidden;
        let s = self.dev.stream;
        let table = self.w.embed.as_ref().ok_or_else(|| Error("the embedding is not loaded".into()))?;
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.w.cfg.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.w.cfg.vocab));
        }
        if tokens.len() == 1 {
            let t = tokens[0] as i64;
            let row0 = (t / 64) * 64;
            self.lib.unpack(&table.pack, row0, 64, self.scratch.ptr, s)?;
            return self.kern.copy_row(s, self.scratch.ptr, (t - row0) as i32, hid as i32, dst);
        }
        let (groups, by_batch) = embed_plan(tokens, EMBED_GROUPS);
        let counts: Vec<usize> = by_batch.iter().map(Vec::len).collect();
        let pairs: Vec<i32> = by_batch.iter().flatten().flat_map(|&(d, s_)| [d, s_]).collect();
        self.embed_pairs.write(0, &pairs)?;
        let mut done = 0usize;
        for (b, batch) in groups.chunks(EMBED_GROUPS).enumerate() {
            for (j, &g) in batch.iter().enumerate() {
                self.lib.unpack(&table.pack, g * 64, 64, self.embed_groups.ptr + (j * 64 * hid * 2) as u64, s)?;
            }
            self.kern.copy_rows(s, self.embed_groups.ptr, self.embed_pairs.ptr + (done * 8) as u64, counts[b] as i32, hid as i32, dst)?;
            done += counts[b];
        }
        Ok(())
    }

    /// A Kimi delta layer's mixer on the n rows in `h` (the layer's normed input) into `o`: the q, k and v product and the forget and output gates' two products each, the input gate's, the convolution, the delta rule
    /// (a prompt's in chunks or token by token, a step by the recurrence), the gated norm, the o product.
    unsafe fn delta(&self, ly: &LayerGlm, n: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (heads, qd) = (c.kda.heads as i32, c.qkv());
        let (m, sm) = (&ly.mats, &ly.small);
        let conv = self.convst.ptr + (ly.idx * 3 * 3 * qd * 2) as u64;
        let state = self.ssm.ptr + (ly.idx * c.kda.heads * GM_D * GM_D * 4) as u64;
        // (a verified draft's pass keeps this layer's inputs of the recurrence, for a rewind: `SpecBufs`)
        let (mixp, xcp, fgp, bbp) = match (&self.spec, self.save.get()) {
            (Some(sp), true) => sp.slot(ly.idx, qd),
            _ => (self.mix.ptr, self.xc.ptr, self.fg.ptr, self.bb.ptr),
        };
        mark!(self, "kda start");
        self.lin(&m[0], self.h.ptr, n, mixp)?;
        mark!(self, "kda q k v product");
        self.k.conv(s, mixp, (3 * qd) as i32, sm[0].ptr, false, conv, xcp, n as i32, (3 * qd) as i32)?;
        mark!(self, "kda convolution");
        self.lin(&m[2], self.h.ptr, n, self.low.ptr)?;
        self.lin(&m[3], self.low.ptr, n, fgp)?;
        self.lin(&m[6], self.h.ptr, n, bbp)?;
        self.lin(&m[4], self.h.ptr, n, self.low.ptr)?;
        self.lin(&m[5], self.low.ptr, n, self.og.ptr)?;
        mark!(self, "kda gates (5 products)");
        self.k.delta(s, self.chunked && n > 1, xcp, fgp, sm[2].ptr, sm[1].ptr, bbp, state, self.core.ptr, n as i32, heads, c.kda.bound)?;
        mark!(self, "kda delta rule");
        self.k.delta_out(s, self.core.ptr, self.og.ptr, sm[3].ptr, false, self.att.ptr, n as u32, heads, c.eps)?;
        mark!(self, "kda gated norm");
        self.lin(&m[1], self.att.ptr, n, self.o.ptr)
    }

    /// A dense MLP on the n rows in `h` into `o`: the gate and up product, the clamped SwiGLU, the down product.
    unsafe fn dense_mlp(&self, ly: &LayerGlm, n: usize) -> Result<()> {
        let c = &self.w.cfg;
        let (gu, down) = (&ly.mats[ly.mats.len() - 2], &ly.mats[ly.mats.len() - 1]);
        self.lin(gu, self.h.ptr, n, self.mix.ptr)?;
        self.kmoe.act(self.dev.stream, self.mix.ptr, self.xc.ptr, n, c.inter, &Act::SwiGlu { limit: Some(c.experts.limit) })?;
        self.lin(down, self.xc.ptr, n, self.o.ptr)
    }

    /// The token mixer's half of layer `ly` on the n streams in `x` (positions at `pos`, int32 on the device; `start` tokens before them): the mixing weights and the collapse into `h` (normed), the mixer into `o`, the
    /// streams updated by it.
    ///
    /// # Safety
    /// Live buffers of the engine, sized for a chunk; the stream orders every step.
    pub(crate) unsafe fn mixer_half(&self, ly: &LayerGlm, n: usize, pos: u64, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let hid = c.hidden as i32;
        mark!(self, "mixer half start");
        self.k.hc_mix(s, self.x.ptr, &ly.hc[0], ly.norm_a.ptr, self.h.ptr, self.hcp[0].ptr, n as u32, hid, c.eps, c.hc.eps, c.hc.iters as i32)?;
        mark!(self, "mHC mix (mixer)");
        match ly.kind {
            GlmKind::Kda => self.delta(ly, n)?,
            GlmKind::Mla => {
                ly.mla.as_ref().ok_or_else(|| Error("a latent attention layer has no block".into()))?.forward(self, self.h.ptr, n, pos, start, self.o.ptr)?
            }
        }
        mark!(self, "o product");
        self.k.hc_post(s, self.x.ptr, self.o.ptr, self.hcp[0].ptr, n as u32, hid)?;
        mark!(self, "mHC update (mixer)");
        Ok(())
    }

    /// The feed-forward's half of layer `ly` on the n streams in `x`: as `mixer_half`, with the dense MLP or the block of experts.
    ///
    /// # Safety
    /// As `mixer_half`.
    pub(crate) unsafe fn mlp_half(&self, ly: &LayerGlm, n: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let hid = c.hidden as i32;
        mark!(self, "feed-forward half start");
        self.k.hc_mix(s, self.x.ptr, &ly.hc[1], ly.norm_f.ptr, self.h.ptr, self.hcp[1].ptr, n as u32, hid, c.eps, c.hc.eps, c.hc.iters as i32)?;
        mark!(self, "mHC mix (feed-forward)");
        match &ly.moe {
            Some(b) => b.forward(self, self.h.ptr, n, self.o.ptr)?,
            None => self.dense_mlp(ly, n)?,
        }
        mark!(self, "experts or MLP");
        self.k.hc_post(s, self.x.ptr, self.o.ptr, self.hcp[1].ptr, n as u32, hid)?;
        mark!(self, "mHC update (feed-forward)");
        Ok(())
    }

    /// The layers on the n token rows in `o` (their embeddings; positions at `pos`, int32 on the device; `start` tokens before them); where `head`, the mean of the last row's streams, the final norm, the LM head and
    /// the argmax.
    unsafe fn layers_pass(&self, n: usize, pos: u64, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        self.k.streams(s, self.o.ptr, self.x.ptr, n as u32, c.hidden as i32)?;
        for (l, slot) in self.w.layers.iter().enumerate() {
            let ly = slot.as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
            self.mixer_half(ly, n, pos, start)?;
            self.mlp_half(ly, n)?;
            if let Some(t) = &self.trace {
                t(self, l, n, start);
            }
        }
        Ok(())
    }

    /// The layers on the n token rows, then for every row the final norm, the LM head and the argmax (into the draft buffers): what a verified draft needs.
    unsafe fn forward_rows(&self, n: usize, pos: u64, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let sp = self.spec.as_ref().ok_or_else(|| Error("the engine was made without room for a draft (OptsGlm::spec)".into()))?;
        self.layers_pass(n, pos, start)?;
        let (norm_f, lm) = (self.w.norm_f.as_ref(), self.w.lm_head());
        let (Some(norm_f), Some(lm)) = (norm_f, lm) else { return err("the final norm or the LM head is not loaded") };
        self.k.hc_head(s, self.x.ptr, norm_f.ptr, sp.hl.ptr, n as u32, c.hidden as i32, c.eps)?;
        self.lin_packed(lm, sp.hl.ptr, n, sp.lg.ptr)?;
        self.kern.argmax_rows(s, sp.lg.ptr, c.vocab as i32, n as u32, sp.tk.ptr, sp.vl.ptr)
    }

    unsafe fn forward(&self, n: usize, pos: u64, start: usize, head: bool) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        self.layers_pass(n, pos, start)?;
        if head {
            let (norm_f, lm) = (self.w.norm_f.as_ref(), self.w.lm_head());
            let (Some(norm_f), Some(lm)) = (norm_f, lm) else { return err("the final norm or the LM head is not loaded") };
            let last = ((n - 1) * GM_HC * c.hidden * 2) as u64;
            self.k.hc_head(s, self.x.ptr + last, norm_f.ptr, self.hlast.ptr, 1, c.hidden as i32, c.eps)?;
            self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        }
        Ok(())
    }

    /// The decode step as a graph (the position read from `st`).
    fn capture_graph(&mut self) -> Result<()> {
        self.st.write(0, &[0i32])?;
        // one eager step first (lazy initializations out of the capture); it writes the states, which `reset` and a prompt rewrite
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            self.embed(&[0u32])?;
            self.forward(1, self.st.ptr, 0, true)?;
        }
        self.dev.sync()?;
        let g = self.dev.capture(|| {
            // SAFETY: as above; nothing runs during a capture.
            unsafe { self.forward(1, self.st.ptr, 0, true) }
        })?;
        self.graph = Some(g);
        self.reset()
    }

    /// A development check's reset of what one layer run on its own needs: the states zeroed, the engine at position 0.
    ///
    /// # Safety
    /// The stream is idle.
    pub unsafe fn reset_for_block(&self) -> Result<()> {
        self.dev.sync()?;
        self.ssm.zero()?;
        self.convst.zero()
    }

    /// The final normed hidden state (bf16 bits, [n, hidden]) of each of the `n` rows of the last prompt chunk: the mean of the four streams and the final norm, a development check's view of a prefill.
    pub fn final_hidden_rows(&self, n: usize) -> Result<Vec<u16>> {
        let c = &self.w.cfg;
        let norm_f = self.w.norm_f.as_ref().ok_or_else(|| Error("the final norm is not loaded".into()))?;
        let out = Buf::new(n * c.hidden * 2)?;
        // SAFETY: live buffers of the engine, sized for a chunk; the stream is synchronized before the read.
        unsafe { self.k.hc_head(self.dev.stream, self.x.ptr, norm_f.ptr, out.ptr, n as u32, c.hidden as i32, c.eps)? };
        self.dev.sync()?;
        let mut v = vec![0u16; n * c.hidden];
        out.read(0, &mut v)?;
        Ok(v)
    }

    /// The token chosen after the work queued (the largest logit).
    fn read_sample(&mut self) -> Result<Sample> {
        // SAFETY: the pinned buffer holds 4096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(self.host.ptr, self.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((self.host.ptr as *mut u8).add(8) as *mut _, self.val.ptr, 4, self.dev.stream)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*(self.host.ptr as *const i32), *((self.host.ptr as *const u8).add(8) as *const f32)) };
        Ok(Sample { token: token as u32, logit, logprob: None })
    }

    /// The prompt through the model after the tokens already in the sequence; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where a sequence may reach {}", self.n, tokens.len(), self.ctx));
        }
        self.pending.set(None);
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?; // (the previous chunk's kernels read the positions until then)
            self.pos.write(0, &pos)?;
            let last = at + n == tokens.len();
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, self.n, last)?;
            }
            self.n += n;
            at += n;
        }
        self.read_sample()
    }

    /// One decode step: `token` at the next position; the token that follows it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err(format!("a sequence may reach {} positions", self.ctx));
        }
        self.pending.set(None);
        let p = [self.n as i32];
        // SAFETY: pinned staging copied before the launches in stream order; live buffers.
        unsafe {
            let hp = (self.host.ptr as *mut u8).add(64);
            std::ptr::copy_nonoverlapping(p.as_ptr() as *const u8, hp, 4);
            copy_h2d_async(self.st.ptr, hp as *const _, 4, self.dev.stream)?;
            self.embed(&[token])?;
            match &self.graph {
                Some(g) => g.launch(self.dev.stream)?,
                None if self.runs_in_parts() => self.forward_segmented()?,
                None => self.forward(1, self.st.ptr, self.n, true)?,
            }
        }
        self.n += 1;
        self.read_sample()
    }

    /// Whether a decode step is run in parts: it is asked for, the model has mixture-of-experts layers and every block of them can be.
    fn runs_in_parts(&self) -> bool {
        self.segmented.get()
            && self.trace.is_none()
            && self.w.layers.iter().flatten().filter_map(|l| l.moe.as_ref()).all(|b| b.segmented())
            && self.w.layers.iter().flatten().any(|l| l.moe.is_some())
    }

    /// The graphs of the parts forgotten (the settings they were made for have changed).
    pub fn forget_segments(&self) {
        self.segs.borrow_mut().clear();
    }

    /// The stretch `key` of GPU work: launched as the graph made of it, which the first call captures by running `f` with the stream capturing.
    ///
    /// # Safety
    /// As `forward`; `f` queues the stretch alone, the same one each time.
    unsafe fn seg(&self, key: usize, f: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut segs = self.segs.borrow_mut();
        if segs.len() <= key {
            segs.resize_with(key + 1, || None);
        }
        if segs[key].is_none() {
            segs[key] = Some(self.dev.capture(f)?);
        }
        segs[key].as_ref().expect("a graph").launch(self.dev.stream)
    }

    /// The decode step with the expert cache in parts: the layers before the first block of experts and that block's front are one graph; for each block the host's part (`MoeBlock::host`), then a graph of its back, the
    /// streams' update, and the next layer's mixer and front (the last one's: the head). One wait a layer, the GPU's work in a launch each.
    unsafe fn forward_segmented(&self) -> Result<()> {
        let c = &self.w.cfg;
        let (s, hid) = (self.dev.stream, c.hidden as i32);
        let (pos, start) = (self.st.ptr, self.n);
        let layers: Vec<(usize, &LayerGlm)> = self.w.layers.iter().enumerate().filter_map(|(l, x)| x.as_ref().map(|x| (l, x))).collect();
        let first = layers.iter().position(|(_, ly)| ly.moe.is_some()).ok_or_else(|| Error("no layer of experts".into()))?;
        if self.segs.borrow().is_empty() {
            // (the host's buffers are made right before a capture)
            layers[first].1.moe.as_ref().ok_or_else(|| Error("a layer of experts has no block".into()))?.prepare(self)?;
        }
        let front = |ly: &LayerGlm| -> Result<()> {
            self.k.hc_mix(s, self.x.ptr, &ly.hc[1], ly.norm_f.ptr, self.h.ptr, self.hcp[1].ptr, 1, hid, c.eps, c.hc.eps, c.hc.iters as i32)?;
            ly.moe.as_ref().ok_or_else(|| Error("a layer of experts has no block".into()))?.front(self, self.h.ptr, 1)
        };
        self.seg(0, || {
            self.k.streams(s, self.o.ptr, self.x.ptr, 1, hid)?;
            for (_, ly) in &layers[..first] {
                self.mixer_half(ly, 1, pos, start)?;
                self.mlp_half(ly, 1)?;
            }
            let ly = layers[first].1;
            self.mixer_half(ly, 1, pos, start)?;
            front(ly)
        })?;
        for i in first..layers.len() {
            let ly = layers[i].1;
            let b = ly.moe.as_ref().ok_or_else(|| Error("a layer of experts has no block".into()))?;
            crate::tmark!(s, "segment: host part");
            b.host(self)?;
            self.seg(layers[i].0 + 1, || {
                b.back(self, self.o.ptr, 1)?;
                self.k.hc_post(s, self.x.ptr, self.o.ptr, self.hcp[1].ptr, 1, hid)?;
                match layers.get(i + 1) {
                    Some((_, nx)) => {
                        self.mixer_half(nx, 1, pos, start)?;
                        front(nx)
                    }
                    None => {
                        let (norm_f, lm) = (self.w.norm_f.as_ref(), self.w.lm_head());
                        let (Some(norm_f), Some(lm)) = (norm_f, lm) else { return err("the final norm or the LM head is not loaded") };
                        self.k.hc_head(s, self.x.ptr, norm_f.ptr, self.hlast.ptr, 1, hid, c.eps)?;
                        self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
                        self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)
                    }
                }
            })?;
        }
        Ok(())
    }

    /// A draft verified in one pass: `tokens` (the last token the model has not seen, then the guesses that follow it) at the next positions; the token the model picks after each of them, in order. The delta rules' states
    /// go forward over all the rows (the recurrence, as a decode step runs it) after being kept, and the sequence is not advanced: `commit` says how many of the rows stay.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Sample>> {
        let n = tokens.len();
        if self.spec.is_none() {
            return err("the engine was made without room for a draft (OptsGlm::spec)");
        }
        if n == 0 || n > SPEC_ROWS.min(self.chunk) {
            return err(format!("a draft of {n} rows, where 1 to {} are verified at once", SPEC_ROWS.min(self.chunk)));
        }
        if self.n + n + 1 > self.ctx {
            return err(format!("{} tokens and {n} more, where a sequence may reach {}", self.n, self.ctx));
        }
        let s = self.dev.stream;
        if let Some(sp) = &self.spec {
            // SAFETY: the engine's own buffers, of equal sizes; the stream orders the copies before the pass.
            unsafe {
                copy_d2d_async(sp.ssm0.ptr, self.ssm.ptr, self.ssm.bytes, s)?;
                copy_d2d_async(sp.conv0.ptr, self.convst.ptr, self.convst.bytes, s)?;
            }
        }
        let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        self.save.set(true);
        let chunked = std::mem::replace(&mut self.chunked, false);
        // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
        let r = unsafe { self.embed(tokens).and_then(|()| self.forward_rows(n, self.pos.ptr, self.n)) };
        self.chunked = chunked;
        self.save.set(false);
        r?;
        self.pending.set(Some(n));
        self.dev.sync()?;
        let sp = self.spec.as_ref().ok_or_else(|| Error("no draft buffers".into()))?;
        let (mut tk, mut vl) = (vec![0i32; n], vec![0f32; n]);
        sp.tk.read(0, &mut tk)?;
        sp.vl.read(0, &mut vl)?;
        Ok(tk.iter().zip(&vl).map(|(&t, &v)| Sample { token: t as u32, logit: v, logprob: None }).collect())
    }

    /// Of the `rows` of the last `verify`, the first `kept` stay in the sequence: the Kimi delta layers go back to where they were after the `kept`th (their states from before the pass, the recurrence run again over the kept
    /// rows' saved inputs; the convolutions' states the last three of the old ones and those rows' products).
    pub fn commit(&mut self, rows: usize, kept: usize) -> Result<()> {
        if kept == 0 || kept > rows || rows > SPEC_ROWS {
            return err(format!("{kept} of {rows} rows kept"));
        }
        if self.pending.get() != Some(rows) {
            return err(format!("{kept} of {rows} rows kept, where the pass waiting to be settled is {:?}", self.pending.get()));
        }
        self.pending.set(None);
        if kept < rows {
            let c = self.w.cfg.clone();
            let sp = self.spec.as_ref().ok_or_else(|| Error("the engine was made without room for a draft (OptsGlm::spec)".into()))?;
            let (heads, qd, s) = (c.kda.heads as i32, c.qkv(), self.dev.stream);
            let (state_bytes, conv_bytes, row) = (c.kda.heads * GM_D * GM_D * 4, 3 * 3 * qd * 2, 3 * qd * 2);
            for ly in self.w.layers.iter().flatten().filter(|l| l.kind == GlmKind::Kda) {
                let (mixp, xcp, fgp, bbp) = sp.slot(ly.idx, qd);
                let state = self.ssm.ptr + (ly.idx * state_bytes) as u64;
                let (conv, old) = (self.convst.ptr + (ly.idx * conv_bytes) as u64, sp.conv0.ptr + (ly.idx * conv_bytes) as u64);
                // SAFETY: the engine's own buffers; the stream orders the copies and the kernel.
                unsafe {
                    copy_d2d_async(state, sp.ssm0.ptr + (ly.idx * state_bytes) as u64, state_bytes, s)?;
                    self.k.delta(s, false, xcp, fgp, ly.small[2].ptr, ly.small[1].ptr, bbp, state, self.core.ptr, kept as i32, heads, c.kda.bound)?;
                    for r in 0..3 {
                        let q = kept + r;
                        let src = if q < 3 { old + (q * row) as u64 } else { mixp + ((q - 3) * row) as u64 };
                        copy_d2d_async(conv + (r * row) as u64, src, row, s)?;
                    }
                }
            }
            self.dev.sync()?;
        }
        self.n += kept;
        Ok(())
    }

    /// Greedy tokens as `generate` makes them, `max_new` of them, by rounds of drafts: `draft` is given the tokens so far (the prompt and the replies, the last of them not yet seen by the model) and returns the guesses
    /// that follow; the last token and the guesses are verified in one pass, the guesses that are the model's own picks stay, with the pick after the last of them. Whatever the drafts are, the tokens are `generate`'s.
    pub fn generate_spec(&mut self, prompt: &[u32], max_new: usize, draft: &mut dyn FnMut(&[u32]) -> Vec<u32>) -> Result<(Vec<Sample>, SpecStats)> {
        self.reset()?;
        let mut out = vec![self.prefill(prompt)?];
        let mut all: Vec<u32> = prompt.to_vec();
        all.push(out[0].token);
        let mut stats = SpecStats::default();
        while out.len() < max_new && self.n + 2 <= self.ctx {
            // (a pass of r rows needs n + r + 1 positions: at the edge of the context the tokens made so far are returned)
            let mut guesses = draft(&all);
            guesses.truncate((SPEC_ROWS - 1).min(max_new - out.len()).min(self.ctx - self.n - 2));
            let mut rows = vec![all[all.len() - 1]];
            rows.extend_from_slice(&guesses);
            let picks = self.verify(&rows)?;
            let right = guesses.iter().zip(&picks).take_while(|(g, p)| **g == p.token).count();
            self.commit(rows.len(), right + 1)?;
            stats.passes += 1;
            stats.drafted += guesses.len();
            stats.accepted += right;
            for p in &picks[..=right] {
                if out.len() < max_new {
                    out.push(*p);
                    all.push(p.token);
                }
            }
        }
        Ok((out, stats))
    }

    /// `prefill`, with the MTP head run on every row of the prompt (the final normed hidden state of a position and the token after it; the last row waits for the token the model picks): the head's cache then holds the
    /// prompt. Returns the token the model picks and the hidden state's address for the first draft (the head's scratch).
    fn prefill_with_head(&mut self, head: &crate::mtp_glm::MtpHead, tokens: &[u32]) -> Result<(Sample, u32, u64)> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where a sequence may reach {}", self.n, tokens.len(), self.ctx));
        }
        let (hid, s) = (self.w.cfg.hidden, self.dev.stream);
        let hall = self.hall.as_ref().ok_or_else(|| Error("the engine has no MTP head".into()))?.ptr;
        let mut at = 0;
        let mut last_rows = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?;
            self.pos.write(0, &pos)?;
            let last = at + n == tokens.len();
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, self.n, last)?;
                let norm_f = self.w.norm_f.as_ref().ok_or_else(|| Error("the final norm is not loaded".into()))?;
                self.k.hc_head(s, self.x.ptr, norm_f.ptr, hall, n as u32, hid as i32, self.w.cfg.eps)?;
                let m = if last { n - 1 } else { n };
                if m > 0 {
                    head.forward(self, hall, &tokens[at + 1..at + 1 + m], self.n)?;
                }
            }
            self.n += n;
            at += n;
            last_rows = n;
        }
        let first = self.read_sample()?;
        let last_row = hall + ((last_rows - 1) * hid * 2) as u64;
        // SAFETY: the last row of the last chunk's hidden states is in `hall`.
        let hout = unsafe { head.forward(self, last_row, &[first.token], self.n - 1)? };
        // SAFETY: the head's hidden state of that row is in its scratch.
        let d1 = unsafe { head.picks(self, hout, 1)? }[0].0;
        Ok((first, d1, hout))
    }

    /// The draft that follows a first guess `d1`, which came from the head's hidden state at `hrow` for the row at `pos`: `k` guesses in all, each from the head run on the one before.
    ///
    /// # Safety
    /// `hrow` the address of the head's last hidden state.
    unsafe fn draft_chain(&self, head: &crate::mtp_glm::MtpHead, hrow: u64, d1: u32, pos: usize, k: usize) -> Result<Vec<u32>> {
        let mut out = vec![d1];
        let (mut h, mut tok) = (hrow, d1);
        for step in 1..k {
            let hout = head.forward(self, h, &[tok], pos + step)?;
            tok = head.picks(self, hout, 1)?[0].0;
            out.push(tok);
            h = hout;
        }
        Ok(out)
    }

    /// Greedy tokens as `generate` makes them, `max_new` of them, drafted by the MTP head: `k` guesses a pass (at most `SPEC_ROWS - 1`), verified as `generate_spec` does; the head is run on the rows the pass keeps (their
    /// real hidden states, and the token after each) to fill its cache and draft the next guesses. Whatever the head guesses, the tokens are `generate`'s.
    pub fn generate_mtp(&mut self, prompt: &[u32], max_new: usize, k: usize) -> Result<(Vec<Sample>, SpecStats)> {
        let head = self.mtp.take().ok_or_else(|| Error("the engine has no MTP head (OptsGlm::mtp)".into()))?;
        let r = self.generate_mtp_with(&head, prompt, max_new, k);
        self.mtp = Some(head);
        r
    }

    fn generate_mtp_with(&mut self, head: &crate::mtp_glm::MtpHead, prompt: &[u32], max_new: usize, k: usize) -> Result<(Vec<Sample>, SpecStats)> {
        self.reset()?;
        let k = k.clamp(1, SPEC_ROWS - 1);
        let (first, d1, mut hrow) = self.prefill_with_head(head, prompt)?;
        let mut out = vec![first];
        let mut drafts = vec![d1];
        let mut pos = self.n - 1;
        // SAFETY: `hrow` is the head's last hidden state.
        unsafe {
            // (no more guesses than tokens still to make, nor than the context has room for)
            drafts = self.draft_chain(head, hrow, drafts[0], pos, k.min(max_new.saturating_sub(1)).min(self.ctx.saturating_sub(self.n + 2)).max(1))?;
        }
        let mut stats = SpecStats::default();
        let hid = self.w.cfg.hidden;
        while out.len() < max_new && self.n + 2 <= self.ctx {
            let mut guesses = std::mem::take(&mut drafts);
            guesses.truncate((max_new - out.len()).min(self.ctx - self.n - 2));
            let mut rows = vec![out[out.len() - 1].token];
            rows.extend_from_slice(&guesses);
            let n0 = self.n;
            let picks = self.verify(&rows)?;
            let right = guesses.iter().zip(&picks).take_while(|(g, p)| **g == p.token).count();
            self.commit(rows.len(), right + 1)?;
            stats.passes += 1;
            stats.drafted += guesses.len();
            stats.accepted += right;
            for p in &picks[..=right] {
                if out.len() < max_new {
                    out.push(*p);
                }
            }
            if out.len() >= max_new {
                break;
            }
            // the head on the rows the pass kept: the hidden state of each position and the token after it (the pick, for the last)
            let mut toks: Vec<u32> = rows[1..=right].to_vec();
            toks.push(picks[right].token);
            let sp_hl = self.spec.as_ref().ok_or_else(|| Error("no draft buffers".into()))?.hl.ptr;
            // SAFETY: `hl` holds the pass's rows of hidden states; the head's results are read before they change.
            unsafe {
                let hout = head.forward(self, sp_hl, &toks, n0)?;
                hrow = hout + (right * hid * 2) as u64;
                let d1 = head.picks(self, hrow, 1)?[0].0;
                pos = n0 + right;
                // (no more guesses than tokens still to make, nor than the context has room for)
                drafts = self.draft_chain(head, hrow, d1, pos, k.min(max_new - out.len()).min(self.ctx.saturating_sub(self.n + 2)).max(1))?;
            }
        }
        let _ = pos;
        Ok((out, stats))
    }

    /// Greedy tokens: the prompt, then up to `max_new` tokens (stopping at an end-of-sequence token unless `ignore_eos`).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, ignore_eos: bool) -> Result<Vec<Sample>> {
        self.reset()?;
        let mut out = Vec::with_capacity(max_new);
        let mut s = self.prefill(prompt)?;
        out.push(s);
        while out.len() < max_new {
            if !ignore_eos && self.w.cfg.eos.contains(&(s.token as i64)) {
                break;
            }
            s = self.step(s.token)?;
            out.push(s);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = include_str!("../kernels/glm.cu");

    /// The stand-ins for the latent attention and the mixture-of-experts layers say what they are; the hooks around them have the shapes a plug-in needs.
    #[test]
    fn the_stand_ins_say_what_they_are() {
        let b: Box<dyn MoeBlock> = Box::new(NoMoe);
        assert!(!b.graphable() && b.bytes() == 0 && b.pack_shapes().is_empty());
        let m: Box<dyn MlaBlock> = Box::new(NoMla);
        assert!(!m.graphable() && m.bytes() == 0 && m.pack_shapes().is_empty());
        let moe: MoeFactory = &|_, _, _, _| Ok(Box::new(NoMoe));
        let mla: MlaFactory = &|_, _, _, _| Ok(Box::new(NoMla));
        let _ = (moe, mla);
    }

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
        // the latent attention's: the limits the launches hold to and the rows a block takes are the kernels' defines
        assert_eq!(
            (define("GM_NOPE_MAX"), define("GM_KSEL_MAX"), define("GM_IDX_W"), define("GM_RB"), define("GM_VB")),
            (Some(256), Some(1024), Some(8), Some(8), Some(4))
        );
        for name in [
            "glm_kda_rec",
            "glm_kda_chunk",
            "glm_kda_out",
            "glm_kda_out_w32",
            "glm_rms2",
            "glm_layernorm",
            "glm_scatter",
            "glm_absorb",
            "glm_kp_keys",
            "glm_idx_score",
            "glm_select",
            "glm_mla_attn",
            "glm_expand_v",
        ] {
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

    /// The random tiny model `tools/glm_ref.py tiny` builds (its config is the fixture the script's selftest holds it to) is a model this engine runs: the real model's pattern in eight layers (three of Kimi delta attention with a
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
