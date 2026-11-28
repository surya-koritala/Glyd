//! GLM-5.3-Flash's and Kimi-K3's kernels (kernels/glm.cu, kernels/mla.cu) with typed launches: Kimi delta attention, the embedding into the residual streams, the indexer's
//! score and pick and the small dense product (`KernGlm`), and the streams' mixing weights a launch takes (`Hc`), which the executor and Ibis run. The latent attention's
//! own launches are the layer kind's (src/dynamic/mla.rs), the streams' mixing the streams kind's (src/dynamic/residual.rs).

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Result, Stream};
use crate::launch;

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
    /// (the convolution and the recurrence of a step's rows that decode, each through its own sequence's state)
    conv_rows: crate::kt::conv_silu_rows,
    conv_w32_rows: crate::kt::conv_silu_w32_rows,
    rec_rows: crate::kt::glm_kda_rec_rows,
    chunk: crate::kt::glm_kda_chunk,
    out: crate::kt::glm_kda_out,
    out_w32: crate::kt::glm_kda_out_w32,
    small: crate::kt::small_linear,
    // the indexer's score and pick (kernels/mla.cu), which Ibis's engine runs on its own caches; the latent attention itself is src/dynamic/mla.rs's
    idx_score: crate::kt::idx_score,
    select: crate::kt::idx_pick,
}

/// The dynamic shared memory `mla_attn` (kernels/mla.cu) may take on `dev`: the GPU's most a block, less a KiB for the kernel's own (its block sums' 256 bytes).
pub fn mla_smem(dev: &Dev) -> Result<u32> {
    Ok(dev.smem_optin()?.saturating_sub(1 << 10))
}

/// The most positions a pass of the latent attention (scores over `ks` channels) holds where a block has `smem` bytes of shared memory: `mla_attn`'s
/// tile, a multiple of 16.
pub fn mla_positions(smem: u32, ks: i32) -> usize {
    ((smem as usize / 4).saturating_sub(ks as usize) / 2) / 16 * 16
}

impl KernGlm {
    pub fn load(dev: &Dev) -> Result<KernGlm> {
        let chunk = dev.typed::<crate::kt::glm_kda_chunk>()?;
        chunk.allow_smem(GM_CHUNK_SMEM)?;
        Ok(KernGlm {
            hc: crate::dynamic::residual::KernRes::load(dev)?,
            conv: dev.typed()?,
            conv_w32: dev.typed()?,
            rec: dev.typed()?,
            conv_rows: dev.typed()?,
            conv_w32_rows: dev.typed()?,
            rec_rows: dev.typed()?,
            chunk,
            out: dev.typed()?,
            out_w32: dev.typed()?,
            small: dev.typed()?,
            idx_score: dev.typed()?,
            select: dev.typed()?,
        })
    }

    /// x [n, 4, d] <- the embedding rows e [n, d] four times.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/glm.cu and kernels/residual.cu).
    pub unsafe fn streams(&self, s: Stream, e: u64, x: u64, n: u32, d: i32) -> Result<()> {
        self.hc.streams(s, e, x, n, d, 4)
    }

    /// The causal convolution and SiLU of n rows (the first c channels of rows xs apart), the bf16 state [3, c] read and written (qwen35.cu's `conv_silu`; `w32`: the weights fp32, `conv_silu_w32`).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, w32: bool, st: u64, y: u64, n: i32, c: i32) -> Result<()> {
        launch!(if w32 { self.conv_w32.sig() } else { self.conv.sig() }, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, st, y, n, c)
    }

    /// `conv` of a step's `rows` rows that decode, a token of a sequence each: row r's state at the address `sts` holds at entry r ss (0: a padding row, its output zeroed).
    ///
    /// # Safety
    /// As `streams`; `sts` holds `rows` entries `ss` apart, each 0 or a live state of the convolution's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_rows(&self, s: Stream, x: u64, xs: i32, w: u64, w32: bool, (sts, ss): (u64, usize), y: u64, rows: u32, c: i32) -> Result<()> {
        let k = if w32 { self.conv_w32_rows.sig() } else { self.conv_rows.sig() };
        launch!(k, [(c as u32).div_ceil(256), rows, 1], [256, 1, 1], s; x, xs, w, sts, ss as i32, y, c)
    }

    /// The delta rule's recurrence (`delta`'s, a token) of a step's `rows` rows that decode, a token of a sequence each: row r's state at the address `sts` holds at entry
    /// r ss (0: a padding row, its output zeroed), its rows of xc, fg, bb and core r rows on.
    ///
    /// # Safety
    /// As `conv_rows`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_rows(
        &self,
        s: Stream,
        xc: u64,
        fg: u64,
        dtb: u64,
        alog: u64,
        bb: u64,
        (sts, ss): (u64, usize),
        core: u64,
        rows: u32,
        heads: i32,
        bound: f32,
    ) -> Result<()> {
        launch!(self.rec_rows, [heads as u32, rows, 1], [128, 1, 1], s; xc, fg, dtb, alog, bb, sts, ss as i32, core, heads, bound)
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

    /// scores [n, pmax] fp32 for the rows whose pools are more than `ksel`: the indexer's heads (`ih` of `d` channels, queries iq with rows `iqs` apart, head weights wts with rows `wsr` apart, bf16) against the pool keys pk.
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
        launch!(self.idx_score, [(pmax as u32).div_ceil(8), n, 1], [256, 1, 1], smem smem, s; iq, iqs, wts, wsr, 0i32, pk, d, 1i32, 0i32, 0u64, 0i64, pos, scores, pmax, ih, d, kp, ksel, scale, wscale)
    }

    /// The tokens each of n rows attends to: sel [n, tmax] int32 (the first cnt[i] of a row), from the scores.
    ///
    /// # Safety
    /// As `streams`; ksel at most 1,024: this entry's own bound; the kernel here, `idx_pick`, takes any count.
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
