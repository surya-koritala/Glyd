//! Qwen3.5's kernels (kernels/qwen35.cu) with typed launches: the gated delta rule, its convolution and gated norm, the zero-centered norms (`Kern35`), which the executor's
//! Qwen3.5 layout and Condor run.

use crate::dev::Dev;
use crate::ffi::{Result, Stream};
use crate::kern::norm_threads;
use crate::launch;

/// Bytes of dynamic shared memory gdn_chunk takes (kernels/qwen35.cu).
const GDN_CHUNK_SMEM: u32 = (2 * 64 * 65 + 64 * 64 + 64 * 64 + 128 * 64 + 2 * 64) * 4;
/// Qwen3.5's kernels with typed launches.
pub struct Kern35 {
    rmsnorm: crate::kt::rmsnorm1p,
    rmsnorm_add: crate::kt::rmsnorm1p_add,
    conv: crate::kt::conv_silu,
    conv_rows: crate::kt::conv_silu_rows,
    rec: crate::kt::gdn_recurrent,
    rec_rows: crate::kt::gdn_recurrent_rows,
    chunk: crate::kt::gdn_chunk,
    gnorm: crate::kt::gated_norm,
    small: crate::kt::small_linear,
}

impl Kern35 {
    pub fn load(dev: &Dev) -> Result<Kern35> {
        let chunk = dev.typed::<crate::kt::gdn_chunk>()?;
        chunk.allow_smem(GDN_CHUNK_SMEM)?;
        Ok(Kern35 {
            rmsnorm: dev.typed()?,
            rmsnorm_add: dev.typed()?,
            conv: dev.typed()?,
            conv_rows: dev.typed()?,
            rec: dev.typed()?,
            rec_rows: dev.typed()?,
            chunk,
            gnorm: dev.typed()?,
            small: dev.typed()?,
        })
    }

    /// out [rows, k] = rmsnorm(x) * (1 + w).
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/qwen35.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm(&self, s: Stream, x: u64, w: u64, out: u64, rows: u32, k: i32, eps: f32) -> Result<()> {
        launch!(self.rmsnorm, [rows, 1, 1], [norm_threads(k)?, 1, 1], s; x, w, out, k, eps)
    }

    /// x += y, then out = rmsnorm(x) * (1 + w).
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_add(&self, s: Stream, x: u64, y: u64, w: u64, out: u64, rows: u32, k: i32, eps: f32) -> Result<()> {
        launch!(self.rmsnorm_add, [rows, 1, 1], [norm_threads(k)?, 1, 1], s; x, y, w, out, k, eps)
    }

    /// The causal convolution and SiLU of n rows (the first c channels of rows xs apart), the state read and written.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, st: u64, y: u64, n: i32, c: i32) -> Result<()> {
        launch!(self.conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, st, y, n, c)
    }

    /// `conv` of `rows` sequences' rows, a token each: row r's state at the address `sts` holds at entry r ss (0: a padding row, its output zeroed).
    ///
    /// # Safety
    /// As `rmsnorm`; `sts` holds `rows` entries `ss` apart, each 0 or a live state of the convolution's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_rows(&self, s: Stream, x: u64, xs: i32, w: u64, sts: u64, ss: i32, y: u64, rows: u32, c: i32) -> Result<()> {
        launch!(self.conv_rows, [(c as u32).div_ceil(256), rows, 1], [256, 1, 1], s; x, xs, w, sts, ss, y, c)
    }

    /// The gated delta rule of `rows` sequences' rows, a token each (`delta`'s recurrence): row r's state at the address `sts` holds at entry r ss (0: a padding row, its
    /// output zeroed).
    ///
    /// # Safety
    /// As `conv_rows`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_rows(
        &self,
        s: Stream,
        xc: u64,
        c: i32,
        ba: u64,
        sba: i32,
        a_log: u64,
        dt_bias: u64,
        sts: u64,
        ss: i32,
        core: u64,
        rows: u32,
        nk: i32,
        nv: i32,
    ) -> Result<()> {
        launch!(self.rec_rows, [nv as u32, rows, 1], [128, 1, 1], s; xc, c, ba, sba, a_log, dt_bias, sts, ss, core, nk, nv)
    }

    /// The gated delta rule over n tokens: one after another, or in chunks of 64 (`chunked`).
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta(
        &self,
        s: Stream,
        chunked: bool,
        xc: u64,
        c: i32,
        ba: u64,
        sba: i32,
        a_log: u64,
        dt_bias: u64,
        state: u64,
        core: u64,
        n: i32,
        nk: i32,
        nv: i32,
    ) -> Result<()> {
        if chunked {
            launch!(self.chunk, [nv as u32, 2, 1], [256, 1, 1], smem GDN_CHUNK_SMEM, s; xc, c, ba, sba, a_log, dt_bias, state, core, n, nk, nv)
        } else {
            launch!(self.rec, [nv as u32, 1, 1], [128, 1, 1], s; xc, c, ba, sba, a_log, dt_bias, state, core, n, nk, nv)
        }
    }

    /// The gated norm of n rows of nv value heads, in place.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gated_norm(&self, s: Stream, core: u64, z: u64, zs: i32, w: u64, n: u32, nv: i32, eps: f32) -> Result<()> {
        launch!(self.gnorm, [n, nv as u32, 1], [128, 1, 1], s; core, z, zs, w, nv, eps)
    }

    /// y [m, rows] = x [m, cols] W^T for a bf16 W the library does not pack.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn small_linear(&self, s: Stream, x: u64, xs: i32, w: u64, y: u64, ys: i32, m: u32, rows: i32, cols: i32) -> Result<()> {
        launch!(self.small, [m, (rows.max(1) as u32).div_ceil(8), 1], [256, 1, 1], s; x, xs, w, y, ys, rows, cols)
    }
}
