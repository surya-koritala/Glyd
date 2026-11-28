//! The Mamba-2 layer kind's kernels (kernels/nemotron.cu) with typed launches: the scan, its convolution and gated norm (`KernMamba2`) and the chunked scan's
//! buffers (`SsdWs`), which the executor's `Op::MambaConv`, `Op::MambaScan` and `Op::MambaNorm` run for any family whose description has the kind. The kernels
//! compile in the kind's limits they run: heads of `MAMBA2_HEAD` channels, a state of `MAMBA2_STATE` a channel, prompts scanned in chunks of `MAMBA2_CHUNK`
//! tokens, a convolution of 4 taps; a description of other sizes is refused where it is lowered (`lower::nemotron`).

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Result, Stream};
use crate::launch;

/// The kernels' limits (kernels/nemotron.cu compiles them in): a Mamba-2 head's channels, the state a channel has, the tokens of a chunk of the chunked scan.
pub const MAMBA2_HEAD: usize = 64;
pub const MAMBA2_STATE: usize = 128;
pub const MAMBA2_CHUNK: usize = 128;

/// The Mamba-2 chunked scan's buffers for up to `rows` tokens a call: dt and its running sum, C B^T of every chunk and group, every chunk's own state.
pub struct SsdWs {
    pub rows: usize,
    dtv: Buf,
    acum: Buf,
    g: Buf,
    st: Buf,
}

impl SsdWs {
    /// For `heads` heads in `groups` groups.
    pub fn sized(heads: usize, groups: usize, rows: usize) -> Result<SsdWs> {
        let nch = rows.div_ceil(MAMBA2_CHUNK).max(1);
        Ok(SsdWs {
            rows,
            dtv: Buf::new(nch * MAMBA2_CHUNK * heads * 4)?,
            acum: Buf::new(nch * MAMBA2_CHUNK * heads * 4)?,
            g: Buf::new(nch * groups * MAMBA2_CHUNK * MAMBA2_CHUNK * 4)?,
            st: Buf::new(nch * heads * MAMBA2_HEAD * MAMBA2_STATE * 4)?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.dtv.bytes + self.acum.bytes + self.g.bytes + self.st.bytes
    }
}

/// The Mamba-2 kind's kernels with typed launches.
pub struct KernMamba2 {
    conv: crate::kt::nh_conv,
    conv_rows: crate::kt::nh_conv_rows,
    ssm: crate::kt::nh_ssm,
    ssm_rows: crate::kt::nh_ssm_rows,
    cum: crate::kt::nh_ssd_cum,
    gmat: crate::kt::nh_ssd_g,
    states: crate::kt::nh_ssd_states,
    pass: crate::kt::nh_ssd_pass,
    scan_y: crate::kt::nh_ssd_y,
    gnorm: crate::kt::nh_gated_norm,
}

impl KernMamba2 {
    pub fn load(dev: &Dev) -> Result<KernMamba2> {
        Ok(KernMamba2 {
            conv: dev.typed()?,
            conv_rows: dev.typed()?,
            ssm: dev.typed()?,
            ssm_rows: dev.typed()?,
            cum: dev.typed()?,
            gmat: dev.typed()?,
            states: dev.typed()?,
            pass: dev.typed()?,
            scan_y: dev.typed()?,
            gnorm: dev.typed()?,
        })
    }

    /// The causal convolution (kernel 4) with its bias and SiLU over the first `c` channels of `n` rows `xs` apart; the bf16 state [3, c] read and left holding the last three inputs; y [n, c].
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/nemotron.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, bias: u64, st: u64, y: u64, n: i32, c: i32) -> Result<()> {
        launch!(self.conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, bias, st, y, n, c)
    }

    /// `conv` of `rows` sequences' rows, a token each: row r's state at the address `sts` holds at entry r ss (0: a padding row, its output zeroed).
    ///
    /// # Safety
    /// As `conv`; `sts` holds `rows` entries `ss` apart, each 0 or a live state of the convolution's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_rows(&self, s: Stream, x: u64, xs: i32, w: u64, bias: u64, sts: u64, ss: i32, y: u64, rows: u32, c: i32) -> Result<()> {
        launch!(self.conv_rows, [(c as u32).div_ceil(256), rows, 1], [256, 1, 1], s; x, xs, w, bias, sts, ss, y, c)
    }

    /// `ssm` of `rows` sequences' rows, a token each: row r's state at the address `sts` holds at entry r ss (0: a padding row, its output zeroed).
    ///
    /// # Safety
    /// As `conv_rows`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ssm_rows(
        &self,
        s: Stream,
        xc: u64,
        cs: i32,
        dtr: u64,
        dts: i32,
        dt_bias: u64,
        a_log: u64,
        d: u64,
        sts: u64,
        ss: i32,
        y: u64,
        rows: u32,
        nh: i32,
        ng: i32,
        dt_min: f32,
    ) -> Result<()> {
        launch!(self.ssm_rows, [nh as u32, (MAMBA2_HEAD / 8) as u32, rows], [256, 1, 1], s; xc, cs, dtr, dts, dt_bias, a_log, d, sts, ss, y, nh, ng, dt_min)
    }

    /// The selective state update of n rows, one token after another (a decode step: n = 1): xc [n, cs] the convolution's output, dtr the dt columns of the in_proj's output (rows `dts` apart),
    /// state fp32 [nh, 64, 128] read and written back, y [n, nh 64].
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ssm(
        &self,
        s: Stream,
        xc: u64,
        cs: i32,
        dtr: u64,
        dts: i32,
        dt_bias: u64,
        a_log: u64,
        d: u64,
        state: u64,
        y: u64,
        n: i32,
        nh: i32,
        ng: i32,
        dt_min: f32,
    ) -> Result<()> {
        launch!(self.ssm, [nh as u32, (MAMBA2_HEAD / 8) as u32, 1], [256, 1, 1], s; xc, cs, dtr, dts, dt_bias, a_log, d, state, y, n, nh, ng, dt_min)
    }

    /// The same for a prompt, in chunks of MAMBA2_CHUNK tokens (state-space duality); `ws` holds at least n rows.
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ssd(
        &self,
        s: Stream,
        ws: &SsdWs,
        xc: u64,
        cs: i32,
        dtr: u64,
        dts: i32,
        dt_bias: u64,
        a_log: u64,
        d: u64,
        state: u64,
        y: u64,
        n: i32,
        nh: i32,
        ng: i32,
        dt_min: f32,
    ) -> Result<()> {
        if n as usize > ws.rows {
            return err(format!("{n} rows in a scan that has buffers for {}", ws.rows));
        }
        let nch = (n as u32).div_ceil(MAMBA2_CHUNK as u32);
        launch!(self.cum, [nch, (nh as u32).div_ceil(128), 1], [128, 1, 1], s; dtr, dts, dt_bias, a_log, ws.dtv.ptr, ws.acum.ptr, n, nh, dt_min)?;
        launch!(self.gmat, [nch, ng as u32, 1], [256, 1, 1], s; xc, cs, ws.g.ptr, n, nh, ng)?;
        launch!(self.states, [nch, nh as u32, 1], [256, 1, 1], s; xc, cs, ws.dtv.ptr, ws.acum.ptr, ws.st.ptr, n, nh, ng)?;
        launch!(self.pass, [((nh as usize * MAMBA2_HEAD * MAMBA2_STATE) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; ws.acum.ptr, ws.st.ptr, state, n, nh, nch as i32)?;
        launch!(self.scan_y, [nch, nh as u32, 2], [256, 1, 1], s; xc, cs, ws.dtv.ptr, ws.acum.ptr, ws.g.ptr, ws.st.ptr, d, y, n, nh, ng)
    }

    /// The gated norm of n rows, in place on y [n, di]: z at z[t * zs ..], the weight bf16 [di], ng groups of di / ng channels (at most 1024).
    ///
    /// # Safety
    /// As `conv`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gated_norm(&self, s: Stream, y: u64, z: u64, zs: i32, w: u64, n: u32, di: i32, ng: i32, eps: f32) -> Result<()> {
        if ng == 0 || di % ng != 0 || di / ng > 1024 {
            return err(format!("a gated norm of {di} channels in {ng} groups: not supported"));
        }
        launch!(self.gnorm, [n, ng as u32, 1], [256, 1, 1], s; y, z, zs, w, di, ng, eps)
    }
}
