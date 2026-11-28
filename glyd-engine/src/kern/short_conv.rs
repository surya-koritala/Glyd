//! The gated short convolution kind's kernels (kernels/short_conv.cu) with typed launches, which the executor's `Op::ShortConv` runs for any family whose
//! description has the kind (`kinds::ShortConv`; LFM's conv layers). The kernels compile in the most taps a channel they take (`SHORT_CONV_TAPS`); a description
//! of more is refused where it is lowered (`lower::dense`).

use crate::dev::Dev;
use crate::ffi::{Result, Stream};
use crate::launch;

/// The kernels' limit (kernels/short_conv.cu compiles it in, SC_TAPS): the most taps a channel.
pub const SHORT_CONV_TAPS: usize = 16;

/// The short convolution kind's kernels with typed launches.
pub struct KernShortConv {
    conv: crate::kt::sc_conv,
    conv_rows: crate::kt::sc_conv_rows,
}

impl KernShortConv {
    pub fn load(dev: &Dev) -> Result<KernShortConv> {
        Ok(KernShortConv { conv: dev.typed()?, conv_rows: dev.typed()? })
    }

    /// The gated convolution of `n` rows of one sequence, one after another: x [n, xs] the input product's rows (B, C and x, `c` columns each), w bf16 [c, taps],
    /// the fp32 state [c, taps - 1] read and left holding the last `taps` - 1 values of u = B x; y [n, c] = C conv(u).
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/short_conv.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, st: u64, y: u64, n: i32, c: i32, taps: i32) -> Result<()> {
        launch!(self.conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, st, y, n, c, taps)
    }

    /// `conv` of `rows` sequences' rows, a token each: row r's state at the address `sts` holds at entry r ss (0: a padding row, its output zeroed).
    ///
    /// # Safety
    /// As `conv`; `sts` holds `rows` entries `ss` apart, each 0 or a live state of the convolution's.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_rows(&self, s: Stream, x: u64, xs: i32, w: u64, sts: u64, ss: i32, y: u64, rows: u32, c: i32, taps: i32) -> Result<()> {
        launch!(self.conv_rows, [(c as u32).div_ceil(256), rows, 1], [256, 1, 1], s; x, xs, w, sts, ss, y, c, taps)
    }
}
