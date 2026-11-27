//! The residual rules' kernels (kernels/residual.cu), every family's: the residual streams (kind `Residual::Streams`, mHC), the depth mix (`Residual::DepthMix`), the
//! gated residual (`Residual::Gated`), the plain add, and the RMS norm whose weight scales its input. The streams' arithmetic is the kind's `StreamsP::rounding`: `Once`,
//! the release's fp32 mixing projection and update, each result rounded once (DeepSeek-V4); `Twice`, transformers' bf16 tensors between operations, the collapse's norm
//! rounding the normed value before its weight (GLM-5.3-Flash).

use crate::dev::Dev;
use crate::ffi::{err, Result, Stream};
use crate::launch;
use crate::runtime::kinds::Rounding;

/// Rows of a mixing projection (kernels/residual.cu HC_NP: the collapse weights, the spreading weights, the 4 x 4 mixing matrix) and the streams (HC_N).
pub const HC_NP: usize = 24;
pub const HC_N: usize = 4;

/// The columns of a row of streams a block of the split projection takes (kernels/residual.cu HC_CPB_F32, HC_CPB_BF16).
fn cpb(round: Rounding) -> Result<usize> {
    match round {
        Rounding::Once => Ok(256),
        Rounding::Twice => Ok(512),
        Rounding::F32 => err("residual streams with every value in f32: the kernels round once (the release's) or between operations (transformers' bf16)"),
    }
}

/// The partial sums the split projection writes for `rows` rows of streams `d` wide (fp32 words): every row's, whatever its rounding.
pub fn part_words(round: Rounding, rows: usize, d: usize) -> Result<usize> {
    Ok(rows * (HC_N * d / cpb(round)?).max(1) * (HC_NP + 1))
}

/// Whether a block's mixing weights of streams `d` wide are the split projection's (`hc_proj_*` then `hc_fin`), else `hc_mix`'s (a block a row): by the
/// layer's shape and rounding alone, never by the rows of the call, so that a row's sums are those of its own pass whatever shares it (the two kernels add
/// a row's products in different orders: a prompt of 7 rows alone took one, the same rows in a pass of 15 the other, and the logits came apart).
fn split(round: Rounding, d: usize) -> Result<bool> {
    Ok(round == Rounding::Once || (HC_N * d).is_multiple_of(cpb(round)?))
}

pub struct KernRes {
    streams: crate::kt::hc_streams,
    tap: crate::kt::hc_tap,
    mix: crate::kt::hc_mix,
    proj_f32: crate::kt::hc_proj_f32,
    proj_bf16: crate::kt::hc_proj_bf16,
    fin: crate::kt::hc_fin,
    post: crate::kt::hc_post,
    head: crate::kt::hc_head,
    lag: crate::kt::hc_lag,
    add: crate::kt::res_add,
    gate: crate::kt::res_gate,
    rms_in: crate::kt::rmsnorm_in,
    depth: crate::kt::depth_mix,
}

impl KernRes {
    pub fn load(dev: &Dev) -> Result<KernRes> {
        Ok(KernRes {
            streams: dev.typed()?,
            tap: dev.typed()?,
            mix: dev.typed()?,
            proj_f32: dev.typed()?,
            proj_bf16: dev.typed()?,
            fin: dev.typed()?,
            post: dev.typed()?,
            head: dev.typed()?,
            lag: dev.typed()?,
            add: dev.typed()?,
            gate: dev.typed()?,
            rms_in: dev.typed()?,
            depth: dev.typed()?,
        })
    }

    /// x [n, ns, d] <- the embedding rows e [n, d] ns times.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/residual.cu), on stream `s`.
    pub unsafe fn streams(&self, s: Stream, e: u64, x: u64, n: u32, d: i32, ns: usize) -> Result<()> {
        launch!(self.streams, [n, (d as u32).div_ceil(256), 1], [256, 1, 1], s; e, x, d, ns as i32)
    }

    /// The mean of each row's `ns` streams x [n, ns, d] into its place `at` of out's rows of `of` places [n, of, d] (`Op::Tap`).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tap(&self, s: Stream, x: u64, out: u64, n: u32, d: i32, ns: usize, at: usize, of: usize) -> Result<()> {
        launch!(self.tap, [n, (d as u32).div_ceil(256), 1], [256, 1, 1], s; x, out, d, ns as i32, at as i32, of as i32)
    }

    /// The split projection of the streams x [n, 4, d] by `f` ([np, 4d], fp32 where `round` is `Once`, else bf16) into `part`: the number of parts a row.
    unsafe fn proj(&self, s: Stream, x: u64, f: u64, part: u64, n: u32, d: i32, np: i32, round: Rounding) -> Result<usize> {
        let ks = HC_N * d as usize / cpb(round)?;
        match round {
            Rounding::Once => launch!(self.proj_f32, [n, ks as u32, 1], [128, 1, 1], s; x, f, part, d, np)?,
            _ => launch!(self.proj_bf16, [n, ks as u32, 1], [128, 1, 1], s; x, f, part, d, np)?,
        }
        Ok(ks)
    }

    /// A block's mixing weights of the streams x [n, 4, d] (the projection `f` [24, 4d], base [24] and scale [3] fp32) into hcp [n, 24] fp32, made doubly stochastic
    /// (`iters` rounds, `hc_eps`), and the collapse normed (`eps`) with the weight nw [d] into h [n, d]. `part`: the split projection's sums (`part_words`).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn pre(
        &self,
        s: Stream,
        x: u64,
        (f, base, scale): (u64, u64, u64),
        nw: u64,
        h: u64,
        hcp: u64,
        part: u64,
        n: u32,
        d: i32,
        (eps, hc_eps, iters): (f32, f32, i32),
        round: Rounding,
    ) -> Result<()> {
        if d <= 0 || d % 2 != 0 || d > 16 * 480 {
            return err(format!("streams of {d} values: not supported"));
        }
        let single = (round == Rounding::Once) as i32;
        // (split over many blocks, then its rest; a block a row where the shape does not split)
        if split(round, d as usize)? {
            let ks = self.proj(s, x, f, part, n, d, HC_NP as i32, round)?;
            return launch!(self.fin, [n, 1, 1], [512, 1, 1], s; x, part, ks as i32, base, scale, nw, h, hcp, d, eps, hc_eps, iters, single, single);
        }
        launch!(self.mix, [n, 1, 1], [512, 1, 1], s; x, f, base, scale, nw, h, hcp, d, eps, hc_eps, iters, single)
    }

    /// The streams x [n, 4, d] after a block whose output is o [n, d]: updated in place by the weights hcp [n, 24] holds.
    ///
    /// # Safety
    /// As `streams`.
    pub unsafe fn post(&self, s: Stream, x: u64, o: u64, hcp: u64, n: u32, d: i32, round: Rounding) -> Result<()> {
        cpb(round)?;
        launch!(self.post, [n, (d as u32).div_ceil(256), 1], [256, 1, 1], s; x, o, hcp, d, (round == Rounding::Twice) as i32)
    }

    /// out [rows, d] = the final norm (`eps`, the weight nw [d]) of each row of the streams x [rows, 4, d] collapsed: by the hyper-head's weights where `weighted`
    /// gives its projection [4, 4d], base [4] and scale [1] (`part` its sums, `hc_eps`), else the mean.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn head(
        &self,
        s: Stream,
        x: u64,
        weighted: Option<(u64, u64, u64)>,
        part: u64,
        nw: u64,
        out: u64,
        rows: u32,
        d: i32,
        (eps, hc_eps): (f32, f32),
        round: Rounding,
    ) -> Result<()> {
        if d <= 0 || d > 16 * 512 {
            return err(format!("streams of {d} values: not supported"));
        }
        let single = (round == Rounding::Once) as i32;
        match weighted {
            Some((f, base, scale)) => {
                let ks = self.proj(s, x, f, part, rows, d, HC_N as i32, round)?;
                launch!(self.head, [rows, 1, 1], [512, 1, 1], s; x, part, ks as i32, base, scale, nw, out, d, eps, hc_eps, 1i32, single)
            }
            None => launch!(self.head, [rows, 1, 1], [512, 1, 1], s; x, 0u64, 0i32, 0u64, 0u64, nw, out, d, eps, 0f32, 0i32, single),
        }
    }

    /// Lagged mHC (`StreamsP::lagged`): h [n, d] = the streams x [n, 4, d] weighed by the previous lagged block's weights `lag` [n, 4] fp32 (one-hot on stream 0 where `first`), normed
    /// (`eps`, the weight nw [d]); then, where `keep`, `lag` takes this block's own weights (hcp [n, 24]'s first four, `pre`'s). `keep` false: the head's collapse (`Collapse::LastPre`).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lag(
        &self,
        s: Stream,
        x: u64,
        lag: u64,
        hcp: u64,
        nw: u64,
        h: u64,
        n: u32,
        d: i32,
        eps: f32,
        (first, keep): (bool, bool),
        round: Rounding,
    ) -> Result<()> {
        if d <= 0 || d > 16 * 512 {
            return err(format!("streams of {d} values: not supported"));
        }
        cpb(round)?;
        launch!(self.lag, [n, 1, 1], [512, 1, 1], s; x, lag, hcp, nw, h, d, eps, first as i32, keep as i32, (round == Rounding::Once) as i32)
    }

    /// x = bf16(x + a), e values.
    ///
    /// # Safety
    /// As `streams`.
    pub unsafe fn add(&self, s: Stream, x: u64, a: u64, e: usize) -> Result<()> {
        launch!(self.add, [(e as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, a, e as i64)
    }

    /// The gated residual of n rows of k: x = bf16(alpha x + beta h), the factors fp32 [k], one rounding (`muse::gated`).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gate(&self, s: Stream, x: u64, h: u64, alpha: u64, beta: u64, n: u32, k: i32) -> Result<()> {
        launch!(self.gate, [n, (k as u32).div_ceil(256), 1], [256, 1, 1], s; x, h, alpha, beta, k)
    }

    /// y = RMS(bf16(x w)) of n rows of k (eps), the weight scaling the input.
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rms_in(&self, s: Stream, x: u64, w: u64, y: u64, n: u32, k: i32, eps: f32) -> Result<()> {
        if k > 256 * 32 {
            return err(format!("an input-scaled norm of {k} values: the kernel takes up to 8,192"));
        }
        launch!(self.rms_in, [n, 1, 1], [256, 1, 1], s; x, w, y, k, eps)
    }

    /// A block's start in a depth mix: x [n, d] = bf16(x + o) where o is given (0: none); the depth mix of the `nb` sources at blk (a row's sources side by side, rows
    /// `rs` values apart) and x, scored by wn wp (each source normed by its RMS with `eps`); x made the source at `push` (blk's place for it; 0: none) and then
    /// zeros; y (rows ys apart) = the norm of the mix by w (`eps`; the normed value rounded, times the weight, rounded again).
    ///
    /// # Safety
    /// As `streams`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn depth_mix(
        &self,
        s: Stream,
        blk: u64,
        rs: usize,
        nb: usize,
        x: u64,
        o: u64,
        (wn, wp): (u64, u64),
        push: u64,
        w: u64,
        y: u64,
        ys: usize,
        d: usize,
        eps: f32,
        n: u32,
    ) -> Result<()> {
        if nb + 1 > 16 || d > 8192 || d % 8 != 0 || rs % 8 != 0 || ys % 8 != 0 {
            return err(format!(
                "a depth mix of {} sources over {d} values (rows {rs} and {ys} apart): the kernel takes 16 sources and up to 8,192 values, in 8s",
                nb + 1
            ));
        }
        launch!(self.depth, [n, 1, 1], [512, 1, 1], s; blk, rs as i64, nb as i32, x, o, wn, wp, push, w, y, ys as i32, d as i32, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = include_str!("../../kernels/residual.cu");

    /// The constants this file names are the kernels' (`#define`s of kernels/residual.cu), and each kernel it launches is there.
    #[test]
    fn the_constants_and_kernels_are_the_sources() {
        let define = |n: &str| -> Option<usize> {
            SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!((define("HC_N"), define("HC_NP")), (Some(HC_N), Some(HC_NP)));
        assert_eq!((define("HC_CPB_F32"), define("HC_CPB_BF16")), (cpb(Rounding::Once).ok(), cpb(Rounding::Twice).ok()));
        for name in ["hc_streams", "hc_mix", "hc_proj_f32", "hc_proj_bf16", "hc_fin", "hc_post", "hc_head", "res_add", "res_gate", "rmsnorm_in", "depth_mix"] {
            assert!(SRC.contains(&format!(" {name}(")), "no kernel {name} in kernels/residual.cu");
        }
    }

    /// The split projection's sums: every row's, by the rounding's columns a block; none where nothing is rounded.
    #[test]
    fn the_partial_sums_are_sized_by_the_rounding() {
        assert_eq!(part_words(Rounding::Once, 64, 4096).unwrap(), 64 * 64 * 25);
        assert_eq!(part_words(Rounding::Twice, 64, 4096).unwrap(), 64 * 32 * 25);
        assert!(part_words(Rounding::F32, 1, 4096).is_err());
    }

    /// Which kernel makes a block's mixing weights is the layer's (its width and rounding), never the call's rows: a streams width the bf16 projection's
    /// blocks divide is split, one they do not is a block a row, at every row count.
    #[test]
    fn the_mixing_kernel_is_chosen_by_the_shape_alone() {
        assert!(split(Rounding::Once, 4096).unwrap());
        assert!(split(Rounding::Twice, 4096).unwrap() && split(Rounding::Twice, 256).unwrap());
        assert!(!split(Rounding::Twice, 4000).unwrap());
        assert!(split(Rounding::F32, 4096).is_err());
    }
}
