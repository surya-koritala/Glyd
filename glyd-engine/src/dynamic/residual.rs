//! The residual rules' kernels (kernels/residual.cu), every family's: the residual streams (kind `Residual::Streams`, mHC), the depth mix (`Residual::DepthMix`), the
//! gated residual (`Residual::Gated`), the plain add, and the RMS norm whose weight scales its input. The streams' arithmetic is the kind's `StreamsP::rounding`: `Once`,
//! the release's fp32 mixing projection and update, each result rounded once (DeepSeek-V4); `Twice`, transformers' bf16 tensors between operations, the collapse's norm
//! rounding the normed value before its weight (GLM-5.3-Flash).

use crate::dev::Dev;
use crate::ffi::{err, Result, Stream};
use crate::launch;
use crate::runtime::kinds::Rounding;

/// Rows of streams up to which a bf16 mixing projection is split over many blocks (a decode step, a verified draft's rows); more take a block a row.
pub const HC_FEW: u32 = 8;

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

/// The partial sums the split projection writes for `rows` rows of streams `d` wide (fp32 words): every row's where the projection is fp32 (it is always split),
/// HC_FEW rows' where it is bf16.
pub fn part_words(round: Rounding, rows: usize, d: usize) -> Result<usize> {
    let c = cpb(round)?;
    let parts = if round == Rounding::Once { rows * (HC_N * d / c).max(1) } else { HC_FEW as usize * 64 };
    Ok(parts * (HC_NP + 1))
}

pub struct KernRes {
    streams: crate::kt::hc_streams,
    mix: crate::kt::hc_mix,
    proj_f32: crate::kt::hc_proj_f32,
    proj_bf16: crate::kt::hc_proj_bf16,
    fin: crate::kt::hc_fin,
    post: crate::kt::hc_post,
    head: crate::kt::hc_head,
    add: crate::kt::res_add,
    gate: crate::kt::res_gate,
    rms_in: crate::kt::rmsnorm_in,
    depth: crate::kt::depth_mix,
}

impl KernRes {
    pub fn load(dev: &Dev) -> Result<KernRes> {
        Ok(KernRes {
            streams: dev.typed()?,
            mix: dev.typed()?,
            proj_f32: dev.typed()?,
            proj_bf16: dev.typed()?,
            fin: dev.typed()?,
            post: dev.typed()?,
            head: dev.typed()?,
            add: dev.typed()?,
            gate: dev.typed()?,
            rms_in: dev.typed()?,
            depth: dev.typed()?,
        })
    }

    /// x [n, 4, d] <- the embedding rows e [n, d] four times.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/residual.cu), on stream `s`.
    pub unsafe fn streams(&self, s: Stream, e: u64, x: u64, n: u32, d: i32) -> Result<()> {
        launch!(self.streams, [n, (d as u32).div_ceil(256), 1], [256, 1, 1], s; e, x, d)
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
        // (an fp32 projection, and a bf16 one's few rows (a decode step): split over many blocks, then its rest; a bf16 one's more rows: a block a row)
        let k = HC_N * d as usize;
        if round == Rounding::Once || (n <= HC_FEW && k.is_multiple_of(cpb(round)?) && k / cpb(round)? <= 64) {
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

    /// x = bf16(x + a), e values.
    ///
    /// # Safety
    /// As `streams`.
    pub unsafe fn add(&self, s: Stream, x: u64, a: u64, e: usize) -> Result<()> {
        launch!(self.add, [(e as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, a, e as i64)
    }

    /// The gated residual of n rows of k: x = bf16(bf16(alpha x) + bf16(beta h)).
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
    /// `rs` values apart) and x, scored by wn wp; x made the source at `push` (blk's place for it; 0: none) and then zeros; y (rows ys apart) = the norm of the mix
    /// by w (`eps`; the normed value rounded, times the weight, rounded again).
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

    /// The split projection's sums: every row's where the projection is fp32 (the release's, always split), the few rows' where it is bf16; none where nothing is rounded.
    #[test]
    fn the_partial_sums_are_sized_by_the_rounding() {
        assert_eq!(part_words(Rounding::Once, 64, 4096).unwrap(), 64 * 64 * 25);
        assert_eq!(part_words(Rounding::Twice, 64, 4096).unwrap(), HC_FEW as usize * 64 * 25);
        assert!(part_words(Rounding::F32, 1, 4096).is_err());
    }
}
