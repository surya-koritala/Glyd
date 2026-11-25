//! Qwen3.8-Flash-Next's own kernels (kernels/qwen4x.cu) with typed launches, as the description executor runs them: the gated residuals' norm, mix and update
//! (src/dynamic/mod.rs), the delta rule's gated norm with the sigmoid gate, the n-gram embedding's rows, gate and convolution (src/dynamic/ngram.rs), the sparse
//! attention's indexer and its attention over the picked tokens (src/dynamic/picked.rs).

use crate::dev::Dev;
use crate::ffi::{err, Result, Stream};
use crate::launch;

/// Keys a tile of `qx_attn_sel` and its padded row (kernels/qwen4x.cu).
const QX_KB: u32 = 32;
const QX_KS: u32 = 129;

/// The kernels of kernels/qwen4x.cu the executor launches.
pub(crate) struct KernQx {
    hc_norm: crate::kt::qx_hc_norm,
    silu4: crate::kt::qx_silu4,
    hc_mix: crate::kt::qx_hc_mix,
    hc_post: crate::kt::qx_hc_post,
    gnorm: crate::kt::qx_gnorm_sig,
    ple_gate: crate::kt::qx_ple_gate,
    ple_conv: crate::kt::qx_ple_conv,
    /// (the n-gram embedding: its rows from the tokens, and the sequence's last tokens after a call)
    ngram_rows: crate::kt::qx_ngram_rows,
    ngram_hist: crate::kt::qx_ngram_hist,
    ix_prep: crate::kt::qx_ix_prep,
    ix_cmp: crate::kt::qx_ix_cmp,
    attn_sel: crate::kt::qx_attn_sel,
    /// (the gated merge of a split attention's parts: the attention kind's, src/dynamic/attn.rs)
    merge: crate::kt::attn_merge,
}

/// The attention kind's merge of `splits` parts of nq heads of 256 into `out` (rows `os` apart), times sigmoid of each query head's gate (`gate`: the rows of q, `gs` apart,
/// each head's 256 queries then its 256 gates).
fn merge_args(part: u64, splits: usize, gate: u64, gs: usize, out: u64, os: usize, nq: usize) -> crate::dynamic::attn::AttnArgs {
    crate::dynamic::attn::AttnArgs {
        part,
        S: splits as i32,
        d: 256,
        nq: nq as i32,
        out,
        os: os as i32,
        gate: gate + 512,
        gs: gs as i32,
        g_step: 512,
        ..Default::default()
    }
}

impl KernQx {
    pub(crate) fn load(dev: &Dev) -> Result<KernQx> {
        Ok(KernQx {
            hc_norm: dev.typed()?,
            silu4: dev.typed()?,
            hc_mix: dev.typed()?,
            hc_post: dev.typed()?,
            gnorm: dev.typed()?,
            ple_gate: dev.typed()?,
            ple_conv: dev.typed()?,
            ngram_rows: dev.typed()?,
            ngram_hist: dev.typed()?,
            ix_prep: dev.typed()?,
            ix_cmp: dev.typed()?,
            attn_sel: dev.typed()?,
            merge: dev.typed()?,
        })
    }

    /// xn [n, hc d] = each of the hc streams of x [n, hc, d] normed, times (1 + w).
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/qwen4x.cu).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_norm(&self, s: Stream, x: u64, w: u64, xn: u64, n: usize, d: usize, eps: f32, hc: usize) -> Result<()> {
        self.norm4(s, x, w, xn, n, d, eps, false, hc)
    }

    /// The same, the rms over each row's hc d values where `whole` (a draft head's input: the model's last streams normed as one vector).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn norm4(&self, s: Stream, x: u64, w: u64, xn: u64, n: usize, d: usize, eps: f32, whole: bool, hc: usize) -> Result<()> {
        streams_fit(d, hc)?;
        launch!(self.hc_norm, [n as u32, 1, 1], [256 * hc as u32, 1, 1], s; x, w, xn, d as i32, eps, whole as i32, hc as i32)
    }

    /// y = silu(y / hc) in place, `count` values.
    ///
    /// # Safety
    /// As `hc_norm`.
    pub(crate) unsafe fn silu4(&self, s: Stream, y: u64, count: usize, hc: usize) -> Result<()> {
        launch!(self.silu4, [(count as u32).div_ceil(256), 1, 1], [256, 1, 1], s; y, count as i32, hc as i32)
    }

    /// h [n, d] = the gated mean of the hc normed streams; inj [n, hc] fp32 from the inject product injl [n, hc] where given (0: none).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_mix(&self, s: Stream, gw: u64, xn: u64, injl: u64, h: u64, inj: u64, n: usize, d: usize, hc: usize) -> Result<()> {
        streams_fit(d, hc)?;
        launch!(self.hc_mix, [n as u32, (d as u32).div_ceil(256), 1], [256, 1, 1], s; gw, xn, injl, h, inj, d as i32, hc as i32)
    }

    /// x [n, hc, d] += o [n, d] times each stream's weight inj [n, hc].
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_post(&self, s: Stream, x: u64, o: u64, inj: u64, n: usize, d: usize, hc: usize) -> Result<()> {
        launch!(self.hc_post, [n as u32, (d as u32).div_ceil(256), 1], [256, 1, 1], s; x, o, inj, d as i32, hc as i32)
    }

    /// The GDN's gated norm with the sigmoid gate, in place: core [n, nv 128], z rows zs apart, w bf16 [128].
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gnorm(&self, s: Stream, core: u64, z: u64, zs: usize, w: u64, n: usize, nv: usize, eps: f32) -> Result<()> {
        launch!(self.gnorm, [n as u32, nv as u32, 1], [128, 1, 1], s; core, z, zs as i32, w, nv as i32, eps)
    }

    /// The n-gram embedding's gate from the key and value products kv [n, 5 d] and the streams x: Condor's form (`engram` false: gv and gvn [n, 4 d] for
    /// `ple_conv`), or Engram's (`engram`: the gated value added into x, the weights wk and wq fp32; wc unread).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ple_gate(
        &self,
        s: Stream,
        kv: u64,
        x: u64,
        wk: u64,
        wq: u64,
        wc: u64,
        gv: u64,
        gvn: u64,
        n: usize,
        d: usize,
        eps: f32,
        engram: bool,
    ) -> Result<()> {
        if d > 32 * 256 {
            return err(format!("streams of {d}: the n-gram embedding's gate takes up to 8,192"));
        }
        launch!(self.ple_gate, [n as u32, 1, 1], [1024, 1, 1], s; kv, x, wk, wq, wc, gv, gvn, d as i32, eps, engram as i32)
    }

    /// PLE's dilated convolution of `taps` taps `dilation` rows apart over the n rows (state st [(taps - 1) dilation, c]) and its sum into the streams x.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ple_conv(
        &self,
        s: Stream,
        gvn: u64,
        gv: u64,
        w: u64,
        st: u64,
        x: u64,
        n: usize,
        c: usize,
        (taps, dilation): (usize, usize),
    ) -> Result<()> {
        if taps == 0 || (taps - 1) * dilation > 32 {
            return err(format!("a convolution of {taps} taps {dilation} apart: the kernel keeps up to 32 rows before a row"));
        }
        launch!(self.ple_conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; gvn, gv, w, st, x, n as i32, c as i32, taps as i32, dilation as i32)
    }

    /// The n-gram embedding's rows e [n, H row] of a call's n rows of one sequence: their tokens tok (int32), the sequence's `order` - 1 ids before them hist (int32, + 1;
    /// 0: none), the hash's numbers (`hash`: its `order` multipliers, H sizes and H offsets, int64), the table [rows, row] bf16; each token its compressed id first where
    /// `map` (int32 [vocabulary]; 0: none), a segment ended by `eos` where `segments`; then hist moved on past the call (`qx_ngram_rows`, `qx_ngram_hist`).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ngram_rows(
        &self,
        s: Stream,
        tok: u64,
        hist: u64,
        hash: [u64; 3],
        table: u64,
        e: u64,
        n: usize,
        h: usize,
        per: usize,
        row: usize,
        eos: u32,
        (order, map, segments): (usize, u64, bool),
    ) -> Result<()> {
        if !(2..=4).contains(&order) {
            return err(format!("n-grams of order {order}: the kernel takes 2 to 4"));
        }
        let [mult, sizes, offsets] = hash;
        launch!(self.ngram_rows, [n as u32, 1, 1], [256, 1, 1], s; tok, hist, mult, sizes, offsets, table, e, h as i32, per as i32, row as i32, eos as i32, order as i32, map, segments as i32)?;
        launch!(self.ngram_hist, [1, 1, 1], [1, 1, 1], s; tok, hist, n as i32, order as i32, map)
    }

    /// The indexer's queries (normed, turned) into iq [n, ih dim] and its raw keys into the cache of the executor's pages `ik` at the rows' positions (its
    /// rows through its block table, `ik.1`; 0: none).
    ///
    /// # Safety
    /// As `hc_norm`; the table names a block for every position in `pos`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ix_prep_at(
        &self,
        s: Stream,
        qk: u64,
        wq: u64,
        cs: u64,
        sn: u64,
        half: usize,
        pos: u64,
        iq: u64,
        ik: (u64, u64),
        n: usize,
        ih: usize,
        eps: f32,
        dim: usize,
    ) -> Result<()> {
        index_fits(dim, 1)?;
        launch!(self.ix_prep, [n as u32, ih as u32 + 1, 1], [dim as u32, 1, 1], s; qk, wq, cs, sn, half as i32, pos, iq, ik.0, ik.1, ih as i32, eps, dim as i32)
    }

    /// The keys of the pools of `block` tokens the call's rows complete into pk, on caches of the executor's pages: the raw keys' and the pools' rows through their
    /// block tables (0: none).
    ///
    /// # Safety
    /// As `hc_norm`; the tables name a block for every position and pool the call reads and writes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ix_cmp_at(
        &self,
        s: Stream,
        ik: (u64, u64),
        wk: u64,
        cs: u64,
        sn: u64,
        half: usize,
        pos: u64,
        pk: (u64, u64),
        n: usize,
        eps: f32,
        dim: usize,
        block: usize,
    ) -> Result<()> {
        index_fits(dim, block)?;
        launch!(self.ix_cmp, [(n as u32).div_ceil(block as u32) + 1, 1, 1], [dim as u32, 1, 1], s; ik.0, ik.1, wk, cs, sn, half as i32, pos, pk.0, pk.1, n as i32, eps, dim as i32, block as i32)
    }

    /// The gated attention of the n rows over the tokens sel/cnt picked (in `splits` parts merged by attn_merge where more than one: `part` holds n nq splits 258
    /// floats), over a cache of the executor's pages: kc and vc's rows through their block table (`kv.2`, 0: none), position p's at row tab[p / 16] 16 + p % 16.
    ///
    /// # Safety
    /// As `hc_norm`; the table names a block for every position picked.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attn_sel_at(
        &self,
        s: Stream,
        q: u64,
        qs: usize,
        gate: u64,
        gs: usize,
        kv: (u64, u64, u64),
        sel: u64,
        cnt: u64,
        tmax: usize,
        out: u64,
        os: usize,
        nq: usize,
        nk: usize,
        part: u64,
        splits: usize,
        n: usize,
    ) -> Result<()> {
        let g = (nq / nk) as u32;
        let smem = (QX_KB * (QX_KS + 128) + 128 * g) * 4;
        let sl2 = (1.0f64 / 16.0 * std::f64::consts::LOG2_E) as f32;
        launch!(self.attn_sel, [n as u32, nk as u32, splits as u32], [32 * g, 1, 1], smem smem, s; q, qs as i32, gate, gs as i32, kv.0, kv.1, kv.2, sel, cnt, tmax as i32, out, os as i32, nq as i32, nk as i32, sl2, part, splits as i32)?;
        if splits > 1 {
            launch!(self.merge, [n as u32, nq as u32, 1], [256, 1, 1], s; merge_args(part, splits, gate, gs, out, os, nq))?;
        }
        Ok(())
    }
}

/// The indexer's kernels take heads of a multiple of 32 up to 1,024 values (a thread each) over pools of at least one token.
pub fn index_fits(dim: usize, block: usize) -> Result<()> {
    if dim == 0 || dim > 1024 || !dim.is_multiple_of(32) || block == 0 {
        return err(format!("index heads of {dim} over pools of {block}: the kernels take heads of a multiple of 32 up to 1,024"));
    }
    Ok(())
}

/// The gated residuals' kernels take 1 to 4 streams of up to 4,096 values (a block's 256 threads a stream, 16 values a thread).
pub fn streams_fit(d: usize, hc: usize) -> Result<()> {
    if !(1..=4).contains(&hc) || d > 16 * 256 {
        return err(format!("{hc} streams of {d}: the kernels take 1 to 4 streams of up to 4,096"));
    }
    Ok(())
}
