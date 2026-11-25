//! Qwen3.8-Flash-Next on the GPU, one sequence at a time (others wait beside it: `SeqQx`, `EngineQx::swap_seq`): four residual streams per token joined by Gated Residuals around each layer's two blocks (`x_s += out * inj_s`, the block's input a gated mean of the normed streams),
//! the first block Gated DeltaNet (Qwen3.5's kernels, the gated norm's gate a sigmoid) or Qwen Sparse Attention (Qwen3.5's gated attention over the tokens the indexer picks: dense up to 2,051 tokens), the second a
//! mixture of experts (`MoeBlockQx`: `QxMoe` holds a layer's experts on the GPU; the expert cache's block plugs in the same way), the n-gram embedding (PLE) added to the streams before layer 1, and the final mixer
//! into the LM head. The embedding and the n-gram table are read on the host from the checkpoint's files mapped (their rows copied up with the step), never loaded to the GPU. The layers' math: transformers 5.19.0's
//! modeling_qwen4_exp.py (tools/qwen4x_ref.py checks this file against it); the config and tensor names: src/qwen4x.rs; the new kernels: kernels/qwen4x.cu.

use crate::dev::{copy_d2h_async, copy_h2d_async, Buf, Dev, Pinned};
use crate::ffi::{err, Error, Lib, Result, Stream};
use crate::kern::Kern;
use crate::kern35::{Kern35, Mat};
use crate::kern_glm::{f32_bytes, KernGlm};
use crate::launch;
use crate::mapped::Mapping;
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeScratch, MoeWeights, Router, Shared};
use crate::moe_rows::Down;
use crate::qwen35::Pack35;
use crate::qwen4x::{ConfigQx, QxKind};
use crate::sched::Sample;
use crate::weights::{ordered, read_pack, rope_tables, upload_bytes, upload_packed, Layout, PackBuf};
use glyd_gpu::safetensors::{Checkpoint, TensorInfo};
use std::cell::{Cell, OnceCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Keys a tile of `qx_attn_sel` and its padded row (kernels/qwen4x.cu), and the parts a decode step's attention over picked tokens is split into.
const QX_KB: u32 = 32;
const QX_KS: u32 = 129;
const SEL_SPLITS: usize = 16;

/// The kernels of kernels/qwen4x.cu with typed launches.
pub struct KernQx {
    /// a decode step's GDN mixer in fewer launches: the convolution beside the b and a product, the rule with the gated norm after it
    gdn_conv_ba: crate::kt::qx_gdn_conv_ba,
    gdn_rec_gn: crate::kt::qx_gdn_rec_gn,
    /// a row's top token's probability (`qx_top_prob`)
    top_prob: crate::kt::qx_top_prob,
    /// the Gated Residual's norm and down for 5 to 8 rows by slices of columns (`qx_gr1s_8`) and its scratch: the slices' sums, their squares, a counter a group of rows
    gr1s: crate::kt::qx_gr1s_8,
    gr1s_part: Buf,
    gr1s_sq: Buf,
    gr1s_cnt: Buf,
    hc_norm: crate::kt::qx_hc_norm,
    silu4: crate::kt::qx_silu4,
    hc_mix: crate::kt::qx_hc_mix,
    hc_post: crate::kt::qx_hc_post,
    gnorm: crate::kt::qx_gnorm_sig,
    ple_gate: crate::kt::qx_ple_gate,
    ple_conv: crate::kt::qx_ple_conv,
    /// (the description executor's n-gram embedding: its rows from the tokens, and the sequence's last tokens after a call)
    ngram_rows: crate::kt::qx_ngram_rows,
    ngram_hist: crate::kt::qx_ngram_hist,
    ix_prep: crate::kt::qx_ix_prep,
    ix_cmp: crate::kt::qx_ix_cmp,
    attn_sel: crate::kt::qx_attn_sel,
    attn_tc: crate::kt::qx_attn_tc,
    ix_score: crate::kt::qx_ix_score,
    select: crate::kt::qx_select,
    /// (the split attention of a decode step's rows with 4 and 8 warps a block)
    attn_dec: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_attn_dec4>>; 2],
    conv_tok: crate::kt::qx_conv_tok,
    conv_state: crate::kt::qx_conv_state,
    gdn_prep: crate::kt::qx_gdn_prep,
    gdn_scan: crate::kt::qx_gdn_scan,
    /// the two-pass delta rule's scratch (`delta2`), made by its first call and grown by a longer one
    gdn_buf: std::cell::RefCell<Option<Buf>>,
    /// (the gated merge of a split attention's parts: the attention kind's, src/dynamic/attn.rs)
    merge: crate::kt::attn_merge,
    // (the fused Gated Residual's for 1, 2, 4 and 8 rows; its first launch also by streams, `qx_gr1q`, on gr1s's scratch)
    gr1: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_gr1_1>>; 4],
    gr1q: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_gr1q_1>>; 3],
    gr2: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::qx_gr2_1>>; 4],
    i8bf16: crate::kt::qx_i8_bf16,
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

/// The kernel of a few-rows family (1, 2, 4 or 8 rows) for n rows (1 to 8).
pub(crate) fn rows_kernel<A>(k: &[crate::dev::ByArgs<A>; 4], n: usize) -> &crate::dev::ByArgs<A> {
    &k[match n {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        _ => 3,
    }]
}

impl KernQx {
    pub fn load(dev: &Dev) -> Result<KernQx> {
        Ok(KernQx {
            gdn_conv_ba: dev.typed()?,
            gdn_rec_gn: dev.typed()?,
            top_prob: dev.typed()?,
            gr1s: dev.typed()?,
            gr1s_part: Buf::new(GR1S_SLICES * GR1S_GROUPS * 32 * 8 * 4)?,
            gr1s_sq: Buf::new(GR1S_GROUPS * GR1S_SLICES * 8 * 4)?,
            gr1s_cnt: Buf::zeroed(GR1S_GROUPS.max(GR1Q_GROUPS) * 4)?,
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
            ix_score: dev.typed()?,
            attn_dec: {
                let ks = [dev.by_args::<crate::kt::qx_attn_dec4>()?, dev.by_args::<crate::kt::qx_attn_dec8>()?];
                for (k, w) in ks.iter().zip([4u32, 8]) {
                    k.allow_smem(attn_dec_smem(w))?;
                }
                ks
            },
            select: {
                let k = dev.typed::<crate::kt::qx_select>()?;
                k.allow_smem((QX_SEL_CAP * 4) as u32)?;
                k
            },
            attn_tc: dev.typed()?,
            conv_tok: dev.typed()?,
            conv_state: dev.typed()?,
            gdn_prep: {
                let k = dev.typed::<crate::kt::qx_gdn_prep>()?;
                k.allow_smem(GDN_PREP_SMEM)?;
                k
            },
            gdn_scan: {
                let k = dev.typed::<crate::kt::qx_gdn_scan>()?;
                k.allow_smem(GDN_SCAN_SMEM)?;
                k
            },
            gdn_buf: std::cell::RefCell::new(None),
            merge: dev.typed()?,
            gr1: [
                dev.by_args::<crate::kt::qx_gr1_1>()?,
                dev.by_args::<crate::kt::qx_gr1_2>()?,
                dev.by_args::<crate::kt::qx_gr1_4>()?,
                dev.by_args::<crate::kt::qx_gr1_8>()?,
            ],
            gr1q: [dev.by_args::<crate::kt::qx_gr1q_1>()?, dev.by_args::<crate::kt::qx_gr1q_2>()?, dev.by_args::<crate::kt::qx_gr1q_4>()?],
            gr2: [
                dev.by_args::<crate::kt::qx_gr2_1>()?,
                dev.by_args::<crate::kt::qx_gr2_2>()?,
                dev.by_args::<crate::kt::qx_gr2_4>()?,
                dev.by_args::<crate::kt::qx_gr2_8>()?,
            ],
            i8bf16: dev.typed()?,
        })
    }

    /// xn [n, 4 d] = each stream of x [n, 4, d] normed, times (1 + w).
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/qwen4x.cu).
    pub unsafe fn hc_norm(&self, s: Stream, x: u64, w: u64, xn: u64, n: usize, d: usize, eps: f32) -> Result<()> {
        self.norm4(s, x, w, xn, n, d, eps, false)
    }

    /// The same, the rms over each row's 4 d values where `whole`.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn norm4(&self, s: Stream, x: u64, w: u64, xn: u64, n: usize, d: usize, eps: f32, whole: bool) -> Result<()> {
        if d > 16 * 256 {
            return err(format!("streams of {d}: the norm takes up to 4,096"));
        }
        launch!(self.hc_norm, [n as u32, 1, 1], [1024, 1, 1], s; x, w, xn, d as i32, eps, whole as i32)
    }

    /// The fused Gated Residual's first launch for n rows (at most 8): each stream's 1 / rms into rs [n, 4], the bottleneck's activations into lo [n, r], the streams' weights into inj [n, 4] where `injw` (bf16 [4, 4 d]) is
    /// given (0: none).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gr1(
        &self,
        s: Stream,
        x: u64,
        w: u64,
        dq: u64,
        ds: u64,
        injw: u64,
        rs: u64,
        lo: u64,
        inj: u64,
        n: usize,
        d: usize,
        r: usize,
        eps: f32,
        post: (u64, u64),
    ) -> Result<()> {
        if n == 0 || n > 8 || r % 4 != 0 || d % 8 != 0 {
            return err(format!("the fused Gated Residual on {n} rows of a bottleneck of {r} over 4 x {d}"));
        }
        // (3 and 4 rows by streams, `qx_gr1q`: qxbench 4 rows 18.3 -> 13.0 us, but 2 rows 9.7 -> 10.3, 1 row 7.9 -> 8.9)
        if (3..=4).contains(&n) {
            return self.gr1q(s, x, w, dq, ds, injw, rs, lo, inj, n, d, r, eps, post);
        }
        // (5 to 8 rows: by slices of 256 columns, a block also a group of 32 of the bottleneck's and the inject's rows: `qx_gr1s`; qxbench: 8 rows 34.4 -> 16.9 us, but 3 rows 11.6 -> 13.7)
        let (slices, groups) = ((4 * d).div_ceil(256), (r + 4).div_ceil(32));
        if n >= 5 && (4 * d) % 256 == 0 && d % 256 == 0 && slices <= GR1S_SLICES && groups <= GR1S_GROUPS {
            return launch!(self.gr1s, [slices as u32, groups as u32, 1], [256, 1, 1], s; x, w, dq, ds, injw, rs, lo, inj, n as i32, d as i32, r as i32, eps, post.0, post.1, self.gr1s_part.ptr, self.gr1s_sq.ptr, self.gr1s_cnt.ptr);
        }
        self.gr1_rows(s, x, w, dq, ds, injw, rs, lo, inj, n, d, r, eps, post)
    }

    /// `gr1` by slices of rows, whatever the rows of x (qx_gr1's kernel: a decode step's).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gr1_rows(
        &self,
        s: Stream,
        x: u64,
        w: u64,
        dq: u64,
        ds: u64,
        injw: u64,
        rs: u64,
        lo: u64,
        inj: u64,
        n: usize,
        d: usize,
        r: usize,
        eps: f32,
        post: (u64, u64),
    ) -> Result<()> {
        // (kernels/qwen4x.cu QG_RB: 2 rows of down a block, the inject's 4 rows in 2 blocks more)
        let blocks = (r / 2) as u32 + if injw != 0 { 2 } else { 0 };
        launch!(rows_kernel(&self.gr1, n), [blocks, 1, 1], [256, 1, 1], s; x, w, dq, ds, injw, rs, lo, inj, n as i32, d as i32, r as i32, eps, post.0, post.1)
    }

    /// `gr1` by streams (`qx_gr1q`, 1 to 4 rows of x): a block a stream's d columns (d / 8 threads) by a group of 9 rows; `gr1_rows` where it does not fit.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gr1q(
        &self,
        s: Stream,
        x: u64,
        w: u64,
        dq: u64,
        ds: u64,
        injw: u64,
        rs: u64,
        lo: u64,
        inj: u64,
        n: usize,
        d: usize,
        r: usize,
        eps: f32,
        post: (u64, u64),
    ) -> Result<()> {
        // (kernels/qwen4x.cu QG_RQ)
        let groups = (r + if injw != 0 { 4 } else { 0 }).div_ceil(9);
        if n > 4 || d % 256 != 0 || d / 8 > 320 || groups > GR1Q_GROUPS || 4 * (r + 4) * 4 > GR1S_SLICES * GR1S_GROUPS * 32 * 8 {
            return self.gr1_rows(s, x, w, dq, ds, injw, rs, lo, inj, n, d, r, eps, post);
        }
        launch!(self.gr1q[[0, 0, 1, 2, 2][n]], [4, groups as u32, 1], [(d / 8) as u32, 1, 1], s; x, w, dq, ds, injw, rs, lo, inj, n as i32, d as i32, r as i32, eps, post.0, post.1, self.gr1s_part.ptr, self.gr1s_sq.ptr, self.gr1s_cnt.ptr)
    }

    /// Its second launch: the block's input h [n, d] from up's int8 rows times lo and the normed streams.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gr2(
        &self,
        s: Stream,
        x: u64,
        w: u64,
        rs: u64,
        lo: u64,
        uq: u64,
        us: u64,
        h: u64,
        n: usize,
        d: usize,
        r: usize,
        post: (u64, u64),
    ) -> Result<()> {
        if n == 0 || n > 8 || r % 16 != 0 || d % 8 != 0 {
            return err(format!("the fused Gated Residual on {n} rows with a bottleneck of {r} over {d}"));
        }
        launch!(rows_kernel(&self.gr2, n), [(d / 8) as u32, 1, 1], [256, 1, 1], s; x, w, rs, lo, uq, us, h, n as i32, d as i32, r as i32, post.0, post.1)
    }

    /// out [orows, ocols] bf16 = q [rows, cols] int8 times each row's scale, zeros past them (cols and ocols multiples of 4).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn i8_bf16(&self, s: Stream, q: u64, scale: u64, out: u64, rows: usize, cols: usize, orows: usize, ocols: usize) -> Result<()> {
        if cols % 4 != 0 || ocols % 4 != 0 || rows > orows || cols > ocols {
            return err(format!("int8 rows {rows} x {cols} made bf16 into {orows} x {ocols}"));
        }
        launch!(self.i8bf16, [((orows * ocols / 4) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; q, scale, out, rows as i32, cols as i32, orows as i32, ocols as i32)
    }

    /// y = silu(y / 4) in place, `count` values.
    ///
    /// # Safety
    /// As `hc_norm`.
    pub unsafe fn silu4(&self, s: Stream, y: u64, count: usize) -> Result<()> {
        launch!(self.silu4, [(count as u32).div_ceil(256), 1, 1], [256, 1, 1], s; y, count as i32)
    }

    /// h [n, d] = the gated mean of the normed streams; inj [n, 4] fp32 from the inject product injl [n, 4] where given (0: none).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn hc_mix(&self, s: Stream, gw: u64, xn: u64, injl: u64, h: u64, inj: u64, n: usize, d: usize) -> Result<()> {
        launch!(self.hc_mix, [n as u32, (d as u32).div_ceil(256), 1], [256, 1, 1], s; gw, xn, injl, h, inj, d as i32)
    }

    /// x [n, 4, d] += o [n, d] times each stream's weight inj [n, 4].
    ///
    /// # Safety
    /// As `hc_norm`.
    pub unsafe fn hc_post(&self, s: Stream, x: u64, o: u64, inj: u64, n: usize, d: usize) -> Result<()> {
        launch!(self.hc_post, [n as u32, (d as u32).div_ceil(256), 1], [256, 1, 1], s; x, o, inj, d as i32)
    }

    /// The GDN's gated norm with the sigmoid gate, in place: core [n, nv 128], z rows zs apart, w bf16 [128].
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    /// The convolution and SiLU of the first c channels of x [n, xs] (weights w [c, 4], state st [3, c]) into y [n, c], and yb [n, nb] = xb [n, kb] wb^T (a small bf16 matrix), in one launch.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_conv_ba(
        &self,
        s: Stream,
        x: u64,
        xs: usize,
        w: u64,
        st: u64,
        y: u64,
        n: usize,
        c: usize,
        xb: u64,
        wb: u64,
        yb: u64,
        ybs: usize,
        nb: usize,
        kb: usize,
    ) -> Result<()> {
        let (cb, nby) = (c.div_ceil(256), nb.div_ceil(8));
        launch!(self.gdn_conv_ba, [(cb + n * nby) as u32, 1, 1], [256, 1, 1], s; x, xs as i32, w, st, y, n as i32, c as i32, cb as i32, xb, kb as i32, wb, yb, ybs as i32, nb as i32, kb as i32, nby as i32)
    }

    /// The gated delta rule token by token (k35's `delta`, not chunked) with the gated norm (`gnorm`) of each token's outputs after it: core [n, nv 128] the normed outputs; `st1` (0 for none) the state after
    /// the first row too.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_rec_gn(
        &self,
        s: Stream,
        xc: u64,
        c: usize,
        ba: u64,
        sba: usize,
        a_log: u64,
        dt_bias: u64,
        state: u64,
        core: u64,
        n: usize,
        nk: usize,
        nv: usize,
        z: u64,
        zs: usize,
        gw: u64,
        eps: f32,
        st1: u64,
    ) -> Result<()> {
        launch!(self.gdn_rec_gn, [nv as u32, 1, 1], [128, 1, 1], s; xc, c as i32, ba, sba as i32, a_log, dt_bias, state, core, n as i32, nk as i32, nv as i32, z, zs as i32, gw, eps, st1)
    }

    /// p [1] = the probability of the top token of the logits lg (bf16 [v]) whose largest is vmax (fp32, argmax's value).
    ///
    /// # Safety
    /// As `hc_norm`.
    pub unsafe fn top_prob(&self, s: Stream, lg: u64, v: usize, vmax: u64, p: u64) -> Result<()> {
        launch!(self.top_prob, [1, 1, 1], [1024, 1, 1], s; lg, v as i32, vmax, p)
    }

    pub unsafe fn gnorm(&self, s: Stream, core: u64, z: u64, zs: usize, w: u64, n: usize, nv: usize, eps: f32) -> Result<()> {
        launch!(self.gnorm, [n as u32, nv as u32, 1], [128, 1, 1], s; core, z, zs as i32, w, nv as i32, eps)
    }

    /// PLE's gate: gv and gvn [n, 4 d] from the key and value products kv [n, 5 d] and the streams x.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ple_gate(&self, s: Stream, kv: u64, x: u64, wk: u64, wq: u64, wc: u64, gv: u64, gvn: u64, n: usize, d: usize, eps: f32) -> Result<()> {
        if d > 16 * 256 {
            return err(format!("streams of {d}: PLE takes up to 4,096"));
        }
        launch!(self.ple_gate, [n as u32, 1, 1], [1024, 1, 1], s; kv, x, wk, wq, wc, gv, gvn, d as i32, eps)
    }

    /// PLE's dilated convolution over the n rows (state st [9, c]) and its sum into the streams x.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ple_conv(&self, s: Stream, gvn: u64, gv: u64, w: u64, st: u64, x: u64, n: usize, c: usize) -> Result<()> {
        launch!(self.ple_conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; gvn, gv, w, st, x, n as i32, c as i32)
    }

    /// The n-gram embedding's rows e [n, H row] of a call's n rows of one sequence: their tokens tok (int32), the sequence's two tokens before them hist (int32, + 1; 0:
    /// none), the hash's numbers (`hash`: its 3 multipliers, H sizes and H offsets, int64 as the checkpoint holds them), the table [rows, row] bf16; then hist moved on
    /// past the call (`qx_ngram_rows`, `qx_ngram_hist`).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ngram_rows(
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
    ) -> Result<()> {
        let [mult, sizes, offsets] = hash;
        launch!(self.ngram_rows, [n as u32, 1, 1], [256, 1, 1], s; tok, hist, mult, sizes, offsets, table, e, h as i32, per as i32, row as i32, eos as i32)?;
        launch!(self.ngram_hist, [1, 1, 1], [1, 1, 1], s; tok, hist, n as i32)
    }

    /// The indexer's queries (normed, turned) into iq [n, ih 128] and its raw keys into the cache ik at the rows' positions.
    ///
    /// # Safety
    /// As `hc_norm`; the cache holds every position in `pos`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ix_prep(
        &self,
        s: Stream,
        qk: u64,
        wq: u64,
        cs: u64,
        sn: u64,
        half: usize,
        pos: u64,
        iq: u64,
        ik: u64,
        n: usize,
        ih: usize,
        eps: f32,
    ) -> Result<()> {
        self.ix_prep_at(s, qk, wq, cs, sn, half, pos, iq, (ik, 0), n, ih, eps)
    }

    /// `ix_prep` into a cache of the executor's pages: `ik`'s rows through its block table (0: none).
    ///
    /// # Safety
    /// As `ix_prep`; the table names a block for every position in `pos`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ix_prep_at(
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
    ) -> Result<()> {
        launch!(self.ix_prep, [n as u32, ih as u32 + 1, 1], [128, 1, 1], s; qk, wq, cs, sn, half as i32, pos, iq, ik.0, ik.1, ih as i32, eps)
    }

    /// The keys of the blocks of 4 tokens the call's rows complete into pk.
    ///
    /// # Safety
    /// As `ix_prep`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ix_cmp(&self, s: Stream, ik: u64, wk: u64, cs: u64, sn: u64, half: usize, pos: u64, pk: u64, n: usize, eps: f32) -> Result<()> {
        self.ix_cmp_at(s, (ik, 0), wk, cs, sn, half, pos, (pk, 0), n, eps)
    }

    /// `ix_cmp` on caches of the executor's pages: the raw keys' and the pools' rows through their block tables (0: none).
    ///
    /// # Safety
    /// As `ix_cmp`; the tables name a block for every position and pool the call reads and writes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ix_cmp_at(
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
    ) -> Result<()> {
        launch!(self.ix_cmp, [(n as u32).div_ceil(4) + 1, 1, 1], [128, 1, 1], s; ik.0, ik.1, wk, cs, sn, half as i32, pos, pk.0, pk.1, n as i32, eps)
    }

    /// The indexer's scores of a prompt chunk's n rows on the tensor cores (`qx_ix_score`: four heads of 128, no head weights): scores [n, pmax] fp32 for the pools each row sees, `pools` the most a row of the call sees.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ix_score(
        &self,
        s: Stream,
        iq: u64,
        pk: u64,
        pos: u64,
        scores: u64,
        n: usize,
        pmax: usize,
        ksel: usize,
        pools: usize,
        scale: f32,
    ) -> Result<()> {
        if n == 0 || pools == 0 {
            return Ok(());
        }
        launch!(self.ix_score, [pools.div_ceil(128) as u32, n.div_ceil(16) as u32, 1], [128, 1, 1], s; iq, pk, pos, scores, n as i32, pmax as i32, ksel as i32, scale)
    }

    /// `KernGlm::select`'s picks for n rows (`qx_select`: the same sel and cnt, faster over a long context's pools), with the tail's tokens.
    ///
    /// # Safety
    /// As `hc_norm`; ksel at most 1,024.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn select(
        &self,
        s: Stream,
        scores: u64,
        pmax: usize,
        pos: u64,
        sel: u64,
        cnt: u64,
        tmax: usize,
        kp: usize,
        ksel: usize,
        n: usize,
    ) -> Result<()> {
        if ksel == 0 || ksel > 1024 {
            return err(format!("an indexer that picks {ksel} groups of keys: this engine runs up to 1,024"));
        }
        let cap = pmax.min(if crate::dev::smem64() { QX_SEL_CAP_64 } else { QX_SEL_CAP });
        launch!(self.select, [n as u32, 1, 1], [1024, 1, 1], smem (cap * 4) as u32, s; scores, pmax as i32, pos, sel, cnt, tmax as i32, kp as i32, ksel as i32, cap as i32)
    }

    /// `attn_sel` for a decode step's few rows on the tensor cores, split (`qx_attn_dec*`: `wpb` (4 or 8) warps a block, a 16-token step each, combined in the block; the blocks' parts merged by attn_merge;
    /// `part` holds n nq ceil(tmax / (16 wpb)) parts of 258 floats); 24 query heads, 2 key-value heads.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_dec(
        &self,
        s: Stream,
        q: u64,
        qs: usize,
        gate: u64,
        gs: usize,
        kc: u64,
        vc: u64,
        sel: u64,
        cnt: u64,
        tmax: usize,
        out: u64,
        os: usize,
        part: u64,
        wpb: usize,
        n: usize,
    ) -> Result<()> {
        let sl2 = (1.0f64 / 16.0 * std::f64::consts::LOG2_E) as f32;
        let (k, w) = if wpb == 4 { (&self.attn_dec[0], 4u32) } else { (&self.attn_dec[1], 8) };
        let splits = tmax.div_ceil(16 * w as usize);
        launch!(k, [splits as u32, 2, n as u32], [32 * w, 1, 1], smem attn_dec_smem(w), s; q, qs as i32, kc, vc, sel, cnt, tmax as i32, sl2, part)?;
        launch!(self.merge, [n as u32, 24, 1], [256, 1, 1], s; merge_args(part, splits, gate, gs, out, os, 24))
    }

    /// The gated attention of the n rows over the tokens sel/cnt picked (in `splits` parts merged by attn_merge where more than one: `part` holds n nq splits 258 floats).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_sel(
        &self,
        s: Stream,
        q: u64,
        qs: usize,
        gate: u64,
        gs: usize,
        kc: u64,
        vc: u64,
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
        self.attn_sel_at(s, q, qs, gate, gs, (kc, vc, 0), sel, cnt, tmax, out, os, nq, nk, part, splits, n)
    }

    /// `attn_sel` over a cache of the executor's pages: kc and vc's rows through their block table (`kv.2`, 0: none), position p's at row tab[p / 16] 16 + p % 16.
    ///
    /// # Safety
    /// As `attn_sel`; the table names a block for every position picked.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_sel_at(
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

    /// The GDN's causal convolution and SiLU of a prompt chunk's n rows (as `Kern35::conv`, a thread 16 rows of a channel), then its state moved on.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_tok(&self, s: Stream, x: u64, xs: usize, w: u64, st: u64, y: u64, n: usize, c: usize) -> Result<()> {
        launch!(self.conv_tok, [c.div_ceil(256) as u32, n.div_ceil(16) as u32, 1], [256, 1, 1], s; x, xs as i32, w, st, y, n as i32, c as i32)?;
        launch!(self.conv_state, [c.div_ceil(256) as u32, 1, 1], [256, 1, 1], s; x, xs as i32, st, n as i32, c as i32)
    }

    /// The gated delta rule of a prompt chunk's n rows in two passes (`qx_gdn_prep`: every 64-token chunk's part that does not depend on the state, at once; `qx_gdn_scan`: the chunks in order with the state, a
    /// block a value head's 32 columns): `Kern35::delta`'s chunked rule, its arguments, the same arithmetic in another order of its sums.
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta2(
        &self,
        s: Stream,
        xc: u64,
        c: usize,
        ba: u64,
        sba: usize,
        a_log: u64,
        dt_bias: u64,
        state: u64,
        core: u64,
        n: usize,
        nk: usize,
        nv: usize,
    ) -> Result<()> {
        self.delta2_parts(s, xc, c, ba, sba, a_log, dt_bias, state, core, n, nk, nv, 3)
    }

    /// `delta2`'s passes: 1 the chunks' part, 2 the scan (on the scratch the last call left), 3 both (qxbench --gdn times them apart).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta2_parts(
        &self,
        s: Stream,
        xc: u64,
        c: usize,
        ba: u64,
        sba: usize,
        a_log: u64,
        dt_bias: u64,
        state: u64,
        core: u64,
        n: usize,
        nk: usize,
        nv: usize,
        parts: u8,
    ) -> Result<()> {
        if nk == 0 || nv % nk != 0 || n == 0 {
            return err(format!("the delta rule of {n} rows over {nv} value heads and {nk} key heads"));
        }
        let chunks = n.div_ceil(64);
        // (T''s two parts and M bf16 [64, 64] and G fp32 [64] a chunk and value head; the normed keys and queries bf16 [64, 128] a chunk and key head)
        let (mat, gb, kb) = (chunks * nv * 64 * 64 * 2, chunks * nv * 64 * 4, chunks * nk * 64 * 128 * 2);
        let need = 3 * mat + gb + 2 * kb;
        let mut slot = self.gdn_buf.borrow_mut();
        if slot.as_ref().is_none_or(|b| b.bytes < need) {
            *slot = None;
            *slot = Some(Buf::new(need)?);
        }
        let b = slot.as_ref().map_or(0, |b| b.ptr);
        let (th, tl, m, g) = (b, b + mat as u64, b + 2 * mat as u64, b + 3 * mat as u64);
        let (kn, qn) = (g + gb as u64, g + (gb + kb) as u64);
        if parts & 1 != 0 {
            launch!(self.gdn_prep, [chunks as u32, nv as u32, 1], [256, 1, 1], smem GDN_PREP_SMEM, s; xc, c as i32, ba, sba as i32, a_log, dt_bias, n as i32, nk as i32, nv as i32, th, tl, m, g, kn, qn)?;
        }
        crate::tmark!(s, "gdn: chunks' part");
        if parts & 2 == 0 {
            return Ok(());
        }
        launch!(self.gdn_scan, [nv as u32, 4, 1], [256, 1, 1], smem GDN_SCAN_SMEM, s; th, tl, m, g, kn, qn, xc, c as i32, state, core, n as i32, nk as i32, nv as i32)?;
        crate::tmark!(s, "gdn: scan");
        Ok(())
    }

    /// `attn_sel` for a prompt chunk's rows on the tensor cores (`qx_attn_tc`: 12 query heads to each of 2 key-value heads, a warp a row's key-value head).
    ///
    /// # Safety
    /// As `hc_norm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_tc(
        &self,
        s: Stream,
        q: u64,
        qs: usize,
        gate: u64,
        gs: usize,
        kc: u64,
        vc: u64,
        sel: u64,
        cnt: u64,
        tmax: usize,
        out: u64,
        os: usize,
        n: usize,
    ) -> Result<()> {
        let sl2 = (1.0f64 / 16.0 * std::f64::consts::LOG2_E) as f32;
        launch!(self.attn_tc, [n.div_ceil(2) as u32, 1, 1], [128, 1, 1], s; q, qs as i32, gate, gs as i32, kc, vc, sel, cnt, tmax as i32, out, os as i32, n as i32, sl2)
    }
}

/// `qx_attn_dec*`'s dynamic shared memory for w warps a block.
fn attn_dec_smem(w: u32) -> u32 {
    (w / 2) * 16384 + w * 32 * 4
}

/// The decode step's attention's warps a block (`qx_attn_dec*`): 8, or 4 where a block takes 64 KiB of shared memory at most (8 take 72.7 KiB: `dev::smem64`).
fn dec_wpb() -> usize {
    if crate::dev::smem64() {
        4
    } else {
        8
    }
}

/// The most pools whose keys `qx_select` holds in shared memory (a longer context's are read from the scores each pass); where a block takes 64 KiB at most
/// (`dev::smem64`) the fewer that leave room for the kernel's own 1,288 bytes.
const QX_SEL_CAP: usize = 16384;
const QX_SEL_CAP_64: usize = 16000;

/// The two-pass delta rule's dynamic shared memory (kernels/qwen4x.cu `qx_gdn_prep`, `qx_gdn_scan`).
const GDN_PREP_SMEM: u32 = 2 * 64 * 128 * 2 + (3 * 64 * 68 + 192) * 4;
const GDN_SCAN_SMEM: u32 = (2 * 64 * 128 + 3 * 64 * 64 + 2 * 128 * 32 + 4 * 64 * 32) * 2 + 64 * 4;

/// Rows of a tensor in a file mapped into memory (the embedding, a shard of the n-gram table): read in place, the system's page cache holding what was read.
pub struct HostRows {
    map: Arc<Mapping>,
    off: usize,
    pub rows: usize,
    pub row_bytes: usize,
}

impl HostRows {
    /// Tensor `name` of `ck` (bf16, 2-D), its file mapped through `maps` (one mapping a file).
    fn open(ck: &Checkpoint, name: &str, maps: &mut HashMap<PathBuf, Arc<Mapping>>) -> Result<HostRows> {
        HostRows::open_as(ck, name, maps, "BF16", 2)
    }

    /// As `open`, for a 2-D tensor of `dtype` with elements of `elem` bytes (the compressed n-gram table's rows are U8).
    fn open_as(ck: &Checkpoint, name: &str, maps: &mut HashMap<PathBuf, Arc<Mapping>>, dtype: &str, elem: usize) -> Result<HostRows> {
        let (f, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        if t.dtype != dtype || t.shape.len() != 2 {
            return err(format!("{name}: {} {:?}, where a {dtype} matrix is wanted", t.dtype, t.shape));
        }
        let map = match maps.get(&f.path) {
            Some(m) => m.clone(),
            None => {
                let m = Arc::new(Mapping::open(&f.path)?);
                maps.insert(f.path.clone(), m.clone());
                m
            }
        };
        let (rows, row_bytes) = (t.shape[0] as usize, t.shape[1] as usize * elem);
        if t.offset as usize + rows * row_bytes > map.len {
            return err(format!("{name}: past the end of {}", f.path.display()));
        }
        // (rows read at random: a prompt's n-gram rows with the system's read-ahead filled the page cache with ~128 KB of neighbours a row, tens of GB for a 32K prompt, and pushed the expert pack out of it)
        map.random(t.offset as usize, rows * row_bytes);
        Ok(HostRows { map, off: t.offset as usize, rows, row_bytes })
    }

    pub fn row(&self, r: usize) -> &[u8] {
        assert!(r < self.rows, "row {r} of {}", self.rows);
        // SAFETY: inside the mapping (checked at open), read-only, alive with `self`.
        unsafe { std::slice::from_raw_parts((self.map.ptr as *const u8).add(self.off + r * self.row_bytes), self.row_bytes) }
    }
}

/// The n-gram embedding's table and hashes (transformers' Qwen4ExpTextNGramEmbedding): `heads` hashed heads (bigrams then trigrams, `per_order` each), each row `row` bf16, the table in shards of `per_shard` rows.
pub struct Ngram {
    shards: Vec<HostRows>,
    /// The table at 4 bits (a Glyd slim checkpoint's `shard_N.q4`): a row is its scale (bf16) then `row` codes of 4 bits, low nibble first, each a level of NF4 times the scale.
    q4: bool,
    per_shard: usize,
    pub heads: usize,
    per_order: usize,
    pub row: usize,
    mult: [u64; 3],
    size: Vec<u64>,
    offset: Vec<u64>,
    eos: u32,
}

impl Ngram {
    fn load(ck: &Checkpoint, c: &ConfigQx, maps: &mut HashMap<PathBuf, Arc<Mapping>>) -> Result<Option<Ngram>> {
        let Some(p) = c.ple.as_ref() else { return Ok(None) };
        let m = format!("{}.ple.ple_embedding", c.layer_name(p.layer));
        let i64s = |name: &str| -> Result<Vec<u64>> {
            let b = ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?;
            Ok(b.as_chunks::<8>().0.iter().map(|x| i64::from_le_bytes(*x) as u64).collect())
        };
        let mult = i64s(&format!("{m}.layer_multipliers"))?;
        let (size, offset) = (i64s(&format!("{m}.ngram_heads_vocab_sizes"))?, i64s(&format!("{m}.ngram_heads_offsets"))?);
        let heads = (p.ngram - 1) * p.heads;
        if mult.len() != p.ngram || size.len() != heads || offset.len() != heads {
            return err(format!(
                "{m}: {} multipliers, {} sizes, {} offsets, where {} heads of {}-grams are wanted",
                mult.len(),
                size.len(),
                offset.len(),
                heads,
                p.ngram
            ));
        }
        let mut shards = Vec::new();
        let q4 = ck.get(&format!("{m}.ngram_embedding.shard_0.q4")).is_some();
        if q4 {
            while ck.get(&format!("{m}.ngram_embedding.shard_{}.q4", shards.len())).is_some() {
                shards.push(HostRows::open_as(ck, &format!("{m}.ngram_embedding.shard_{}.q4", shards.len()), maps, "U8", 1)?);
            }
        } else if ck.get(&format!("{m}.ngram_embedding.weight")).is_some() {
            shards.push(HostRows::open(ck, &format!("{m}.ngram_embedding.weight"), maps)?);
        } else {
            while ck.get(&format!("{m}.ngram_embedding.shard_{}.weight", shards.len())).is_some() {
                shards.push(HostRows::open(ck, &format!("{m}.ngram_embedding.shard_{}.weight", shards.len()), maps)?);
            }
        }
        let row = p.dim / heads;
        let per_shard = shards.first().map_or(0, |s| s.rows);
        let total: u64 = size.iter().sum();
        let row_bytes = if q4 { 2 + row / 2 } else { row * 2 };
        if shards.is_empty()
            || row % 2 != 0
            || shards.iter().any(|s| s.rows != per_shard || s.row_bytes != row_bytes)
            || ((shards.len() * per_shard) as u64) < total
            || offset.last().copied().unwrap_or(0) + size.last().copied().unwrap_or(0) != total
        {
            return err(format!("{m}: the table's shards are not {total} rows of {row} in shards alike"));
        }
        Ok(Some(Ngram { shards, q4, per_shard, heads, per_order: p.heads, row, mult: [mult[0], mult[1], mult[2]], size, offset, eos: c.eos_ngram }))
    }

    /// The table's rows of a position: its token `t0`, the token before `c1` and the one before that `c2` (the end token where the context is cut by one or missing).
    pub fn ids(&self, t0: u32, c1: u32, c2: u32) -> Vec<u64> {
        let bi = (t0 as u64).wrapping_mul(self.mult[0]) ^ (c1 as u64).wrapping_mul(self.mult[1]);
        let tri = bi ^ (c2 as u64).wrapping_mul(self.mult[2]);
        (0..self.heads).map(|h| (if h < self.per_order { bi } else { tri }) % self.size[h] + self.offset[h]).collect()
    }

    /// The contexts of each of `tokens` (after `ctx`, the two tokens before them, oldest first): (token, the one before, the one before that or the end token where the one before is the end token).
    pub fn contexts(&self, ctx: [u32; 2], tokens: &[u32]) -> Vec<(u32, u32, u32)> {
        let mut hist = vec![ctx[0], ctx[1]];
        hist.extend_from_slice(tokens);
        (0..tokens.len())
            .map(|i| {
                let (t0, c1, c2) = (hist[i + 2], hist[i + 1], hist[i]);
                (t0, c1, if c1 == self.eos { self.eos } else { c2 })
            })
            .collect()
    }

    /// The embedding rows of `tokens` (bf16 [n, heads row]) into `out`.
    pub fn gather(&self, ctx: [u32; 2], tokens: &[u32], out: &mut [u8]) {
        let (rb, cx) = (self.row * 2, self.contexts(ctx, tokens));
        fill_rows(&mut out[..tokens.len() * self.heads * rb], self.heads * rb, |i, o| {
            let (t0, c1, c2) = cx[i];
            for (h, id) in self.ids(t0, c1, c2).into_iter().enumerate() {
                let (s, r) = (id as usize / self.per_shard, id as usize % self.per_shard);
                if self.q4 {
                    q4_row(self.shards[s].row(r), &mut o[h * rb..(h + 1) * rb]);
                } else {
                    o[h * rb..(h + 1) * rb].copy_from_slice(self.shards[s].row(r));
                }
            }
        });
    }
}

/// NF4's levels (the quantiles of a normal distribution, scaled to [-1, 1]).
const NF4: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// A 4-bit row (bf16 scale, then a code a value, low nibble first) as bf16 values into `out`.
fn q4_row(q: &[u8], out: &mut [u8]) {
    let scale = f32::from_bits(((q[0] as u32) | ((q[1] as u32) << 8)) << 16);
    for (j, o) in out.chunks_exact_mut(2).enumerate() {
        let code = (q[2 + j / 2] >> ((j & 1) * 4)) & 15;
        let b = (NF4[code as usize] * scale).to_bits();
        let v = ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16;
        o.copy_from_slice(&v.to_le_bytes());
    }
}

/// `out` (items of `size` bytes) filled by `f(i, item i)`, by up to 128 threads for many items: a row of a mapped table not in the page cache is a read from the disk, and many reads in flight keep the disk busy (a 32K
/// prompt's 512K n-gram rows one at a time took ~20 s from the disk; a new prompt's by 32 threads 0.8-1.0 s, by 128 0.4-0.5 s).
fn fill_rows(out: &mut [u8], size: usize, f: impl Fn(usize, &mut [u8]) + Sync) {
    let n = out.len() / size;
    let threads = (n / 64).clamp(1, 128);
    if threads == 1 {
        out.chunks_mut(size).enumerate().for_each(|(i, o)| f(i, o));
        return;
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for (k, part) in out.chunks_mut(per * size).enumerate() {
            let f = &f;
            sc.spawn(move || part.chunks_mut(size).enumerate().for_each(|(i, o)| f(k * per + i, o)));
        }
    });
}

/// A Gated Residual's weights: the per-stream norm's [4 hidden], its bottleneck's two matrices, the inject matrix [4, 4 hidden] bf16 (none for the final mixer).
pub struct Gr {
    pub norm: Buf,
    pub w: GrW,
    pub inject: Option<Buf>,
}

/// The bottleneck's matrices: packed at the engine's layout (a lossless check's), or int8 rows with an fp32 scale each (`max |w| / 127`), the bottleneck padded with zeros to `r` (a multiple of 128): down [r, 4 hidden],
/// up [4 hidden, r]. The int8 rows are read whole by the fused kernels of a few rows (`qx_gr1`, `qx_gr2`) and made bf16 for a prompt chunk's dense products.
pub enum GrW {
    Mats(Mat, Mat),
    /// `r` the bottleneck's width as the checkpoint has it (the fused kernels read the rows as they are), `rp` padded to a multiple of 128 (a prompt chunk's dense products, made bf16 with zeros past `r`)
    Q {
        r: usize,
        rp: usize,
        dq: Buf,
        ds: Buf,
        uq: Buf,
        us: Buf,
    },
}

/// A matrix's rows as int8 with an fp32 scale each (a row of zeros: scale 1), `rows_pad` rows and `cols_pad` columns (the padding zeros).
pub(crate) fn quant_rows(w: &[u16], rows: usize, cols: usize, rows_pad: usize, cols_pad: usize) -> (Vec<i8>, Vec<f32>) {
    let (mut q, mut sc) = (vec![0i8; rows_pad * cols_pad], vec![1f32; rows_pad]);
    for r in 0..rows {
        let row: Vec<f32> = w[r * cols..(r + 1) * cols].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let m = row.iter().fold(0f32, |a, v| a.max(v.abs()));
        if m > 0.0 {
            sc[r] = m / 127.0;
            for (c, v) in row.iter().enumerate() {
                q[r * cols_pad + c] = (v / sc[r]).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, sc)
}

/// A Gated Residual's bottleneck read from the checkpoint as int8 rows (`GrW::Q`).
fn gr_q(ck: &Checkpoint, module: &str, w: usize, rank: usize) -> Result<GrW> {
    if rank % 16 != 0 {
        return err(format!("{module}: a bottleneck of {rank}, which this engine does not read"));
    }
    let (r, rp) = (rank, rank.next_multiple_of(128));
    let read = |name: &str, n: usize| -> Result<Vec<u16>> {
        let mut v = vec![0u16; n];
        ck.read_bf16(name, &mut v).map_err(|e| Error(format!("{name}: {e}")))?;
        Ok(v)
    };
    let (down, up) = (read(&format!("{module}.input_mix_weight_down.weight"), rank * w)?, read(&format!("{module}.input_mix_weight_up.weight"), w * rank)?);
    let (dq, ds) = quant_rows(&down, rank, w, r, w);
    let (uq, us) = quant_rows(&up, w, rank, w, r);
    let bytes = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    Ok(GrW::Q {
        r,
        rp,
        dq: upload_bytes(&bytes(&dq))?,
        ds: upload_bytes(crate::weights::as_bytes(&ds))?,
        uq: upload_bytes(&bytes(&uq))?,
        us: upload_bytes(crate::weights::as_bytes(&us))?,
    })
}

/// PLE's weights: the key and value products as one, the three norms [4 hidden], the convolution [4 hidden, 4].
pub struct PleW {
    pub kv: Mat,
    pub nk: Buf,
    pub nq: Buf,
    pub nc: Buf,
    pub conv: Buf,
}

/// A layer: the Gated Residuals of its two blocks, its token mixer's matrices (GDN: q, k, v and z as one, b and a as one, out; attention: q, k and v, o, the indexer's) and small tensors (GDN: the convolution
/// [conv dim, 4] bf16, A_log fp32, dt_bias bf16, the gated norm's weight bf16; attention: the q and k norms, the indexer's q and k norms, bf16), PLE where it has it, its block of experts.
pub struct LayerQx {
    pub kind: QxKind,
    pub idx: usize,
    pub gr: [Gr; 2],
    pub mats: Vec<Mat>,
    pub small: Vec<Buf>,
    pub ple: Option<PleW>,
    pub moe: Option<Box<dyn MoeBlockQx>>,
}

/// A mixture-of-experts layer's block: `h` [n, hidden] the block's input, `out` [n, hidden] its output.
pub trait MoeBlockQx {
    /// # Safety
    /// Live device buffers of those sizes, used on the engine's stream.
    unsafe fn forward(&self, e: &EngineQx, h: u64, n: usize, out: u64) -> Result<()>;
    fn graphable(&self) -> bool {
        false
    }
    fn bytes(&self) -> usize {
        0
    }
    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        Vec::new()
    }
    /// Whether a decode step can run it in three parts (`front`, `host`, `back`; the GPU's two in CUDA graphs: `EngineQx`'s segmented decode).
    fn segmented(&self) -> bool {
        false
    }
    /// Its buffers made right for a step in parts (before a capture).
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn prepare(&self, _: &EngineQx) -> Result<()> {
        Ok(())
    }
    /// The GPU work up to the host's wait (the router, the rotation, the choices on their way to the host). Captured in a graph.
    ///
    /// # Safety
    /// As `forward`; `n` at most `SPEC_ROWS`.
    unsafe fn front(&self, _: &EngineQx, _: u64, _: usize) -> Result<()> {
        err("this block does not run in parts")
    }
    /// The host's part for the n rows of the front: the wait for the choices, the plan, the copies, the products of the experts in places, the host's pairs and their sums sent up. Not captured.
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn host(&self, _: &EngineQx, _n: usize) -> Result<()> {
        err("this block does not run in parts")
    }
    /// `host`, then `next` (the GPU's work after it: the next part's graph); a block may queue `next` before its host's part is done (`QxMoeCb`: the back's graph waits for the host's sums on the GPU).
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn host_then(&self, e: &EngineQx, n: usize, next: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        self.host(e, n)?;
        next()
    }
    /// The GPU work after the host's part: the sum and the shared expert's addition into `out`. Captured in a graph.
    ///
    /// # Safety
    /// As `forward`; `n` as the front's.
    unsafe fn back(&self, _: &EngineQx, _: u64, _: usize) -> Result<()> {
        err("this block does not run in parts")
    }
    /// The buffers of a prompt taken layer by layer freed (after it: the next one makes them again).
    ///
    /// # Safety
    /// As `forward`.
    fn whole_release(&self, _: &EngineQx) -> Result<()> {
        Ok(())
    }
    /// The experts the last prompt taken layer by layer chose most placed (hints to the cache), so that the decode starts with them.
    fn rewarm(&self) -> Result<()> {
        Ok(())
    }
    /// The copies its store has queued done (the host waits), so that `rewarm`'s are finished before the decode's first step.
    fn drain(&self) -> Result<()> {
        Ok(())
    }
    /// Before a step in parts (no front of it queued yet): the places its front's graph reads made the cache's where they are not (`QxMoeCb`'s map).
    fn sync_map(&self) -> Result<()> {
        Ok(())
    }
    /// Its places on the GPU freed, their experts forgotten (for a long prompt's buffers, `EngineQx::borrow_places`): the bytes freed. The block takes no decode call until `restore_places`.
    fn release_places(&self) -> Result<usize> {
        Ok(0)
    }
    /// Its places made again (as many as before), empty.
    fn restore_places(&self) -> Result<()> {
        Ok(())
    }
    /// Its places as many now, the most (as loaded), and the bytes of one (none: 0, 0, 0).
    fn place_slots(&self) -> (usize, usize, usize) {
        (0, 0, 0)
    }
    /// Its places made again for `slots` experts (1 to as loaded), the experts in the first of them kept (`EngineQx::resize_places`).
    fn resize_places(&self, _e: &EngineQx, _slots: usize) -> Result<()> {
        Ok(())
    }
    /// `back` and the streams' update after it (x += out inj) in one: by default the two, a block may fuse them (`QxMoeCb`: one launch).
    ///
    /// # Safety
    /// As `forward`; `n` as the front's.
    unsafe fn back_post(&self, e: &EngineQx, n: usize, x: u64, inj: u64) -> Result<()> {
        self.back(e, e.o.ptr, n)?;
        e.kq.hc_post(e.dev.stream, x, e.o.ptr, inj, n, e.w.cfg.hidden)
    }
    /// Whether it takes a prompt layer by layer (`EngineQx::prefill` with `OptsQx::whole`): `whole_front` on each chunk of the prompt's rows, `whole_experts` once for them all (each expert read once for the prompt),
    /// `whole_back` on each chunk.
    fn whole(&self) -> bool {
        false
    }
    /// The router, the experts' input and the shared expert of rows `at` .. `at + n` of the prompt (their input at `h`), kept for `whole_experts`.
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn whole_front(&self, _: &EngineQx, _h: u64, _n: usize, _at: usize) -> Result<()> {
        err("this block takes no prompt layer by layer")
    }
    /// The routed experts of the prompt's first `rows` rows (every `whole_front` before it).
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn whole_experts(&self, _: &EngineQx, _rows: usize) -> Result<()> {
        err("this block takes no prompt layer by layer")
    }
    /// The block's output of rows `at` .. `at + n` into `out` (after `whole_experts`).
    ///
    /// # Safety
    /// As `forward`.
    unsafe fn whole_back(&self, _: &EngineQx, _out: u64, _n: usize, _at: usize) -> Result<()> {
        err("this block takes no prompt layer by layer")
    }
}

/// Makes layer `l`'s block of experts (the layer, the config, the checkpoint, the engine's layout).
pub type MoeFactoryQx<'a> = &'a dyn Fn(usize, &ConfigQx, &Checkpoint, Layout) -> Result<Box<dyn MoeBlockQx>>;

/// The model's weights: on the GPU the layers asked for (`only`; None: all, with the final mixer and the LM head), on the host the embedding and the n-gram table mapped.
pub struct WeightsQx {
    pub cfg: ConfigQx,
    pub layout: Layout,
    pub layers: Vec<Option<LayerQx>>,
    pub mixer: Option<Gr>,
    pub head: Option<PackBuf>,
    pub embed: HostRows,
    pub ngram: Option<Ngram>,
    /// The MTP head, where it was asked for.
    pub mtp: Option<MtpQx>,
    pub bytes: usize,
}

/// The MTP head (1 layer, `mtp.*`; vLLM's qwen4_exp/nvidia/mtp.py has the structure, transformers none): the next token's embedding normed (1 + w) and projected (`fc_embedding`), the main model's last streams [4, hidden] normed
/// as one vector of 4 hidden (1 + w) and each stream projected by the one `fc_hidden`; the embedding's projection added to every stream, then a QSA layer with its two Gated Residuals and its experts (the pack's layer 48), then
/// its final mixer into the model's LM head. Its streams after the layer are the next draft step's input in place of the main model's.
pub struct MtpQx {
    pub ne: Buf,
    pub nh: Buf,
    pub fe: Mat,
    pub fh: Mat,
    pub layer: LayerQx,
    pub mixer: Gr,
}

/// A tensor's bf16 bytes uploaded (a norm's weight, a small matrix).
fn tensor(ck: &Checkpoint, name: &str) -> Result<Buf> {
    upload_bytes(&ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?)
}

impl WeightsQx {
    pub fn load(dir: &Path, threads: usize, layout: Layout, only: Option<&[usize]>, moe: Option<MoeFactoryQx<'_>>, with_mtp: bool) -> Result<WeightsQx> {
        WeightsQx::load_mixed(dir, threads, layout, None, only, only.is_none(), moe, with_mtp)
    }

    /// `load` with the attention layers' matrices (the MTP head's too) and the LM head at `hi` where it is given (the rest at `layout`): the long context's precision kept where a lower layout costs it; the final mixer
    /// and the LM head where `head`.
    #[allow(clippy::too_many_arguments)]
    pub fn load_mixed(
        dir: &Path,
        threads: usize,
        layout: Layout,
        hi: Option<Layout>,
        only: Option<&[usize]>,
        head: bool,
        moe: Option<MoeFactoryQx<'_>>,
        with_mtp: bool,
    ) -> Result<WeightsQx> {
        let cfg = ConfigQx::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let mut wanted: Vec<usize> = only.map_or_else(|| (0..cfg.layers).collect(), |l| l.to_vec());
        wanted.sort_unstable();
        wanted.dedup();
        if let Some(&l) = wanted.iter().find(|&&l| l >= cfg.layers) {
            return err(format!("layer {l}, where the model has {}", cfg.layers));
        }
        let whole = head;
        // (the Gated Residuals' bottlenecks as int8 rows, except in a lossless check's layout)
        let quant_gr = layout != Layout::Tiered;
        // the matrices: packed where the packer takes them (on `threads` workers), else bf16 as they are
        let mut items: Vec<Pack35> = Vec::new();
        for &l in &wanted {
            let p = cfg.layer_name(l);
            let packs = cfg.layer_packs(l);
            items.extend(packs[..packs.len() - 2].iter().cloned());
            if !quant_gr {
                items.extend(cfg.gr_packs(&format!("{p}.attn_hyper_connection")));
                items.extend(cfg.gr_packs(&format!("{p}.mlp_hyper_connection")));
            }
            if cfg.ple.as_ref().is_some_and(|x| x.layer == l) {
                items.extend(cfg.ple_pack());
            }
        }
        if whole {
            if !quant_gr {
                items.extend(cfg.gr_packs(&cfg.mixer_name(false)));
            }
            items.push(Pack35 { key: "lm_head".into(), parts: vec![("lm_head.weight".into(), cfg.vocab)], cols: cfg.hidden });
        }
        let with_mtp = with_mtp && cfg.mtp_layers > 0;
        if with_mtp {
            let (l, p) = (cfg.layers, cfg.layer_name(cfg.layers));
            let packs = cfg.layer_packs(l);
            items.extend(packs[..packs.len() - 2].iter().cloned());
            if !quant_gr {
                items.extend(cfg.gr_packs(&format!("{p}.attn_hyper_connection")));
                items.extend(cfg.gr_packs(&format!("{p}.mlp_hyper_connection")));
                items.extend(cfg.gr_packs(&cfg.mixer_name(true)));
            }
            for f in ["fc_embedding", "fc_hidden"] {
                items.push(Pack35 { key: format!("mtp.{f}"), parts: vec![(format!("mtp.{f}.weight"), cfg.hidden)], cols: cfg.hidden });
            }
        }
        let (packable, dense): (Vec<Pack35>, Vec<Pack35>) = items.into_iter().partition(Pack35::packable);
        // (the keys of the attention layers' matrices and the LM head, where they go at `hi`)
        let mut hi_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        if hi.is_some() {
            for l in (0..cfg.layers).chain(with_mtp.then_some(cfg.layers)) {
                if l == cfg.layers || cfg.kinds[l] == QxKind::Attn {
                    let packs = cfg.layer_packs(l);
                    hi_keys.extend(packs[..packs.len() - 2].iter().map(|p| p.key.clone()));
                }
            }
            hi_keys.insert("lm_head".into());
        }
        let level_of = |pk: &Pack35| hi.filter(|_| hi_keys.contains(&pk.key)).unwrap_or(layout);
        let mut packed: HashMap<String, PackBuf> = HashMap::new();
        ordered(packable.len(), threads, 2 * threads.max(1), &|k| read_pack(&ck, None, &packable[k], level_of(&packable[k])), |k, p| {
            crate::status::STATUS.beat();
            packed.insert(packable[k].key.clone(), upload_packed(p)?);
            Ok(())
        })?;
        let mut bytes: usize = packed.values().map(|p| p.bytes).sum();
        let mut mat = |pk: &Pack35| -> Result<Mat> {
            if let Some(p) = packed.remove(&pk.key) {
                return Ok(Mat::Packed(p));
            }
            if !dense.iter().any(|d| d.key == pk.key) {
                return err(format!("{}: neither packed nor kept", pk.key));
            }
            let mut w = vec![0u16; pk.rows() * pk.cols];
            let mut at = 0;
            for (name, rows) in &pk.parts {
                ck.read_bf16(name, &mut w[at..at + rows * pk.cols]).map_err(|e| Error(format!("{name}: {e}")))?;
                at += rows * pk.cols;
            }
            let b = upload_bytes(crate::weights::as_bytes(&w))?;
            Ok(Mat::Dense(b, pk.rows(), pk.cols))
        };
        let gr = |mat: &mut dyn FnMut(&Pack35) -> Result<Mat>, module: &str, inject: bool| -> Result<Gr> {
            let packs = cfg.gr_packs(module);
            let w = if quant_gr { gr_q(&ck, module, cfg.hc * cfg.hidden, cfg.hc_rank)? } else { GrW::Mats(mat(&packs[0])?, mat(&packs[1])?) };
            Ok(Gr {
                norm: tensor(&ck, &format!("{module}.hc_norm.weight"))?,
                w,
                inject: if inject { Some(tensor(&ck, &format!("{module}.block_inject_weight.weight"))?) } else { None },
            })
        };
        let schedule = cfg.schedule();
        let mut layers: Vec<Option<LayerQx>> = (0..cfg.layers).map(|_| None).collect();
        for &l in &wanted {
            crate::status::STATUS.beat();
            let (kind, idx) = schedule[l];
            let p = cfg.layer_name(l);
            let packs = cfg.layer_packs(l);
            let mats = packs[..packs.len() - 2].iter().map(&mut mat).collect::<Result<Vec<Mat>>>()?;
            let grs = [gr(&mut mat, &format!("{p}.attn_hyper_connection"), true)?, gr(&mut mat, &format!("{p}.mlp_hyper_connection"), true)?];
            let small = match kind {
                QxKind::Gdn => {
                    let a = format!("{p}.linear_attn");
                    vec![
                        tensor(&ck, &format!("{a}.conv1d.weight"))?,
                        upload_bytes(&f32_bytes(&ck, &format!("{a}.A_log"))?)?,
                        tensor(&ck, &format!("{a}.dt_bias"))?,
                        tensor(&ck, &format!("{a}.norm.weight"))?,
                    ]
                }
                QxKind::Attn => {
                    let a = format!("{p}.self_attn");
                    vec![
                        tensor(&ck, &format!("{a}.q_norm.weight"))?,
                        tensor(&ck, &format!("{a}.k_norm.weight"))?,
                        tensor(&ck, &format!("{a}.indexer.q_layernorm.weight"))?,
                        tensor(&ck, &format!("{a}.indexer.k_layernorm.weight"))?,
                    ]
                }
            };
            let ple = match cfg.ple_pack().filter(|_| cfg.ple.as_ref().is_some_and(|x| x.layer == l)) {
                Some(pk) => {
                    let m = format!("{p}.ple");
                    Some(PleW {
                        kv: mat(&pk)?,
                        nk: tensor(&ck, &format!("{m}.norm_key.weight"))?,
                        nq: tensor(&ck, &format!("{m}.norm_query.weight"))?,
                        nc: tensor(&ck, &format!("{m}.norm_conv.weight"))?,
                        conv: tensor(&ck, &format!("{m}.conv1d.weight"))?,
                    })
                }
                None => None,
            };
            let block = match &moe {
                Some(make) => Some(make(l, &cfg, &ck, layout)?),
                None => None,
            };
            bytes += small.iter().map(|b| b.bytes).sum::<usize>()
                + grs.iter().map(|g| g.norm.bytes + g.inject.as_ref().map_or(0, |b| b.bytes)).sum::<usize>()
                + block.as_ref().map_or(0, |b| b.bytes());
            layers[l] = Some(LayerQx { kind, idx, gr: grs, mats, small, ple, moe: block });
        }
        let (mixer, head) = if whole {
            let mixer = gr(&mut mat, &cfg.mixer_name(false), false)?;
            let head = match mat(&Pack35 { key: "lm_head".into(), parts: vec![("lm_head.weight".into(), cfg.vocab)], cols: cfg.hidden })? {
                Mat::Packed(p) => p,
                Mat::Dense(..) => return err("the LM head is not packable"),
            };
            (Some(mixer), Some(head))
        } else {
            (None, None)
        };
        let mtp = if with_mtp {
            let (l, p) = (cfg.layers, cfg.layer_name(cfg.layers));
            let packs = cfg.layer_packs(l);
            let mats = packs[..packs.len() - 2].iter().map(&mut mat).collect::<Result<Vec<Mat>>>()?;
            let grs = [gr(&mut mat, &format!("{p}.attn_hyper_connection"), true)?, gr(&mut mat, &format!("{p}.mlp_hyper_connection"), true)?];
            let a = format!("{p}.self_attn");
            let small = vec![
                tensor(&ck, &format!("{a}.q_norm.weight"))?,
                tensor(&ck, &format!("{a}.k_norm.weight"))?,
                tensor(&ck, &format!("{a}.indexer.q_layernorm.weight"))?,
                tensor(&ck, &format!("{a}.indexer.k_layernorm.weight"))?,
            ];
            let block = match &moe {
                Some(make) => Some(make(l, &cfg, &ck, layout)?),
                None => None,
            };
            let fc = |f: &str| Pack35 { key: format!("mtp.{f}"), parts: vec![(format!("mtp.{f}.weight"), cfg.hidden)], cols: cfg.hidden };
            let (fe, fh) = (mat(&fc("fc_embedding"))?, mat(&fc("fc_hidden"))?);
            let layer = LayerQx { kind: QxKind::Attn, idx: cfg.layers_of(QxKind::Attn), gr: grs, mats, small, ple: None, moe: block };
            Some(MtpQx {
                ne: tensor(&ck, "mtp.pre_fc_norm_embedding.weight")?,
                nh: tensor(&ck, "mtp.pre_fc_norm_hidden.weight")?,
                fe,
                fh,
                layer,
                mixer: gr(&mut mat, &cfg.mixer_name(true), false)?,
            })
        } else {
            None
        };
        if let Some(k) = packed.keys().next() {
            return err(format!("a pack the forward pass does not take: {k}"));
        }
        let mut maps = HashMap::new();
        let embed = HostRows::open(&ck, &format!("{}embed_tokens.weight", cfg.prefix), &mut maps)?;
        let ngram = Ngram::load(&ck, &cfg, &mut maps)?;
        Ok(WeightsQx { cfg, layout, layers, mixer, head, embed, ngram, mtp, bytes })
    }
}

/// How a run is set up.
pub struct OptsQx {
    /// Positions a sequence may reach.
    pub ctx: usize,
    /// Tokens a prompt chunk takes at most.
    pub chunk: usize,
    pub device: i32,
    pub layout: Layout,
    /// A prompt's delta rule in chunks (else token by token, as a decode step). Never where a block takes 64 KiB of shared memory at most (`dev::smem64`): the chunked
    /// kernels take 85.8 to 97 KiB, the rule token by token 1 KiB.
    pub chunked: bool,
    /// Load just these layers (a development check's); None: the model.
    pub only: Option<Vec<usize>>,
    /// Room to verify a draft of up to `SPEC_ROWS - 1` tokens in one pass and to rewind the GDN layers and PLE to where it was right (`EngineQx::verify`, `commit`).
    pub spec: bool,
    /// Load the MTP head too (`EngineQx::generate_mtp`; implies `spec`).
    pub mtp: bool,
    /// Rows of a prompt taken layer by layer at most (each layer's experts read once for them; a longer prompt in parts of this many); 0: every prompt in chunks through all the layers.
    pub whole: usize,
    /// The attention layers' matrices and the LM head at this layout instead of `layout` (None: all at `layout`).
    pub hi: Option<Layout>,
    /// A part of the model split over GPUs: its layers lo..hi (the ones before on the GPU of its `front`, `EngineQx::with_front`; the final mixer, the LM head and the MTP head with the last layer); None: the whole model.
    pub span: Option<(usize, usize)>,
    /// Its GPU in a context of its own, not the GPU's primary one: a part after the first on the same GPU (the split's check on one GPU, `--gpus 0,0`).
    pub own_ctx: bool,
    /// Its streams go on to the next part GPU to GPU, not through pinned host memory (`Hop`).
    pub peer: bool,
}

impl Default for OptsQx {
    fn default() -> OptsQx {
        OptsQx {
            ctx: 4096,
            chunk: 1024,
            device: 0,
            layout: Layout::Int5,
            chunked: true,
            only: None,
            spec: false,
            mtp: false,
            whole: 0,
            hi: None,
            span: None,
            own_ctx: false,
            peer: false,
        }
    }
}

/// How a part of a model split over GPUs sends its streams (bf16 [rows, 4 hidden]) on to the next part: copied out to pinned host memory every context takes (`buf`) and in on the next part's GPU, or GPU to GPU
/// (`peer`); `ev` is recorded on this part's stream after the copy out (or its last kernel), and the next part's stream waits for it.
struct Hop {
    buf: Pinned,
    ev: crate::dev::Ev,
    peer: bool,
}

/// Rows the hop's buffer holds at least (a prompt taken layer by layer goes through it in pieces of this many: 20 MB).
const HOP_ROWS: usize = 1024;

/// What a prompt taken layer by layer made on a part of the model and the parts before it: the layers whose places it lent, the engine's own chunk-sized buffers it swapped out (`EngineQx::prompt_begin`).
struct PromptBufs {
    lent: Vec<usize>,
    grown: Option<ChunkSet>,
    front: Option<Box<PromptBufs>>,
}

/// A development check's look at each layer after it ran: (the engine, the layer, its rows, the tokens before them).
pub type TraceQx = Box<dyn Fn(&EngineQx, usize, usize, usize)>;

pub struct EngineQx {
    pub dev: Dev,
    pub lib: Lib,
    pub(crate) kern: Kern,
    pub(crate) k35: Kern35,
    /// The attention layer kind's kernels (the gated attention's, as the Qwen3.5 layout runs them).
    ka: crate::dynamic::attn::KernAttn,
    pub(crate) kg: KernGlm,
    pub(crate) kq: KernQx,
    skinny: crate::skinny::SkinnyKern,
    /// (shared by the parts of a model split over GPUs)
    pub skinny_on: std::rc::Rc<Cell<bool>>,
    pub w: WeightsQx,
    /// The layers this engine runs (`OptsQx::span`; all of them where the model is not split), the part of the model before them on another GPU (it stages the token rows; its streams come here), and where this
    /// part is not the last, how its streams go on (`Hop`).
    span: (usize, usize),
    pub front: Option<Box<EngineQx>>,
    hop: Option<Hop>,
    chunk: usize,
    /// rows a chunk of a prompt taken layer by layer has now (`chunk`, or `PREFILL_CHUNK` while its set is in), and the library's workspace that set needs
    pchunk: Cell<usize>,
    ws_prefill: usize,
    /// rows of a prompt taken layer by layer (`OptsQx::whole`), and whether it is on; the buffers of the prompt being read so (made for it, freed after it)
    whole: usize,
    wb: std::cell::RefCell<Option<WholeRows>>,
    pub whole_on: Cell<bool>,
    /// positions the sequence in the engine may reach (another's in its place: `swap_seq`), and the most any may (the rotary tables', the scores')
    ctx: usize,
    ctx_max: usize,
    pub chunked: bool,
    // activations, `chunk` rows each
    pub x: Buf,
    pub xn: Buf,
    pub gw: Buf,
    pub lo: Buf,
    pub injl: Buf,
    /// the fused Gated Residual's 1 / rms of each stream of up to 8 rows
    rs: Buf,
    pub inj: [Buf; 2],
    pub h: Buf,
    pub o: Buf,
    pub mix: Buf,
    pub ba: Buf,
    pub xc: Buf,
    pub core: Buf,
    pub qo: Buf,
    pub ixqk: Buf,
    pub ixq: Buf,
    pub scores: Buf,
    pub sel: Buf,
    pub cnt: Buf,
    part: Buf,
    ones: Buf,
    pub ple_e: Buf,
    cos: Buf,
    sin: Buf,
    pub hlast: Buf,
    logits: Buf,
    tok: Buf,
    val: Buf,
    pub pos: Buf,
    // the products
    ws: Buf,
    done: Buf,
    blas: Option<crate::ffi::Blas>,
    own: bool,
    scratch_w: Buf,
    gpu_code: i64,
    // what a sequence holds: the GDN states (fp32 [GDN layers, nv, 128, 128]) and convolutions' (bf16 [GDN layers, 3, conv dim]); the attention layers' keys and values (bf16 [layers, ctx, kv heads, 256]), the
    // indexer's raw keys (bf16 [layers, ctx, 128]) and block keys (bf16 [layers, ctx / 4, 128]); PLE's convolution (bf16 [9, 4 hidden])
    pub ssm: Buf,
    pub convst: Buf,
    pub kc: Buf,
    pub vc: Buf,
    pub ik: Buf,
    pub pk: Buf,
    pub plest: Buf,
    /// pinned staging: a chunk's embedding rows, then its n-gram rows; then the step's position
    host: Pinned,
    /// Tokens in the sequence, and the last two of them (the n-gram's context).
    pub n: usize,
    ng_ctx: [u32; 2],
    pub(crate) moe_scratch: OnceCell<MoeScratch>,
    pub trace: Option<TraceQx>,
    pub work_bytes: usize,
    /// The decode step in parts (`forward_segmented`): the graphs of the stretches of GPU work between the host's waits (made at the first step: a layer's by the layer's number + 1, 0 the stretch before the first), and
    /// whether a step runs so.
    segs: std::cell::RefCell<Vec<Option<crate::dev::Graph>>>,
    pub segmented: Cell<bool>,
    /// What a verified draft keeps (`OptsQx::spec`), whether the pass in flight saves its GDN and PLE inputs, and the rows of the verified pass that `commit` has not yet settled.
    spec: Option<SpecQx>,
    save: Cell<bool>,
    pending: Cell<Option<usize>>,
    /// The MTP head's inputs: the main model's last streams of a pass's rows, the head's own streams of its last draft step, and the unit weights its embedding's projection is spread over the streams with.
    hs: Buf,
    mtp_prev: Buf,
    /// The MTP head's draft from the LM head's first `draft_vocab` rows and its last `DRAFT_TAIL` (the special tokens) alone (0: all of them): the tokens' ids are about in order of how common they are, so a draft is
    /// almost always among them, for a fraction of the head's bytes; `dlg` their logits (bf16 [vocab], -inf between, made at the first draft).
    pub draft_vocab: Cell<usize>,
    dlg: std::cell::RefCell<Option<Buf>>,
    /// The drafts of an MTP pass beyond the first only while the head is this sure of all of them (its top tokens' probabilities multiplied; 0: always the pass's number): a draft that leaves them below it
    /// ends the pass's drafts, left out (`mtp_pass`); `draft_p` the last draft's probability and `dpb` its buffer.
    pub draft_min_p: Cell<f32>,
    /// A development census of the next drafted generation's passes alone (`mtp_start` begins it, after the prompt).
    pub kprof_passes: Cell<bool>,
    /// Where a generation's tokens go as they are made (a streamed request's): called with each pass's new tokens.
    pub emit: std::cell::RefCell<Option<Box<dyn FnMut(&[u32])>>>,
    /// Set (by `emit`'s sink, a client that went away) to end a drafted generation after the pass it is in; the state keeps the tokens made but the last, as at any end.
    pub halt: std::rc::Rc<Cell<bool>>,
    /// Tokens of the next drafted generation's prompt that the engine's state already holds (its first `keep`, as the last request left
    /// them: a conversation's earlier turns), so only the rest is read; 0: a fresh start. Taken by `generate_mtp`.
    pub keep: Cell<usize>,
    /// A request's sampler (temperature, top-k, top-p, min-p, penalties, seed): the tokens drawn from each row's logits, a draft's guess kept only
    /// where the draw is that token (the draws the same as without drafts); None: greedy.
    pub sampler: std::cell::RefCell<Option<crate::sampler::Sampler>>,
    /// The candidates of up to SPEC_ROWS rows for the sampler: the rows' TOPK largest logits (tokens, values) and their sums, and the host's copy.
    samp: (Buf, Buf, Buf, Pinned),
    /// Where the first row's grammar mask is put on the GPU (the later rows of a step follow tokens that are not chosen yet).
    maskgpu: Option<crate::kern::MaskGpu>,
    draft_p: Cell<f32>,
    dpb: std::cell::RefCell<Option<Buf>>,
    ones4: Buf,
    /// A prompt's images (`set_images`): the vision tower's rows of each (bf16 [tokens, hidden]) at the row its tokens start at, staged in place of the image token's embedding.
    img_rows: std::cell::RefCell<Vec<(usize, Vec<u16>)>>,
    /// The row from which the rotary's tables hold a prompt's M-RoPE positions in place of the plain rotary's (`set_images`).
    rope_from: Cell<Option<usize>>,
}

#[cfg(feature = "dev")]
thread_local! {
    /// A development check's look at a prompt taken layer by layer (`forward_whole`): the positions whose streams are kept after every layer, and the rows kept (bf16 [layers, positions, 4 hidden] for each part of
    /// the prompt in turn, the positions of the part in the order asked).
    pub static PEEK: std::cell::RefCell<Option<(Vec<usize>, Vec<u16>)>> = const { std::cell::RefCell::new(None) };
}

/// Rows a chunk of a prompt taken layer by layer has (its buffers made for the prompt: `EngineQx::grow_chunk`); the engine's own chunk (decode, verification, short prompts) keeps its 1,024. At 4,096 the products of a
/// chunk run fuller: a 32K prompt 8.42 -> 7.98 s.
pub const PREFILL_CHUNK: usize = 4096;
/// The most rows of a prompt the delta rule's two passes take a call (the scratch of its chunks' part: ~27 MB).
const GDN_SLICE: usize = 1024;
/// The rows the experts' front takes at once in a prompt taken layer by layer, at least (`EngineQx::front_rows`).
pub const FRONT_ROWS: usize = 1024;

/// The buffers of a chunk's rows the layers work in (the engine's for `chunk` rows; a prompt taken layer by layer swaps in a set for `PREFILL_CHUNK` and back): the Gated Residual's norm, mixing weights, bottleneck
/// and inject logits, the mixer's stream weights, the block's input and output, the mixers' products and states' inputs, the indexer's queries, scores and picks, the library's workspace.
struct ChunkSet {
    xn: Buf,
    gw: Buf,
    lo: Buf,
    injl: Buf,
    inj0: Buf,
    h: Buf,
    o: Buf,
    mix: Buf,
    ba: Buf,
    xc: Buf,
    core: Buf,
    qo: Buf,
    ixqk: Buf,
    ixq: Buf,
    scores: Buf,
    sel: Buf,
    cnt: Buf,
    ws: Buf,
}

impl ChunkSet {
    /// Bytes of each buffer for `ch` rows (positions up to `ctx`; the library's workspace aside).
    fn sizes(c: &ConfigQx, ch: usize, ctx: usize) -> [usize; 17] {
        let (hid, hw) = (c.hidden, c.hc * c.hidden);
        let act = |cols: usize| ch * cols * 2;
        let sparse = ctx > c.ix_tokens();
        [
            act(hw),
            act(hw),
            act(c.hc_rank.next_multiple_of(128)),
            act(c.hc),
            ch * c.hc * 4,
            act(hid),
            act(hid),
            act((c.conv_dim() + c.value_dim()).max(c.qkv_dim()).max(5 * hid).max(2 * c.shared_inter)),
            act(2 * c.nv),
            act(c.conv_dim()),
            act(c.value_dim().max(c.heads * c.head_dim)),
            act(c.heads * c.head_dim),
            act(c.ix_row()),
            act(c.ix_heads * c.ix_dim),
            if sparse { ch * (ctx / c.ix_block) * 4 } else { 16 },
            ch.max(1) * c.ix_tokens() * 4,
            ch * 4,
        ]
    }

    fn new(c: &ConfigQx, ch: usize, ctx: usize, ws: usize) -> Result<ChunkSet> {
        let z = Self::sizes(c, ch, ctx);
        // (the zeroed ones as the engine made them: the stream weights, the indexer's)
        Ok(ChunkSet {
            xn: Buf::new(z[0])?,
            gw: Buf::new(z[1])?,
            lo: Buf::new(z[2])?,
            injl: Buf::new(z[3])?,
            inj0: Buf::zeroed(z[4])?,
            h: Buf::new(z[5])?,
            o: Buf::new(z[6])?,
            mix: Buf::new(z[7])?,
            ba: Buf::new(z[8])?,
            xc: Buf::new(z[9])?,
            core: Buf::new(z[10])?,
            qo: Buf::new(z[11])?,
            ixqk: Buf::new(z[12])?,
            ixq: Buf::new(z[13])?,
            scores: Buf::zeroed(z[14])?,
            sel: Buf::zeroed(z[15])?,
            cnt: Buf::zeroed(z[16])?,
            ws: Buf::zeroed(ws.max(16))?,
        })
    }

    /// Device bytes of a set for `ch` rows.
    fn bytes(c: &ConfigQx, ch: usize, ctx: usize, ws: usize) -> usize {
        Self::sizes(c, ch, ctx).iter().sum::<usize>() + ws
    }
}

/// Rows a verified draft takes at most: the last token and up to seven guesses.
pub const SPEC_ROWS: usize = 8;

/// `qx_gr1s`'s scratch: slices of 256 columns of 4 streams (a width of 4,096 at most) by groups of 32 rows (a bottleneck of 540 at most, with the inject's 4).
const GR1S_SLICES: usize = 64;
const GR1S_GROUPS: usize = 17;
/// `qx_gr1q`'s groups of rows at most (its counters; its sums and norms in gr1s's scratch).
const GR1Q_GROUPS: usize = 256;

/// The LM head's last rows a draft from part of it keeps (`EngineQx::draft_vocab`): the special tokens'.
pub const DRAFT_TAIL: usize = 512;

/// The model's end token (the n-gram context's start).
fn eos_of(c: &ConfigQx) -> u32 {
    c.eos_ngram
}

/// The GDN layers and the attention layers whose states and caches an engine holds, by their places among their kind's (a part of a model split over GPUs: its own; the MTP head's attention layer has the cache after
/// the model's twelve).
fn held(w: &WeightsQx) -> (usize, usize) {
    let held = |k: QxKind| w.layers.iter().flatten().chain(w.mtp.as_ref().map(|m| &m.layer)).filter(|l| l.kind == k).map(|l| l.idx + 1).max().unwrap_or(0);
    (held(QxKind::Gdn), held(QxKind::Attn))
}

/// Bytes of what a sequence holds on the GPU for `ctx` positions, of `ng` GDN layers and `na` attention layers: the GDN states (fp32 [ng, nv, 128, 128]) and convolutions' (bf16 [ng, 3, conv dim]), PLE's convolution
/// (bf16 [9, 4 hidden]), the attention layers' keys and values (bf16 [na, ctx, kv heads, 256] each), the indexer's raw keys (bf16 [na, ctx, 128]) and block keys (bf16 [na, ctx / 4 + 1, 128]).
fn seq_sizes(c: &ConfigQx, ng: usize, na: usize, ctx: usize) -> [usize; 7] {
    let kvb = na * ctx * c.kv_heads * c.head_dim * 2;
    [
        ng * c.nv * 128 * 128 * 4,
        ng * 3 * c.conv_dim() * 2,
        9 * c.hc * c.hidden * 2,
        kvb,
        kvb,
        na * ctx * c.ix_dim * 2,
        na * (ctx / c.ix_block + 1) * c.ix_dim * 2,
    ]
}

/// A sequence beside the engine's own (`EngineQx::swap_seq` puts it in the engine's place): its states and caches (`seq_sizes`, for `ctx` positions), its rotary tables and images (`set_images` turns the tables
/// to a prompt's M-RoPE positions), its length and n-gram context, the decode step's graphs (made with its buffers' addresses), and its request's sampler, sink and halt flag. The draft buffers are the engine's:
/// a verified pass is settled (`commit`) before another sequence runs.
pub struct SeqQx {
    bufs: [Buf; 7],
    rope: [Buf; 2],
    img_rows: Vec<(usize, Vec<u16>)>,
    rope_from: Option<usize>,
    ctx: usize,
    n: usize,
    ng_ctx: [u32; 2],
    segs: Vec<Option<crate::dev::Graph>>,
    sampler: Option<crate::sampler::Sampler>,
    emit: Option<Box<dyn FnMut(&[u32])>>,
    halt: std::rc::Rc<Cell<bool>>,
}

impl SeqQx {
    /// Positions it may reach.
    pub fn ctx(&self) -> usize {
        self.ctx
    }
}

/// What verifying a draft in one pass keeps to rewind to where it was right: the GDN layers' states and convolutions' and PLE's convolution as they were before the pass; per GDN layer the pass's rows of the q, k, v and z
/// product (the convolution's input), of the convolution's output and of b and a (the recurrence's inputs); the pass's rows of PLE's convolution input; the final mixer's rows, the logits and the argmax of the pass; the
/// n-gram context before the pass and the pass's tokens.
pub(crate) struct SpecQx {
    ssm0: Buf,
    conv0: Buf,
    ple0: Buf,
    save: Buf,
    per_layer: usize,
    gvn: Buf,
    hl: Buf,
    lg: Buf,
    tk: Buf,
    vl: Buf,
    ng0: [u32; 2],
    toks: Vec<u32>,
}

impl SpecQx {
    /// Device bytes of the draft buffers.
    #[cfg(feature = "dev")]
    fn bytes(&self) -> usize {
        [&self.ssm0, &self.conv0, &self.ple0, &self.save, &self.gvn, &self.hl, &self.lg, &self.tk, &self.vl].iter().map(|b| b.bytes).sum()
    }

    /// GDN layer `idx`'s saved rows (mix, xc, ba) for SPEC_ROWS rows.
    fn slot(&self, idx: usize, c: &ConfigQx) -> (u64, u64, u64) {
        let base = self.save.ptr + (idx * self.per_layer) as u64;
        let (mix, xc) = (SPEC_ROWS * (c.conv_dim() + c.value_dim()) * 2, SPEC_ROWS * c.conv_dim() * 2);
        (base, base + mix as u64, base + (mix + xc) as u64)
    }
}

/// A prompt taken layer by layer: its rows' streams, positions, n-gram rows and the experts' half's stream weights (`EngineQx::whole_begin`).
struct WholeRows {
    rows: usize,
    x: Buf,
    pos: Buf,
    ple: Buf,
    inj: Buf,
}

/// How a speculative generation went: passes, guesses made, guesses that were right.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub passes: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// When the prompt was done (the first token and the head's first guess), in ms from the call's start: the rest is the drafts' and their verifications'.
    pub prompt_ms: u32,
}

/// A drafted generation between its passes (`EngineQx::mtp_begin`, `mtp_pass`): the tokens made, how it went, the head's drafts for the next pass, and the request's limits.
pub struct MtpRun {
    pub out: Vec<Sample>,
    pub stats: SpecStats,
    drafts: Vec<u32>,
    k: usize,
    max_new: usize,
    stop: Vec<u32>,
}

impl EngineQx {
    pub fn new(dir: &Path, o: &OptsQx, moe: Option<MoeFactoryQx<'_>>) -> Result<EngineQx> {
        let dev = if o.own_ctx { Dev::new_own(o.device)? } else { Dev::new(o.device)? };
        let lib = Lib::find()?;
        let (kern, k35, kg, kq, skinny) =
            (Kern::load(&dev)?, Kern35::load(&dev)?, KernGlm::load(&dev)?, KernQx::load(&dev)?, crate::skinny::SkinnyKern::load(&dev)?);
        let ka = crate::dynamic::attn::KernAttn::load(&dev)?;
        let (free0, _) = dev.mem_info()?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        // (a part of a model split over GPUs: its layers; the final mixer, the LM head and the MTP head with the last one; the n-gram embedding's layer on the first part, where the rows are staged)
        let cfg = ConfigQx::load(dir)?;
        let (l0, l1) = o.span.unwrap_or((0, cfg.layers));
        if l0 >= l1 || l1 > cfg.layers || (l0 > 0 && cfg.ple.as_ref().is_some_and(|p| (l0..l1).contains(&p.layer))) {
            return err(format!("a part of layers {l0}..{l1} of {} (the n-gram embedding's layer goes on the first part)", cfg.layers));
        }
        let span_only: Option<Vec<usize>> = o.span.map(|_| (l0..l1).collect());
        let head = o.span.map_or(o.only.is_none(), |_| l1 == cfg.layers);
        let mut w = WeightsQx::load_mixed(dir, threads, o.layout, o.hi, span_only.as_deref().or(o.only.as_deref()), head, moe, o.mtp && head)?;
        if o.span.is_some() {
            // (a part's states and caches are its own layers': each layer's place among them counted from the part's first)
            let sched = w.cfg.schedule();
            let before = |k: QxKind| sched[..l0].iter().filter(|s| s.0 == k).count();
            let (bg, ba) = (before(QxKind::Gdn), before(QxKind::Attn));
            for ly in w.layers.iter_mut().flatten().chain(w.mtp.as_mut().map(|m| &mut m.layer)) {
                ly.idx -= if ly.kind == QxKind::Gdn { bg } else { ba };
            }
        }
        let c = &w.cfg;
        let (hid, ch) = (c.hidden, o.chunk.max(1));
        let ctx = o.ctx.div_ceil(16) * 16;
        let hw = c.hc * hid;
        // (the streams, the positions, the n-gram rows and the experts' half's stream weights of every row of a prompt taken layer by layer: `whole` rows)
        let whole = o.whole.min(ctx);
        let rows = ch;
        let x = Buf::new(rows * hw * 2)?;
        let cs0 = ChunkSet::new(c, ch, ctx, 16)?;
        let ChunkSet { xn, gw, lo, injl, inj0, h, o: o_, mix, ba, xc, core, qo, ixqk, ixq, scores, sel, cnt, ws: _ } = cs0;
        let rs = Buf::zeroed(8 * c.hc * 4)?;
        let inj = [inj0, Buf::zeroed(rows * c.hc * 4)?];
        // (attn_sel's splits or attn_dec's blocks of 16 dec_wpb() picked tokens, whichever more)
        let part = Buf::zeroed(8 * c.heads * SEL_SPLITS.max(c.ix_tokens().div_ceil(16 * dec_wpb())) * 258 * 4)?;
        let ones = upload_bytes(crate::weights::as_bytes(&vec![0x3f80u16; c.ix_heads.max(8)]))?;
        let ple_e = Buf::new(rows * c.ple.as_ref().map_or(8, |p| p.dim) * 2)?;
        let (cs, sn) = rope_tables(c.theta, c.rotary, ctx);
        let (cos, sin) = (upload_bytes(crate::weights::as_bytes(&cs))?, upload_bytes(crate::weights::as_bytes(&sn))?);
        let (hlast, logits) = (Buf::new(hid * 2)?, Buf::new(c.vocab * 2)?);
        let (tok, val, pos) = (Buf::zeroed(4)?, Buf::zeroed(4)?, Buf::zeroed(rows * 4)?);
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let blas = if ch > 64 && (want == "blas" || (want.is_empty() && dev.cc != (8, 9))) { crate::ffi::Blas::find() } else { None };
        let own = want == "own" || (want.is_empty() && blas.is_none() && matches!(dev.cc.0, 8 | 9));
        let gpu_code = lib.gpu()? as i64;
        let mut shapes: Vec<(i64, i64, bool)> = Vec::new();
        let mut take = |m: &Mat| {
            if let Mat::Packed(p) = m {
                shapes.push((p.pack.rows, p.pack.cols, p.pack.is_twelve()));
            }
        };
        let take_gr = |g: &Gr, take: &mut dyn FnMut(&Mat)| {
            if let GrW::Mats(d, u) = &g.w {
                take(d);
                take(u);
            }
        };
        for ly in w.layers.iter().flatten() {
            ly.mats.iter().for_each(&mut take);
            for g in &ly.gr {
                take_gr(g, &mut take);
            }
            if let Some(p) = &ly.ple {
                take(&p.kv);
            }
        }
        if let Some(m) = &w.mixer {
            take_gr(m, &mut take);
        }
        if let Some(m) = &w.mtp {
            m.layer.mats.iter().for_each(&mut take);
            for g in m.layer.gr.iter().chain([&m.mixer]) {
                take_gr(g, &mut take);
            }
            take(&m.fe);
            take(&m.fh);
        }
        shapes.extend(w.layers.iter().flatten().filter_map(|l| l.moe.as_ref()).flat_map(|b| b.pack_shapes()));
        let gr_dense = c.hc * hid * c.hc_rank.next_multiple_of(128) * 2;
        let scratch_w = Buf::new(shapes.iter().map(|s| (s.0 * s.1) as usize * 2).max().unwrap_or(16).max(gr_dense))?;
        shapes.sort_unstable();
        shapes.dedup();
        let mut need = 0usize;
        for m_ in 1..=ch as i64 {
            for &(rr, kk, twelve) in &shapes {
                need = need.max(lib.linear_workspace(twelve, rr, kk, m_)?);
            }
        }
        // (the MTP head's fc_hidden takes four rows a row: one for each stream)
        if let Some(Mat::Packed(p)) = w.mtp.as_ref().map(|m| &m.fh) {
            for m_ in [(4 * ch) as i64, 4, 8, 32] {
                need = need.max(lib.linear_workspace(p.pack.is_twelve(), p.pack.rows, p.pack.cols, m_)?);
            }
        }
        if let Some(head) = &w.head {
            for m_ in 1..=8i64 {
                need = need.max(lib.linear_workspace(head.pack.is_twelve(), head.pack.rows, head.pack.cols, m_)?);
            }
        }
        let ws = Buf::zeroed(need.max(16))?;
        // (the library's workspace for a prompt chunk's rows: `grow_chunk`'s set)
        // (every row count a prompt chunk may have: the library's workspace is not monotone in the rows)
        let mut ws_prefill = 0usize;
        if whole > 0 && PREFILL_CHUNK > ch {
            for m_ in ch as i64 + 1..=PREFILL_CHUNK as i64 {
                for &(rr, kk, twelve) in &shapes {
                    ws_prefill = ws_prefill.max(lib.linear_workspace(twelve, rr, kk, m_)?);
                }
            }
            ws_prefill = ws_prefill.max(need);
        }
        let done = Buf::zeroed(4 << 18)?;
        let (ng, na) = held(&w);
        let [ssm, convst, plest, kc, vc, ik, pk] = seq_sizes(c, ng, na, ctx).map(Buf::zeroed);
        let (ssm, convst, plest, kc, vc, ik, pk) = (ssm?, convst?, plest?, kc?, vc?, ik?, pk?);
        // (rows for the biggest chunk a prompt takes: its staging steps by its chunk's rows; the last 4,096 bytes for `read_sample`)
        let host = Pinned::new(PREFILL_CHUNK.max(ch) * (hid + c.ple.as_ref().map_or(0, |p| p.dim)) * 2 + 4096)?;
        let spec = if o.spec || o.mtp {
            let (cd, vd) = (c.conv_dim(), c.value_dim());
            let per_layer = SPEC_ROWS * (cd + vd + cd + 2 * c.nv) * 2;
            Some(SpecQx {
                ssm0: Buf::zeroed(ssm.bytes)?,
                conv0: Buf::zeroed(convst.bytes)?,
                ple0: Buf::zeroed(plest.bytes)?,
                save: Buf::zeroed(ng * per_layer)?,
                per_layer,
                gvn: Buf::zeroed(SPEC_ROWS * hw * 2)?,
                hl: Buf::zeroed(SPEC_ROWS * hid * 2)?,
                lg: Buf::zeroed(SPEC_ROWS * c.vocab * 2)?,
                tk: Buf::zeroed(SPEC_ROWS * 4)?,
                vl: Buf::zeroed(SPEC_ROWS * 4)?,
                ng0: [eos_of(c); 2],
                toks: Vec::new(),
            })
        } else {
            None
        };
        let hs = Buf::zeroed(if w.mtp.is_some() { ch * hw * 2 } else { 16 })?;
        let mtp_prev = Buf::zeroed(hw * 2)?;
        // (for the MTP head's steps over a prompt's rows: up to 1,024 a step, `mtp_whole`)
        let ones4 = upload_bytes(crate::weights::as_bytes(&vec![1.0f32; ch.max(1024) * c.hc]))?;
        let hop = if l1 < c.layers { Some(Hop { buf: Pinned::portable(ch.max(HOP_ROWS) * hw * 2)?, ev: crate::dev::Ev::new()?, peer: o.peer }) } else { None };
        let used = free0.saturating_sub(dev.mem_info()?.0);
        let work_bytes = used.saturating_sub(w.bytes);
        // (a development build's GLYD_QX_SIZES=1: the biggest of the engine's own buffers)
        #[cfg(feature = "dev")]
        if std::env::var("GLYD_QX_SIZES").is_ok_and(|v| v == "1") {
            let mb = |b: &Buf| b.bytes as f64 / 1e6;
            eprintln!(
                "[sizes] ws {:.0} MB, scratch_w {:.0}, kv {:.0}+{:.0}, ik {:.0}, ssm {:.0}, scores {:.0}, sel {:.0}, mix {:.0}, x/xn/gw {:.0}/{:.0}/{:.0}, xc {:.0}, core {:.0}, qo {:.0}, spec {:.0}, hs {:.0}",
                mb(&ws), mb(&scratch_w), mb(&kc), mb(&vc), mb(&ik), mb(&ssm), mb(&scores), mb(&sel), mb(&mix), mb(&x), mb(&xn), mb(&gw), mb(&xc), mb(&core), mb(&qo),
                spec.as_ref().map_or(0.0, |sp| sp.bytes() as f64 / 1e6), mb(&hs)
            );
        }
        let eos = c.eos_ngram;
        Ok(EngineQx {
            dev,
            lib,
            kern,
            k35,
            ka,
            kg,
            kq,
            skinny,
            skinny_on: std::rc::Rc::new(Cell::new(false)),
            w,
            span: (l0, l1),
            front: None,
            hop,
            chunk: ch,
            pchunk: Cell::new(ch),
            ws_prefill,
            whole,
            wb: std::cell::RefCell::new(None),
            whole_on: Cell::new(true),
            ctx,
            ctx_max: ctx,
            chunked: o.chunked && !crate::dev::smem64(),
            x,
            xn,
            gw,
            lo,
            injl,
            rs,
            inj,
            h,
            o: o_,
            mix,
            ba,
            xc,
            core,
            qo,
            ixqk,
            ixq,
            scores,
            sel,
            cnt,
            part,
            ones,
            ple_e,
            cos,
            sin,
            hlast,
            logits,
            tok,
            val,
            pos,
            ws,
            done,
            blas,
            own,
            scratch_w,
            gpu_code,
            ssm,
            convst,
            kc,
            vc,
            ik,
            pk,
            plest,
            host,
            n: 0,
            ng_ctx: [eos, eos],
            moe_scratch: OnceCell::new(),
            trace: None,
            work_bytes,
            segs: std::cell::RefCell::new(Vec::new()),
            segmented: Cell::new(false),
            spec,
            save: Cell::new(false),
            pending: Cell::new(None),
            hs,
            mtp_prev,
            draft_vocab: Cell::new(0),
            dlg: std::cell::RefCell::new(None),
            draft_min_p: Cell::new(0.0),
            kprof_passes: Cell::new(false),
            emit: std::cell::RefCell::new(None),
            halt: std::rc::Rc::new(Cell::new(false)),
            keep: Cell::new(0),
            sampler: std::cell::RefCell::new(None),
            samp: (
                Buf::zeroed(4 * crate::kern::TOPK * SPEC_ROWS)?,
                Buf::zeroed(4 * crate::kern::TOPK * SPEC_ROWS)?,
                Buf::zeroed(8 * SPEC_ROWS)?,
                Pinned::new(8 * crate::kern::TOPK * SPEC_ROWS + 8 * SPEC_ROWS)?,
            ),
            maskgpu: crate::kern::MaskGpu::new(cfg.vocab, 1).ok(),
            draft_p: Cell::new(1.0),
            dpb: std::cell::RefCell::new(None),
            ones4,
            img_rows: std::cell::RefCell::new(Vec::new()),
            rope_from: Cell::new(None),
        })
    }

    pub fn describe(&self) -> String {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        let me = format!(
            "{} (sm_{}{}): weights {:.3} GiB (layout {:?}), states and workspaces {:.3} GiB, {:.2} of {:.2} GiB in use",
            self.dev.name,
            self.dev.cc.0,
            self.dev.cc.1,
            gib(self.w.bytes),
            self.w.layout,
            gib(self.work_bytes),
            gib(total - free),
            gib(total)
        );
        if self.front.is_none() && self.hop.is_none() {
            return me;
        }
        let part = format!("layers {}-{} on GPU {}: {me}", self.span.0, self.span.1 - 1, self.dev.ordinal);
        match self.on_front(|f| Ok(f.describe())) {
            Ok(Some(f)) => format!("{f}; {part}"),
            _ => part,
        }
    }

    /// This engine, the last part of a model split over GPUs, with the part before it (`front`, whose layers end where this one's begin: on another GPU, or in a context of its own on this one): the token rows are staged
    /// there, its streams come here after its layers (`Hop`), and the settings both read are shared.
    pub fn with_front(mut self, mut front: EngineQx) -> Result<EngineQx> {
        if front.span.1 != self.span.0 || front.hop.is_none() || self.front.is_some() {
            return err(format!("layers {}..{} after a part of layers {}..{}", self.span.0, self.span.1, front.span.0, front.span.1));
        }
        let mut at = Some(&mut front);
        while let Some(f) = at {
            f.skinny_on = self.skinny_on.clone();
            at = f.front.as_deref_mut();
        }
        self.front = Some(Box::new(front));
        self.dev.make_current()?;
        Ok(self)
    }

    /// `f` on the part of the model before this one (where it is split over GPUs) with that part's context current, then this one's again.
    fn on_front<T>(&self, f: impl FnOnce(&EngineQx) -> Result<T>) -> Result<Option<T>> {
        let Some(fr) = self.front.as_deref() else { return Ok(None) };
        fr.dev.make_current()?;
        let r = f(fr);
        self.dev.make_current()?;
        r.map(Some)
    }

    fn on_front_mut<T>(&mut self, f: impl FnOnce(&mut EngineQx) -> Result<T>) -> Result<Option<T>> {
        let Some(fr) = self.front.as_deref_mut() else { return Ok(None) };
        fr.dev.make_current()?;
        let r = f(fr);
        self.dev.make_current()?;
        r.map(Some)
    }

    /// `f` on this part of the model and every part before it (settings, no GPU work).
    fn each_part(&mut self, f: &mut dyn FnMut(&mut EngineQx)) {
        f(self);
        if let Some(fr) = self.front.as_deref_mut() {
            fr.each_part(f);
        }
    }

    /// Rows `n` of the streams at `src` on the part before this one, after the work it has queued, into `dst` on this one's stream: GPU to GPU where its hop says so, else through the hop's pinned buffer, in pieces of what it
    /// holds (the host waits for a piece before the next; one piece is all queued, no wait).
    ///
    /// # Safety
    /// `src` and `dst` hold n rows; the hop's buffer is not being read by an earlier copy (the host has waited for this stream since).
    unsafe fn take_rows(&self, src: u64, dst: u64, n: usize) -> Result<()> {
        let hop = self.front.as_deref().and_then(|f| f.hop.as_ref()).ok_or_else(|| Error("no part of the model before this one".into()))?;
        let (s, row) = (self.dev.stream, self.w.cfg.hc * self.w.cfg.hidden * 2);
        if hop.peer {
            self.on_front(|f| hop.ev.record(f.dev.stream))?;
            hop.ev.wait(s)?;
            return crate::dev::copy_d2d_async(dst, src, n * row, s);
        }
        let per = (hop.buf.bytes / row).max(1);
        for i in (0..n).step_by(per) {
            let m = per.min(n - i);
            if i > 0 {
                self.dev.sync()?;
            }
            self.on_front(|f| {
                copy_d2h_async(hop.buf.ptr, src + (i * row) as u64, m * row, f.dev.stream)?;
                hop.ev.record(f.dev.stream)
            })?;
            hop.ev.wait(s)?;
            copy_h2d_async(dst + (i * row) as u64, hop.buf.ptr, m * row, s)?;
        }
        Ok(())
    }

    /// The positions of the next pass's rows written for every part of the model (the streams idle).
    fn set_pos(&self, pos: &[i32]) -> Result<()> {
        self.on_front(|f| f.set_pos(pos))?;
        self.pos.write(0, pos)
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// The rows the experts' front (the router, x', the shared expert) takes at once in a prompt taken layer by layer: its scratch's (`QxMoeCb::scratch`), the engine's chunk or `FRONT_ROWS` if more.
    pub fn front_rows(&self) -> usize {
        self.chunk.max(FRONT_ROWS)
    }

    /// The places of the last layers (the MTP head's first, then layer 47 down) freed until the GPU has `bytes` free and 64 MB more: the memory a long prompt's buffers take while it is read (the decode's cache of
    /// those layers starts empty after `return_places`). The layers freed, to be given to `return_places`.
    pub fn borrow_places(&self, bytes: usize) -> Result<Vec<usize>> {
        let want = bytes + (64 << 20);
        let mut freed = Vec::new();
        // (the layers' blocks from the last: the MTP head's as layer `layers`)
        let n = self.w.layers.len();
        for l in (0..=n).rev() {
            if self.dev.mem_info()?.0 >= want {
                break;
            }
            let block =
                if l == n { self.w.mtp.as_ref().and_then(|m| m.layer.moe.as_deref()) } else { self.w.layers[l].as_ref().and_then(|ly| ly.moe.as_deref()) };
            if let Some(b) = block {
                if b.release_places()? > 0 {
                    freed.push(l);
                }
            }
        }
        if self.dev.mem_info()?.0 < want {
            self.return_places(&freed)?;
            return err(format!("{bytes} bytes for a long prompt's buffers: not free with every layer's places given back"));
        }
        Ok(freed)
    }

    /// The places `borrow_places` freed made again, empty.
    pub fn return_places(&self, layers: &[usize]) -> Result<()> {
        let n = self.w.layers.len();
        for &l in layers {
            let block =
                if l == n { self.w.mtp.as_ref().and_then(|m| m.layer.moe.as_deref()) } else { self.w.layers[l].as_ref().and_then(|ly| ly.moe.as_deref()) };
            if let Some(b) = block {
                b.restore_places()?;
            }
        }
        Ok(())
    }

    /// Whether the MTP head is loaded (`generate_mtp`).
    pub fn has_mtp(&self) -> bool {
        self.w.mtp.is_some() && self.spec.is_some()
    }

    /// Rows of a prompt taken layer by layer at most (`OptsQx::whole`; 0: none).
    pub fn whole_rows(&self) -> usize {
        self.whole
    }

    /// Rows of the prompt being taken layer by layer (0: none).
    pub fn whole_now(&self) -> usize {
        self.wb.borrow().as_ref().map_or(0, |w| w.rows)
    }

    /// The buffers of `x`, the positions, the n-gram rows and the stream weights of the prompt being taken layer by layer.
    fn whole_ptrs(&self) -> Result<(u64, u64, u64, u64)> {
        let wb = self.wb.borrow();
        let w = wb.as_ref().ok_or_else(|| Error("no prompt is being taken layer by layer".into()))?;
        Ok((w.x.ptr, w.pos.ptr, w.ple.ptr, w.inj.ptr))
    }

    /// The positions of the prompt's rows written (the stream idle).
    fn whole_pos(&self, pos: &[i32]) -> Result<()> {
        self.wb.borrow().as_ref().ok_or_else(|| Error("no prompt is being taken layer by layer".into()))?.pos.write(0, pos)
    }

    /// The buffers of a prompt of `rows` rows taken layer by layer made, the last layers' places lent to them where the GPU has not the room (`borrow_places`): the layers lent, for `whole_end`.
    fn whole_begin(&self, rows: usize) -> Result<Vec<usize>> {
        let c = &self.w.cfg;
        let (hw, ple) = (c.hc * c.hidden, c.ple.as_ref().map_or(8, |p| p.dim));
        // (the engine's rows, the experts' set, the streaming prefill's ring and plan and its host share; 128 MB beside them)
        // (and the set of the prompt's chunk, `grow_chunk`'s, where it takes one)
        let p = self.prompt_chunk(rows);
        let chunk_set = if p > self.chunk && self.ws_prefill > 0 { ChunkSet::bytes(c, p, self.ctx, self.ws_prefill) } else { 0 };
        // (and the delta rule's scratch, made by its first call for the engine's chunk, freed in `whole_end`)
        let gdn = p.max(self.chunk).min(GDN_SLICE).div_ceil(64) * (c.nv * (3 * 64 * 64 * 2 + 64 * 4) + 2 * c.nk * 64 * 128 * 2);
        let need = rows * (hw * 2 + 4 + ple * 2 + c.hc * 4) + crate::moe_qwen4x::whole_call_bytes(rows, c.top_k) + chunk_set + gdn + (128 << 20);
        let lent = if self.dev.mem_info()?.0 < need { self.borrow_places(need)? } else { Vec::new() };
        let made = (|| -> Result<WholeRows> {
            Ok(WholeRows { rows, x: Buf::new(rows * hw * 2)?, pos: Buf::zeroed(rows * 4)?, ple: Buf::new(rows * ple * 2)?, inj: Buf::zeroed(rows * c.hc * 4)? })
        })();
        match made {
            Ok(w) => {
                *self.wb.borrow_mut() = Some(w);
                Ok(lent)
            }
            Err(e) => {
                self.return_places(&lent)?;
                Err(e)
            }
        }
    }

    /// The prompt's buffers freed (the engine's, the experts' set, the streaming prefill's) and the places lent made again, empty.
    fn whole_end(&self, lent: &[usize]) -> Result<()> {
        self.dev.sync()?;
        *self.wb.borrow_mut() = None;
        *self.kq.gdn_buf.borrow_mut() = None;
        if let Some(b) = self.w.layers.iter().flatten().find_map(|l| l.moe.as_deref()) {
            b.whole_release(self)?;
        }
        self.return_places(lent)?;
        // (the prompt's last rows' experts placed for the decode, the first layers' first)
        for b in self.w.layers.iter().flatten().filter_map(|l| l.moe.as_deref()).chain(self.w.mtp.as_ref().and_then(|m| m.layer.moe.as_deref())) {
            b.rewarm()?;
        }
        if let Some(b) = self.w.layers.iter().flatten().find_map(|l| l.moe.as_deref()) {
            b.drain()?;
        }
        Ok(())
    }

    /// The chunk-sized buffers of a prompt taken layer by layer swapped for a set of `PREFILL_CHUNK` rows (made now, inside the room `whole_begin` made: `ChunkSet::bytes`): the engine's own set, kept for `shrink_chunk`.
    /// None where the engine's chunk is already as big, or the prompt is shorter than it.
    /// A prompt chunk's rows for a prompt of `rows`: up to `PREFILL_CHUNK`, fewer at a long context, where the indexer's scores of a chunk
    /// (a row's against every block of the context) would pass 512 MB (at 262,144 positions: 2,048 rows).
    fn prompt_chunk(&self, rows: usize) -> usize {
        let c = &self.w.cfg;
        let per_row = (self.ctx / c.ix_block.max(1)).max(1) * 4;
        let cap = ((512usize << 20) / per_row).max(self.chunk) / self.chunk * self.chunk;
        PREFILL_CHUNK.min(rows.next_multiple_of(self.chunk)).min(cap)
    }

    fn grow_chunk(&mut self, rows: usize) -> Result<Option<ChunkSet>> {
        let p = self.prompt_chunk(rows);
        if p <= self.chunk || self.ws_prefill == 0 {
            return Ok(None);
        }
        // (made only where it leaves the room the prompt's other buffers need: else the prompt keeps the engine's chunk)
        let bytes = ChunkSet::bytes(&self.w.cfg, p, self.ctx, self.ws_prefill);
        if self.dev.mem_info()?.0 < bytes + (192 << 20) {
            return Ok(None);
        }
        let mut set = ChunkSet::new(&self.w.cfg, p, self.ctx, self.ws_prefill)?;
        self.dev.sync()?;
        self.swap_chunk(&mut set);
        self.pchunk.set(p);
        Ok(Some(set))
    }

    /// The engine's own chunk-sized buffers back (`grow_chunk`'s), the prompt's set freed: the decode step's graphs see the addresses they were made with.
    fn shrink_chunk(&mut self, saved: Option<ChunkSet>) -> Result<()> {
        if let Some(mut set) = saved {
            self.dev.sync()?;
            self.swap_chunk(&mut set);
            self.pchunk.set(self.chunk);
        }
        Ok(())
    }

    fn swap_chunk(&mut self, set: &mut ChunkSet) {
        use std::mem::swap;
        swap(&mut self.xn, &mut set.xn);
        swap(&mut self.gw, &mut set.gw);
        swap(&mut self.lo, &mut set.lo);
        swap(&mut self.injl, &mut set.injl);
        swap(&mut self.inj[0], &mut set.inj0);
        swap(&mut self.h, &mut set.h);
        swap(&mut self.o, &mut set.o);
        swap(&mut self.mix, &mut set.mix);
        swap(&mut self.ba, &mut set.ba);
        swap(&mut self.xc, &mut set.xc);
        swap(&mut self.core, &mut set.core);
        swap(&mut self.qo, &mut set.qo);
        swap(&mut self.ixqk, &mut set.ixqk);
        swap(&mut self.ixq, &mut set.ixq);
        swap(&mut self.scores, &mut set.scores);
        swap(&mut self.sel, &mut set.sel);
        swap(&mut self.cnt, &mut set.cnt);
        swap(&mut self.ws, &mut set.ws);
    }

    /// Whether a prompt of `n` tokens is taken layer by layer: asked for, long enough for the experts to be streamed, no development trace hook, and every layer's block (the MTP head's too) able to.
    fn takes_whole(&self, n: usize) -> bool {
        self.whole > 0 && self.whole_on.get() && n >= crate::moe_qwen4x::STREAM_MIN && self.trace.is_none() && self.whole_able()
    }

    /// Whether every layer's block of this part of the model and of the parts before it (the MTP head's too) takes a prompt layer by layer.
    fn whole_able(&self) -> bool {
        let able = |l: &LayerQx| l.moe.as_ref().is_some_and(|b| b.whole());
        (self.span.0..self.span.1).all(|l| self.w.layers[l].as_ref().is_some_and(able))
            && self.w.mtp.as_ref().is_none_or(|m| able(&m.layer))
            && self.front.as_ref().is_none_or(|f| f.whole > 0 && f.whole_able())
    }

    /// The buffers of a prompt of `rows` rows taken layer by layer, on every part of the model (`whole_begin`, `grow_chunk`); `prompt_end` frees them.
    fn prompt_begin(&mut self, rows: usize) -> Result<PromptBufs> {
        let front = self.on_front_mut(|f| f.prompt_begin(rows))?.map(Box::new);
        let made = self.whole_begin(rows).and_then(|lent| match self.grow_chunk(rows) {
            Ok(grown) => Ok((lent, grown)),
            Err(e) => self.whole_end(&lent).and(Err(e)),
        });
        match made {
            Ok((lent, grown)) => Ok(PromptBufs { lent, grown, front }),
            Err(e) => {
                if let Some(f) = front {
                    self.on_front_mut(|fr| fr.prompt_end(*f))?;
                }
                Err(e)
            }
        }
    }

    fn prompt_end(&mut self, p: PromptBufs) -> Result<()> {
        self.shrink_chunk(p.grown)?;
        self.whole_end(&p.lent)?;
        if let Some(f) = p.front {
            self.on_front_mut(|fr| fr.prompt_end(*f))?;
        }
        Ok(())
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// Positions a sequence may reach at most (`new_seq`'s, the engine's own as made).
    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    /// Device bytes of a sequence of `ctx` positions (`new_seq`).
    pub fn seq_bytes(&self, ctx: usize) -> usize {
        let ((ng, na), ctx) = (held(&self.w), ctx.div_ceil(16) * 16);
        seq_sizes(&self.w.cfg, ng, na, ctx).iter().sum::<usize>() + 2 * ctx * (self.w.cfg.rotary / 2) * 2
    }

    /// Whether sequences beside the engine's own can run on it (`swap_seq`): not on a model split over GPUs (each part would need its share).
    pub fn takes_seqs(&self) -> bool {
        self.front.is_none() && self.hop.is_none()
    }

    /// A sequence of its own for up to `ctx` positions (at most `ctx_max`), empty: its states zeroed, run in the engine's place by `swap_seq`.
    pub fn new_seq(&self, ctx: usize) -> Result<SeqQx> {
        let ctx = ctx.div_ceil(16) * 16;
        if ctx == 0 || ctx > self.ctx_max || !self.takes_seqs() {
            return err(format!("a sequence of {ctx} positions, where the engine was made for {} (and is not split over GPUs)", self.ctx_max));
        }
        let (ng, na) = held(&self.w);
        let [a, b, c, d, e, f, g] = seq_sizes(&self.w.cfg, ng, na, ctx).map(Buf::zeroed);
        let (cs, sn) = rope_tables(self.w.cfg.theta, self.w.cfg.rotary, ctx);
        let rope = [upload_bytes(crate::weights::as_bytes(&cs))?, upload_bytes(crate::weights::as_bytes(&sn))?];
        let eos = eos_of(&self.w.cfg);
        Ok(SeqQx {
            bufs: [a?, b?, c?, d?, e?, f?, g?],
            rope,
            img_rows: Vec::new(),
            rope_from: None,
            ctx,
            n: 0,
            ng_ctx: [eos; 2],
            segs: Vec::new(),
            sampler: None,
            emit: None,
            halt: std::rc::Rc::new(Cell::new(false)),
        })
    }

    /// The sequence `s` and the engine's own swapped: the engine's steps run on `s`'s states, caches and graphs (and its request's sampler, sink and halt flag), `s` keeps the one that was there. Between passes only
    /// (a verified pass settled); the work queued is ordered by the stream, so no wait.
    pub fn swap_seq(&mut self, s: &mut SeqQx) -> Result<()> {
        use std::mem::swap;
        if self.pending.get().is_some() || !self.takes_seqs() {
            return err("a sequence swapped out with a verified pass not settled, or on a model split over GPUs");
        }
        let [a, b, c, d, e, f, g] = &mut s.bufs;
        for (mine, theirs) in
            [&mut self.ssm, &mut self.convst, &mut self.plest, &mut self.kc, &mut self.vc, &mut self.ik, &mut self.pk].into_iter().zip([a, b, c, d, e, f, g])
        {
            swap(mine, theirs);
        }
        let [c, d] = &mut s.rope;
        swap(&mut self.cos, c);
        swap(&mut self.sin, d);
        swap(self.img_rows.get_mut(), &mut s.img_rows);
        swap(self.rope_from.get_mut(), &mut s.rope_from);
        swap(&mut self.ctx, &mut s.ctx);
        swap(&mut self.n, &mut s.n);
        swap(&mut self.ng_ctx, &mut s.ng_ctx);
        swap(self.segs.get_mut(), &mut s.segs);
        swap(self.sampler.get_mut(), &mut s.sampler);
        swap(self.emit.get_mut(), &mut s.emit);
        swap(&mut self.halt, &mut s.halt);
        Ok(())
    }

    /// Every layer's block of experts, the MTP head's last.
    fn blocks(&self) -> impl Iterator<Item = &dyn MoeBlockQx> {
        self.w.layers.iter().flatten().filter_map(|l| l.moe.as_deref()).chain(self.w.mtp.as_ref().and_then(|m| m.layer.moe.as_deref()))
    }

    /// The places of the experts: (a layer's now, a layer's as loaded, the bytes of one place in every layer).
    pub fn place_slots(&self) -> (usize, usize, usize) {
        self.blocks().map(|b| b.place_slots()).fold((usize::MAX, 0, 0), |(s, f, b), (s1, f1, b1)| (s.min(s1), f.max(f1), b + b1))
    }

    /// Every layer's places made again for `slots` experts (each at most as many as it was loaded with), the experts in the first of them kept: the memory of sequences beside the engine's own, and back after them.
    pub fn resize_places(&self, slots: usize) -> Result<()> {
        self.dev.sync()?;
        for b in self.blocks() {
            b.resize_places(self, slots)?;
        }
        Ok(())
    }

    /// A new sequence: the states zeroed, the n-gram's context the end token.
    pub fn reset(&mut self) -> Result<()> {
        self.each_part(&mut |e| {
            e.n = 0;
            e.ng_ctx = [e.w.cfg.eos_ngram; 2];
        });
        self.zero_states()
    }

    /// The GDN layers' and PLE's states zeroed (the caches need none: a position is written before it is read), on every part of the model.
    pub fn zero_states(&self) -> Result<()> {
        self.on_front(|f| f.zero_states())?;
        for b in [&self.ssm, &self.convst, &self.plest] {
            b.fill_async(0, self.dev.stream)?;
        }
        self.dev.sync()
    }

    /// Last forward pass's logits (bf16, widened).
    pub fn logits(&self) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut words = vec![0u16; self.w.cfg.vocab];
        self.logits.read(0, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

    /// The last `verify`'s first `n` rows of logits, one after another.
    pub fn verify_logits(&self, n: usize) -> Result<Vec<f32>> {
        let sp = self.spec.as_ref().ok_or_else(|| Error("no draft buffers".into()))?;
        self.dev.sync()?;
        let mut words = vec![0u16; n.min(SPEC_ROWS) * self.w.cfg.vocab];
        sp.lg.read(0, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

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

    /// Y [m, rows] = X [m, cols] W^T.
    ///
    /// # Safety
    /// Live device buffers of those sizes, on the engine's stream.
    pub unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        match w {
            Mat::Packed(p) => self.lin_packed(p, x, m, y),
            Mat::Dense(b, rows, cols) => self.kg.small_linear(self.dev.stream, x, *cols as i32, b.ptr, y, *rows as i32, m as u32, *rows as i32, *cols as i32),
        }
    }

    /// A Gated Residual on the n streams at `x`: the norm into `xn`, the mixing weights, the block's input into `h`; with an inject matrix, the streams' weights into `inj` (fp32 [n, 4]).
    ///
    /// # Safety
    /// Live buffers of the engine, sized for a chunk.
    pub(crate) unsafe fn gr_mix(&self, g: &Gr, x: u64, n: usize, h: u64, inj: u64) -> Result<()> {
        self.gr_mix_post(g, x, n, h, inj, None)
    }

    /// `gr_mix` after the streams' update the mixer's half left pending (`post`: its output and stream weights, x += out inj as `hc_post` makes it): made inside the fused kernels for a few rows, else first.
    ///
    /// # Safety
    /// As `gr_mix`.
    pub(crate) unsafe fn gr_mix_post(&self, g: &Gr, x: u64, n: usize, h: u64, inj: u64, post: Option<(u64, u64)>) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let hw = c.hc * c.hidden;
        if let GrW::Q { r, dq, ds, uq, us, .. } = &g.w {
            if n <= 8 {
                // (a few rows: the two fused launches)
                let injw = g.inject.as_ref().map_or(0, |b| b.ptr);
                let p = post.unwrap_or((0, 0));
                self.kq.gr1(s, x, g.norm.ptr, dq.ptr, ds.ptr, injw, self.rs.ptr, self.lo.ptr, inj, n, c.hidden, *r, c.eps, p)?;
                crate::tmark!(s, "gr: fused norm, down, inject");
                self.kq.gr2(s, x, g.norm.ptr, self.rs.ptr, self.lo.ptr, uq.ptr, us.ptr, h, n, c.hidden, *r, p)?;
                crate::tmark!(s, "gr: fused up, mix");
                return Ok(());
            }
        }
        if let Some((o, pinj)) = post {
            self.kq.hc_post(s, x, o, pinj, n, c.hidden)?;
        }
        self.kq.hc_norm(s, x, g.norm.ptr, self.xn.ptr, n, c.hidden, c.eps)?;
        crate::tmark!(s, "gr: norm");
        match &g.w {
            GrW::Mats(down, up) => {
                self.lin(down, self.xn.ptr, n, self.lo.ptr)?;
                crate::tmark!(s, "gr: down product");
                self.kq.silu4(s, self.lo.ptr, n * c.hc_rank)?;
                self.lin(up, self.lo.ptr, n, self.gw.ptr)?;
            }
            GrW::Q { r, rp, dq, ds, uq, us } => {
                // (a prompt chunk: the int8 rows made bf16, the engine's dense products; the bottleneck padded with zeros to `rp`)
                let w = self.scratch_w.ptr;
                self.kq.i8_bf16(s, dq.ptr, ds.ptr, w, *r, hw, *rp, hw)?;
                self.kern.gemm_fill(s, self.xn.ptr, w, self.lo.ptr, n as i32, *rp as i32, hw as i32)?;
                crate::tmark!(s, "gr: down product");
                self.kq.silu4(s, self.lo.ptr, n * rp)?;
                self.kq.i8_bf16(s, uq.ptr, us.ptr, w, hw, *r, hw, *rp)?;
                self.kern.gemm_fill(s, self.lo.ptr, w, self.gw.ptr, n as i32, hw as i32, *rp as i32)?;
            }
        }
        crate::tmark!(s, "gr: up product");
        let injl = match &g.inject {
            Some(b) => {
                self.kg.small_linear(s, self.xn.ptr, hw as i32, b.ptr, self.injl.ptr, c.hc as i32, n as u32, c.hc as i32, hw as i32)?;
                self.injl.ptr
            }
            None => 0,
        };
        self.kq.hc_mix(s, self.gw.ptr, self.xn.ptr, injl, h, inj, n, c.hidden)
    }

    /// PLE on the n rows (their n-gram rows in `ple_e`): the key and value products, the gate, the convolution, the sum into the streams.
    ///
    /// # Safety
    /// As `gr_mix`.
    pub(crate) unsafe fn ple_half(&self, p: &PleW, n: usize) -> Result<()> {
        self.ple_half_x(p, n, self.x.ptr, self.ple_e.ptr)
    }

    /// `ple_half` on the n streams at `x` with their n-gram rows at `ple`.
    ///
    /// # Safety
    /// As `gr_mix`.
    unsafe fn ple_half_x(&self, p: &PleW, n: usize, x: u64, ple: u64) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        self.lin(&p.kv, ple, n, self.mix.ptr)?;
        self.kq.ple_gate(s, self.mix.ptr, x, p.nk.ptr, p.nq.ptr, p.nc.ptr, self.gw.ptr, self.xn.ptr, n, c.hidden, c.eps)?;
        if let (Some(sp), true) = (&self.spec, self.save.get()) {
            crate::dev::copy_d2d_async(sp.gvn.ptr, self.xn.ptr, n * c.hc * c.hidden * 2, s)?;
        }
        self.kq.ple_conv(s, self.xn.ptr, self.gw.ptr, p.conv.ptr, self.plest.ptr, x, n, c.hc * c.hidden)
    }

    /// A GDN layer's mixer on the n rows in `h` into `o`.
    unsafe fn gdn(&self, ly: &LayerQx, n: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (cd, vd) = (c.conv_dim(), c.value_dim());
        let row = cd + vd;
        let sm = &ly.small;
        let conv = self.convst.ptr + (ly.idx * 3 * cd * 2) as u64;
        let state = self.ssm.ptr + (ly.idx * c.nv * 128 * 128 * 4) as u64;
        // (a verified draft's pass keeps this layer's inputs of the recurrence, for a rewind: `SpecQx`)
        let (mix, xc, ba) = match (&self.spec, self.save.get()) {
            (Some(sp), true) => sp.slot(ly.idx, c),
            _ => (self.mix.ptr, self.xc.ptr, self.ba.ptr),
        };
        self.lin(&ly.mats[0], self.h.ptr, n, mix)?;
        crate::tmark!(s, "gdn: qkvz product");
        // (a decode step's or a verification's few rows: the convolution beside the b and a product, then the rule with the gated norm, two launches for four)
        if let (true, Mat::Dense(wb, nb, kb)) = (n <= SPEC_ROWS && c.gate_sigmoid, &ly.mats[1]) {
            self.kq.gdn_conv_ba(s, mix, row, sm[0].ptr, conv, xc, n, cd, self.h.ptr, wb.ptr, ba, *nb, *nb, *kb)?;
            crate::tmark!(s, "gdn: convolution, b and a");
            // (a verification of two rows: the state after the first kept beside, for `commit` to go back to)
            let st1 = match (&self.spec, self.save.get() && n == 2 && self.gdn_st1()) {
                (Some(sp), true) => sp.ssm0.ptr + (ly.idx * c.nv * 128 * 128 * 4) as u64,
                _ => 0,
            };
            self.kq.gdn_rec_gn(
                s,
                xc,
                cd,
                ba,
                2 * c.nv,
                sm[1].ptr,
                sm[2].ptr,
                state,
                self.core.ptr,
                n,
                c.nk,
                c.nv,
                mix + (cd * 2) as u64,
                row,
                sm[3].ptr,
                c.eps,
                st1,
            )?;
            crate::tmark!(s, "gdn: delta rule, gated norm");
            self.lin(&ly.mats[2], self.core.ptr, n, self.o.ptr)?;
            crate::tmark!(s, "gdn: out product");
            return Ok(());
        }
        self.lin(&ly.mats[1], self.h.ptr, n, ba)?;
        crate::tmark!(s, "gdn: ba product");
        if n > 16 {
            self.kq.conv_tok(s, mix, row, sm[0].ptr, conv, xc, n, cd)?;
        } else {
            self.k35.conv(s, mix, row as i32, sm[0].ptr, conv, xc, n as i32, cd as i32)?;
        }
        crate::tmark!(s, "gdn: convolution");
        if self.chunked && n > 16 && !crate::tune!("GLYD_GDN_ONEPASS").is_some_and(|v| v == "1") {
            #[cfg(feature = "dev")]
            let check = if crate::tune!("GLYD_GDN_CHECK").is_some() && ly.idx % 12 == 5 {
                // (a development check: the one-pass kernel on a copy of the state beside the two passes)
                let (st2, out2) = (Buf::new(c.nv * 128 * 128 * 4)?, Buf::new(n * vd * 2)?);
                crate::dev::copy_d2d_async(st2.ptr, state, st2.bytes, s)?;
                self.k35.delta(s, true, xc, cd as i32, ba, 2 * c.nv as i32, sm[1].ptr, sm[2].ptr, st2.ptr, out2.ptr, n as i32, c.nk as i32, c.nv as i32)?;
                Some((st2, out2))
            } else {
                None
            };
            // (GDN_SLICE rows at most at a time: the scratch of its chunks' part is made for the biggest slice, freed with the prompt's buffers; the state carries from slice to slice)
            let sl = self.pchunk.get().min(GDN_SLICE);
            for i in (0..n).step_by(sl) {
                let m = sl.min(n - i);
                self.kq.delta2(
                    s,
                    xc + (i * cd * 2) as u64,
                    cd,
                    ba + (i * 2 * c.nv * 2) as u64,
                    2 * c.nv,
                    sm[1].ptr,
                    sm[2].ptr,
                    state,
                    self.core.ptr + (i * vd * 2) as u64,
                    m,
                    c.nk,
                    c.nv,
                )?;
            }
            #[cfg(feature = "dev")]
            if let Some((st2, out2)) = check {
                let st1 = Buf::new(st2.bytes)?;
                crate::dev::copy_d2d_async(st1.ptr, state, st1.bytes, s)?;
                self.dev.sync()?;
                let rel = |a: &[f32], b: &[f32]| {
                    (a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().max(1e-30)).sqrt()
                };
                let (mut sa, mut sb) = (vec![0f32; c.nv * 128 * 128], vec![0f32; c.nv * 128 * 128]);
                st1.read(0, &mut sa)?;
                st2.read(0, &mut sb)?;
                let (mut oa, mut ob) = (vec![0u16; n * vd], vec![0u16; n * vd]);
                self.core.read(0, &mut oa)?;
                out2.read(0, &mut ob)?;
                let f = |v: &[u16]| v.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect::<Vec<f32>>();
                eprintln!("gdn check: layer {} ({n} rows): outputs relative {:.3e}, state relative {:.3e}", ly.idx, rel(&f(&oa), &f(&ob)), rel(&sa, &sb));
            }
        } else {
            self.k35.delta(
                s,
                self.chunked && n > 1,
                xc,
                cd as i32,
                ba,
                2 * c.nv as i32,
                sm[1].ptr,
                sm[2].ptr,
                state,
                self.core.ptr,
                n as i32,
                c.nk as i32,
                c.nv as i32,
            )?;
        }
        crate::tmark!(s, "gdn: delta rule");
        if !c.gate_sigmoid {
            return err("a GDN with a silu output gate: Qwen3.8-Flash-Next's is a sigmoid");
        }
        self.kq.gnorm(s, self.core.ptr, mix + (cd * 2) as u64, row, sm[3].ptr, n, c.nv, c.eps)?;
        crate::tmark!(s, "gdn: gated norm");
        self.lin(&ly.mats[2], self.core.ptr, n, self.o.ptr)?;
        crate::tmark!(s, "gdn: out product");
        Ok(())
    }

    /// An attention layer's mixer on the n rows in `h` (positions at `pos`, `start` tokens before them) into `o`: the q, k and v product, the norms and rotary, the cache; the indexer's product, its queries and
    /// keys, the block keys; the attention over every token (a prompt chunk within the budget) or over the indexer's picks; the o product.
    unsafe fn attn(&self, ly: &LayerQx, n: usize, pos: u64, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (nq, nk, qd) = (c.heads, c.kv_heads, c.qkv_dim());
        let half = c.rotary / 2;
        let sm = &ly.small;
        let kvl = (self.ctx * nk * c.head_dim * 2) as u64;
        let (kc, vc) = (self.kc.ptr + ly.idx as u64 * kvl, self.vc.ptr + ly.idx as u64 * kvl);
        let ik = self.ik.ptr + (ly.idx * self.ctx * c.ix_dim * 2) as u64;
        let pk = self.pk.ptr + (ly.idx * (self.ctx / c.ix_block + 1) * c.ix_dim * 2) as u64;
        self.lin(&ly.mats[0], self.h.ptr, n, self.mix.ptr)?;
        crate::tmark!(s, "attn: qkv product");
        // (the Qwen3.5 layout's gated attention, the attention kind's kernels: the preparation; the attention over every token below, its warp kernels as before)
        let op = crate::dynamic::qwen35::attention(nq, nk, c.rotary, (0, 0), 0);
        if op.stride != qd || c.head_dim != 256 {
            return err(format!("attention rows of {qd} with heads of {}: the gated attention's are {} with heads of 256", c.head_dim, op.stride));
        }
        let mut at = op.args(n, c.eps);
        (at.src, at.qw, at.kw, at.cs, at.sn, at.qpos, at.kpos) = (self.mix.ptr, sm[0].ptr, sm[1].ptr, self.cos.ptr, self.sin.ptr, pos, pos);
        (at.qo, at.kc, at.vc, at.cap, at.out, at.gate) = (self.qo.ptr, kc, vc, self.ctx as i32, self.core.ptr, self.mix.ptr + 512);
        self.ka.prep(s, &at, nq + nk, (crate::gemma4::G4Pf::default(), 0))?;
        self.lin(&ly.mats[2], self.h.ptr, n, self.ixqk.ptr)?;
        crate::tmark!(s, "attn: indexer product");
        self.kq.ix_prep(s, self.ixqk.ptr, sm[2].ptr, self.cos.ptr, self.sin.ptr, half, pos, self.ixq.ptr, ik, n, c.ix_heads, c.eps)?;
        self.kq.ix_cmp(s, ik, sm[3].ptr, self.cos.ptr, self.sin.ptr, half, pos, pk, n, c.eps)?;
        crate::tmark!(s, "attn: norms, rotary, block keys");
        let qs = nq * c.head_dim;
        // (a few rows, a decode step's or a verification's, always through the picks: the same launches at any position, which a graph replays)
        if n > SPEC_ROWS && start + n <= c.ix_tokens() {
            self.ka.rows(s, &at)?;
        } else {
            let (pmax, ksel) = (self.ctx / c.ix_block, c.ix_budget / c.ix_block);
            if self.ctx > c.ix_tokens() && c.ix_heads == 4 {
                // (on the tensor cores: a prompt chunk's pools up to its last row's, a decode step's every pool the context may hold, its graph replayed at any position, the blocks past a row's pools idle)
                let pools = if n > SPEC_ROWS { (start + n) / c.ix_block } else { pmax };
                self.kq.ix_score(s, self.ixq.ptr, pk, pos, self.scores.ptr, n, pmax, ksel, pools, 1.0 / (c.ix_dim as f32).sqrt())?;
                #[cfg(feature = "dev")]
                if start % 8192 == 0 && std::env::var("GLYD_IX_CHECK").is_ok() {
                    // (a development check: the scalar kernel's scores of the same rows beside these)
                    let other = Buf::new(n * pmax * 4)?;
                    self.kg.idx_score(
                        s,
                        self.ixq.ptr,
                        (c.ix_heads * c.ix_dim) as i32,
                        self.ones.ptr,
                        0,
                        pk,
                        pos,
                        other.ptr,
                        pmax as i32,
                        c.ix_heads as i32,
                        c.ix_dim as i32,
                        c.ix_block as i32,
                        ksel as i32,
                        1.0 / (c.ix_dim as f32).sqrt(),
                        1.0,
                        n as u32,
                    )?;
                    self.dev.sync()?;
                    let (mut a, mut b) = (vec![0f32; n * pmax], vec![0f32; n * pmax]);
                    self.scores.read(0, &mut a)?;
                    other.read(0, &mut b)?;
                    let (mut worst, mut big) = (0f32, 0f32);
                    for i in 0..n {
                        let pi = (start + i + 1) / c.ix_block;
                        if pi > ksel {
                            for p in 0..pi {
                                worst = worst.max((a[i * pmax + p] - b[i * pmax + p]).abs());
                                big = big.max(b[i * pmax + p].abs());
                            }
                        }
                    }
                    eprintln!("ix check: attention layer {}, rows {start}..{}: largest difference {worst:.3e} of scores up to {big:.3e}", ly.idx, start + n);
                }
            } else if self.ctx > c.ix_tokens() {
                self.kg.idx_score(
                    s,
                    self.ixq.ptr,
                    (c.ix_heads * c.ix_dim) as i32,
                    self.ones.ptr,
                    0,
                    pk,
                    pos,
                    self.scores.ptr,
                    pmax as i32,
                    c.ix_heads as i32,
                    c.ix_dim as i32,
                    c.ix_block as i32,
                    ksel as i32,
                    1.0 / (c.ix_dim as f32).sqrt(),
                    1.0,
                    n as u32,
                )?;
            }
            crate::tmark!(s, "attn: indexer scores");
            self.kq.select(s, self.scores.ptr, pmax, pos, self.sel.ptr, self.cnt.ptr, c.ix_tokens(), c.ix_block, ksel, n)?;
            crate::tmark!(s, "attn: select");
            #[cfg(feature = "dev")]
            if n >= 64 && start % 8192 == 0 && std::env::var("GLYD_SEL_UNION").is_ok() {
                self.sel_union(ly.idx, n, start)?;
            }
            if n > 8 && nq == 24 && nk == 2 && !crate::tune!("GLYD_ATTN_SCALAR").is_some_and(|v| v == "1") {
                self.kq.attn_tc(s, self.qo.ptr, qs, self.mix.ptr, qd, kc, vc, self.sel.ptr, self.cnt.ptr, c.ix_tokens(), self.core.ptr, qs, n)?;
                #[cfg(feature = "dev")]
                if start % 8192 == 0 && std::env::var("GLYD_ATTN_CHECK").is_ok() {
                    // (a development check: the scalar kernel's output of the same rows beside this one)
                    let other = Buf::new(n * qs * 2)?;
                    self.kq.attn_sel(
                        s,
                        self.qo.ptr,
                        qs,
                        self.mix.ptr,
                        qd,
                        kc,
                        vc,
                        self.sel.ptr,
                        self.cnt.ptr,
                        c.ix_tokens(),
                        other.ptr,
                        qs,
                        nq,
                        nk,
                        self.part.ptr,
                        1,
                        n,
                    )?;
                    self.dev.sync()?;
                    let (mut a, mut b) = (vec![0u16; n * qs], vec![0u16; n * qs]);
                    self.core.read(0, &mut a)?;
                    other.read(0, &mut b)?;
                    let f = |x: u16| f32::from_bits((x as u32) << 16);
                    let (mut worst, mut big, mut sq, mut sd) = (0f32, 0f32, 0f64, 0f64);
                    for (x, y) in a.iter().zip(&b) {
                        let (x, y) = (f(*x), f(*y));
                        worst = worst.max((x - y).abs());
                        big = big.max(y.abs());
                        sq += (y as f64).powi(2);
                        sd += ((x - y) as f64).powi(2);
                    }
                    eprintln!(
                        "attn check: attention layer {}, rows {start}..{}: largest difference {worst:.3e} of outputs up to {big:.3e}, relative {:.3e}",
                        ly.idx,
                        start + n,
                        (sd / sq.max(1e-30)).sqrt()
                    );
                }
            } else if n <= 8 && nq == 24 && nk == 2 && !crate::tune!("GLYD_ATTN_SCALAR").is_some_and(|v| v == "1") {
                // (a decode step's rows: a 16-token step a warp, 8 warps a block, merged)
                self.kq.attn_dec(
                    s,
                    self.qo.ptr,
                    qs,
                    self.mix.ptr,
                    qd,
                    kc,
                    vc,
                    self.sel.ptr,
                    self.cnt.ptr,
                    c.ix_tokens(),
                    self.core.ptr,
                    qs,
                    self.part.ptr,
                    dec_wpb(),
                    n,
                )?;
            } else {
                let splits = if n <= 8 { SEL_SPLITS } else { 1 };
                self.kq.attn_sel(
                    s,
                    self.qo.ptr,
                    qs,
                    self.mix.ptr,
                    qd,
                    kc,
                    vc,
                    self.sel.ptr,
                    self.cnt.ptr,
                    c.ix_tokens(),
                    self.core.ptr,
                    qs,
                    nq,
                    nk,
                    self.part.ptr,
                    splits,
                    n,
                )?;
            }
        }
        crate::tmark!(s, "attn: attention");
        self.lin(&ly.mats[1], self.core.ptr, n, self.o.ptr)?;
        crate::tmark!(s, "attn: o product");
        Ok(())
    }

    /// A development look at the indexer's picks of a chunk (GLYD_SEL_UNION=1): for groups of T consecutive rows, how many pools of 4 tokens the group's picks cover together against one row's.
    #[cfg(feature = "dev")]
    fn sel_union(&self, layer: usize, n: usize, start: usize) -> Result<()> {
        let tmax = self.w.cfg.ix_tokens();
        self.dev.sync()?;
        let (mut cnt, mut sel) = (vec![0i32; n], vec![0i32; n * tmax]);
        self.cnt.read(0, &mut cnt)?;
        self.sel.read(0, &mut sel)?;
        let pools = |i: usize| -> std::collections::BTreeSet<i32> { sel[i * tmax..i * tmax + cnt[i] as usize].iter().map(|&k| k / 4).collect() };
        let one = (0..n).map(|i| pools(i).len()).sum::<usize>() as f64 / n as f64;
        let mut line = format!("sel union: attention layer {layer}, rows {start}..{}: {one:.0} pools a row", start + n);
        for t in [4usize, 8, 16, 32] {
            let groups: Vec<usize> = (0..n / t).map(|g| (g * t..g * t + t).flat_map(pools).collect::<std::collections::BTreeSet<i32>>().len()).collect();
            line += &format!(", T={t} {:.2}x", groups.iter().sum::<usize>() as f64 / groups.len().max(1) as f64 / one);
        }
        eprintln!("{line}");
        Ok(())
    }

    /// The token mixer alone on the n rows in `h` into `o` (a development check's).
    ///
    /// # Safety
    /// As `gr_mix`.
    #[cfg_attr(not(feature = "dev"), allow(dead_code))] // development tools only
    pub(crate) unsafe fn mixer_only(&self, ly: &LayerQx, n: usize, pos: u64, start: usize) -> Result<()> {
        match ly.kind {
            QxKind::Gdn => self.gdn(ly, n),
            QxKind::Attn => self.attn(ly, n, pos, start),
        }
    }

    /// The token mixer's half of a layer on the n streams in `x`.
    ///
    /// # Safety
    /// As `gr_mix`.
    pub(crate) unsafe fn mixer_half(&self, ly: &LayerQx, n: usize, pos: u64, start: usize) -> Result<()> {
        self.mixer_half_x(ly, n, pos, start, self.x.ptr)
    }

    /// `mixer_half` on the n streams at `x`.
    ///
    /// # Safety
    /// As `gr_mix`.
    unsafe fn mixer_half_x(&self, ly: &LayerQx, n: usize, pos: u64, start: usize, x: u64) -> Result<()> {
        self.mixer_half_defer(ly, n, pos, start, x, false)
    }

    /// `mixer_half_x`; where `defer`, the streams' update after the mixer left to the next Gated Residual (`gr_mix_post` with `(o, inj[0])`).
    ///
    /// # Safety
    /// As `gr_mix`.
    unsafe fn mixer_half_defer(&self, ly: &LayerQx, n: usize, pos: u64, start: usize, x: u64, defer: bool) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        crate::tmark!(s, "mixer half start");
        self.gr_mix(&ly.gr[0], x, n, self.h.ptr, self.inj[0].ptr)?;
        crate::tmark!(s, "gated residual (mixer)");
        match ly.kind {
            QxKind::Gdn => self.gdn(ly, n)?,
            QxKind::Attn => self.attn(ly, n, pos, start)?,
        }
        crate::tmark!(s, "token mixer");
        if !defer {
            self.kq.hc_post(s, x, self.o.ptr, self.inj[0].ptr, n, c.hidden)?;
            crate::tmark!(s, "mHC update (mixer)");
        }
        Ok(())
    }

    /// Layer `ly` on the first `t` rows of `x` (positions `start` ..; `pos` holds them), layer by layer: its n-gram half and mixer half on each chunk of rows in turn, the experts' input and the block's front on each, the
    /// block's experts once for all the rows, then its output and the streams' update on each chunk.
    ///
    /// # Safety
    /// As `gr_mix`; `x`, `pos`, `ple_e` and `inj[1]` hold the t rows (at most `whole`), the block takes prompts layer by layer.
    unsafe fn layer_whole(&self, ly: &LayerQx, t: usize, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        // (the prompt's chunks: `pchunk` rows, the block's front of the experts a sub-chunk of the engine's chunk at a time, the rows its scratch holds)
        let (ch, hw, ple_row) = (self.pchunk.get(), c.hc * c.hidden, c.ple.as_ref().map_or(0, |p| p.dim * 2));
        let moe = ly.moe.as_ref().ok_or_else(|| Error("a layer without its block of experts".into()))?;
        let parts: Vec<(usize, usize)> = (0..t).step_by(ch).map(|j| (j, ch.min(t - j))).collect();
        let (xw, posw, plew, injw) = self.whole_ptrs()?;
        let xr = |j: usize| xw + (j * hw * 2) as u64;
        let inj = |j: usize| injw + (j * c.hc * 4) as u64;
        for &(j, n) in &parts {
            if let Some(p) = &ly.ple {
                self.ple_half_x(p, n, xr(j), plew + (j * ple_row) as u64)?;
            }
            self.mixer_half_x(ly, n, posw + (j * 4) as u64, start + j, xr(j))?;
        }
        for &(j, n) in &parts {
            crate::tmark!(s, "feed-forward half start");
            self.gr_mix(&ly.gr[1], xr(j), n, self.h.ptr, inj(j))?;
            crate::tmark!(s, "mHC mix (feed-forward)");
            for i in (0..n).step_by(self.front_rows()) {
                moe.whole_front(self, self.h.ptr + (i * c.hidden * 2) as u64, self.front_rows().min(n - i), j + i)?;
            }
        }
        moe.whole_experts(self, t)?;
        for &(j, n) in &parts {
            moe.whole_back(self, self.o.ptr, n, j)?;
            self.kq.hc_post(s, xr(j), self.o.ptr, inj(j), n, c.hidden)?;
            crate::tmark!(s, "mHC update (feed-forward)");
        }
        Ok(())
    }

    /// The rows of `tokens` (at most `whole`) through the model layer by layer after the tokens already in the sequence (`layer_whole`), their streams left in `x`; where `head`, the final mixer of the last row, the LM
    /// head and the argmax.
    ///
    /// # Safety
    /// The stream is idle (the positions and the staging buffer are written here).
    unsafe fn forward_whole(&mut self, tokens: &[u32], head: bool) -> Result<()> {
        let (t, ch) = (tokens.len(), self.pchunk.get());
        let (hid, hw, vocab, ple_row) =
            (self.w.cfg.hidden, self.w.cfg.hc * self.w.cfg.hidden, self.w.cfg.vocab, self.w.cfg.ple.as_ref().map_or(0, |p| p.dim * 2));
        if t == 0 || t > self.whole_now() {
            return err(format!("{t} rows taken layer by layer, where the buffers hold {}", self.whole_now()));
        }
        let s = self.dev.stream;
        crate::tmark!(s, "whole: start");
        self.dev.sync()?;
        let (xw, _, plew, _) = self.whole_ptrs()?;
        self.whole_pos(&(0..t).map(|i| (self.n + i) as i32).collect::<Vec<i32>>())?;
        let ctx0 = self.ng_ctx;
        let n0 = self.n;
        // (a model split over GPUs: the part before this one takes the rows through its layers, and its streams come here)
        if let Some(fx) = self.on_front_mut(|f| {
            f.n = n0;
            f.forward_whole(tokens, false)?;
            Ok(f.whole_ptrs()?.0)
        })? {
            self.take_rows(fx, xw, t)?;
        } else {
            // the rows' embeddings spread over the streams, a chunk at a time (the host's staging buffer holds a chunk)
            for j in (0..t).step_by(ch) {
                let n = ch.min(t - j);
                self.dev.sync()?;
                self.stage_to(n0 + j, &tokens[j..j + n], None)?;
                self.kg.streams(s, self.o.ptr, xw + (j * hw * 2) as u64, n as u32, hid as i32)?;
            }
        }
        crate::tmark!(s, "whole: staged");
        // their n-gram rows gathered by the host's threads while the layers before the one that reads them run (a new prompt's rows are reads from the disk: ~0.7 s for 32K tokens)
        let me = &*self;
        std::thread::scope(|sc| -> Result<()> {
            let mut gather = me.w.ngram.as_ref().filter(|_| me.front.is_none()).map(|g| {
                sc.spawn(move || {
                    let mut rows = vec![0u8; t * ple_row];
                    g.gather(ctx0, tokens, &mut rows);
                    rows
                })
            });
            for l in me.span.0..me.span.1 {
                let ly = me.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
                if let Some(g) = gather.take_if(|_| ly.ple.is_some()) {
                    let rows = g.join().map_err(|_| Error("the n-gram rows' gather panicked".into()))?;
                    crate::tmark!(s, "whole: n-gram rows");
                    // (pageable memory: the copy has read the rows when it returns)
                    copy_h2d_async(plew, rows.as_ptr() as *const _, rows.len(), s)?;
                }
                me.layer_whole(ly, t, me.n)?;
                #[cfg(feature = "dev")]
                PEEK.with(|p| -> Result<()> {
                    if let Some((at, rows)) = p.borrow_mut().as_mut() {
                        me.dev.sync()?;
                        for &q in at.iter().filter(|&&q| q >= me.n && q < me.n + t) {
                            let mut v = vec![0u16; hw];
                            me.wb
                                .borrow()
                                .as_ref()
                                .ok_or_else(|| Error("no prompt is being taken layer by layer".into()))?
                                .x
                                .read((q - me.n) * hw * 2, &mut v)?;
                            rows.extend(v);
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(())
        })?;
        if head {
            let (Some(mixer), Some(lm)) = (&self.w.mixer, &self.w.head) else { return err("the final mixer or the LM head is not loaded") };
            let last = xw + ((t - 1) * hw * 2) as u64;
            self.gr_mix(mixer, last, 1, self.hlast.ptr, 0)?;
            self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
            self.kern.argmax_rows(s, self.logits.ptr, vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        }
        Ok(())
    }

    /// The experts' half of a layer on the n streams in `x`.
    ///
    /// # Safety
    /// As `gr_mix`.
    pub(crate) unsafe fn mlp_half(&self, ly: &LayerQx, n: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        crate::tmark!(s, "feed-forward half start");
        self.gr_mix(&ly.gr[1], self.x.ptr, n, self.h.ptr, self.inj[1].ptr)?;
        crate::tmark!(s, "mHC mix (feed-forward)");
        ly.moe.as_ref().ok_or_else(|| Error("a layer without its block of experts".into()))?.forward(self, self.h.ptr, n, self.o.ptr)?;
        crate::tmark!(s, "experts or MLP");
        self.kq.hc_post(s, self.x.ptr, self.o.ptr, self.inj[1].ptr, n, c.hidden)?;
        crate::tmark!(s, "mHC update (feed-forward)");
        Ok(())
    }

    /// The token rows (in `o`) spread over the streams (on a part of a model split over GPUs: the part before it run, its streams taken), then the layers; where `head`, the final mixer of the last row, the LM head and
    /// the argmax.
    unsafe fn forward(&self, n: usize, pos: u64, start: usize, head: bool) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        match self.on_front(|f| f.forward(n, f.pos.ptr, start, false).map(|()| f.x.ptr))? {
            Some(fx) => self.take_rows(fx, self.x.ptr, n)?,
            None => self.kg.streams(s, self.o.ptr, self.x.ptr, n as u32, c.hidden as i32)?,
        }
        for l in self.span.0..self.span.1 {
            let ly = self.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} is not loaded")))?;
            if let Some(p) = &ly.ple {
                self.ple_half(p, n)?;
            }
            self.mixer_half(ly, n, pos, start)?;
            self.mlp_half(ly, n)?;
            if let Some(t) = &self.trace {
                t(self, l, n, start);
            }
        }
        if head {
            let (Some(mixer), Some(lm)) = (&self.w.mixer, &self.w.head) else { return err("the final mixer or the LM head is not loaded") };
            let last = self.x.ptr + ((n - 1) * c.hc * c.hidden * 2) as u64;
            self.gr_mix(mixer, last, 1, self.hlast.ptr, 0)?;
            self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        }
        Ok(())
    }

    /// The final mixer's output (bf16 [n, hidden]) of each of the last call's `n` rows: a development check's view.
    pub fn final_rows(&self, n: usize) -> Result<Vec<u16>> {
        let c = &self.w.cfg;
        let mixer = self.w.mixer.as_ref().ok_or_else(|| Error("the final mixer is not loaded".into()))?;
        let out = Buf::new(n * c.hidden * 2)?;
        // SAFETY: live buffers of the engine, sized for a chunk; synchronized before the read.
        unsafe { self.gr_mix(mixer, self.x.ptr, n, out.ptr, 0)? };
        self.dev.sync()?;
        let mut v = vec![0u16; n * c.hidden];
        out.read(0, &mut v)?;
        Ok(v)
    }

    /// The LM head's logits (bf16 bits, [n, vocab]) of `n` rows of final hidden states at `h`: a development check's view.
    pub fn head_rows(&self, h: &[u16]) -> Result<Vec<u16>> {
        let c = &self.w.cfg;
        let n = h.len() / c.hidden;
        let lm = self.w.head.as_ref().ok_or_else(|| Error("the LM head is not loaded".into()))?;
        let (hb, out) = (Buf::new(h.len() * 2)?, Buf::new(n * c.vocab * 2)?);
        hb.write(0, h)?;
        let mut v = vec![0u16; n * c.vocab];
        for r in (0..n).step_by(8) {
            let m = 8.min(n - r);
            // SAFETY: live buffers of those sizes; synchronized before the read.
            unsafe { self.lin_packed(lm, hb.ptr + (r * c.hidden * 2) as u64, m, out.ptr + (r * c.vocab * 2) as u64)? };
        }
        self.dev.sync()?;
        out.read(0, &mut v)?;
        Ok(v)
    }

    /// The rows of `tokens` (the embedding's into `o`, the n-gram embedding's into `ple_e`) staged on the host and copied up; the n-gram context moves on.
    ///
    /// # Safety
    /// The stream has finished with the staging buffer (synchronized since its last use).
    pub(crate) unsafe fn stage(&mut self, tokens: &[u32]) -> Result<()> {
        let at = self.n;
        self.stage_rows(at, tokens)
    }

    /// `stage` of the tokens at rows `at`.. of the sequence (a model split over GPUs: on its first part, where its layers start; the row from this part, which keeps the sequence's length).
    ///
    /// # Safety
    /// As `stage`.
    unsafe fn stage_rows(&mut self, at: usize, tokens: &[u32]) -> Result<()> {
        if self.on_front_mut(|f| f.stage_rows(at, tokens))?.is_some() {
            return Ok(());
        }
        self.stage_to(at, tokens, Some(self.ple_e.ptr))
    }

    /// `stage` of the tokens at rows `at`.. of the sequence, with the n-gram rows copied to `ple_dst` (device, n rows), or none gathered (None: the caller gathers them). An image's rows
    /// (`set_images`) take the place of its tokens' embedding.
    ///
    /// # Safety
    /// As `stage`.
    unsafe fn stage_to(&mut self, at: usize, tokens: &[u32], ple_dst: Option<u64>) -> Result<()> {
        let c = &self.w.cfg;
        let n = tokens.len();
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= c.vocab) {
            return err(format!("token {t}: the vocabulary has {}", c.vocab));
        }
        let rb = c.hidden * 2;
        let ple = c.ple.as_ref().map_or(0, |p| p.dim * 2);
        let buf = std::slice::from_raw_parts_mut(self.host.ptr as *mut u8, n * (rb + ple));
        fill_rows(&mut buf[..n * rb], rb, |i, o| o.copy_from_slice(self.w.embed.row(tokens[i] as usize)));
        self.image_rows(at, n, &mut buf[..n * rb]);
        copy_h2d_async(self.o.ptr, buf.as_ptr() as *const _, n * rb, self.dev.stream)?;
        if let (Some(g), Some(ple_dst)) = (&self.w.ngram, ple_dst) {
            let rows = &mut buf[n * rb..n * (rb + ple)];
            g.gather(self.ng_ctx, tokens, rows);
            copy_h2d_async(ple_dst, rows.as_ptr() as *const _, n * ple, self.dev.stream)?;
        }
        let mut hist = vec![self.ng_ctx[0], self.ng_ctx[1]];
        hist.extend_from_slice(tokens);
        self.ng_ctx = [hist[hist.len() - 2], hist[hist.len() - 1]];
        Ok(())
    }

    /// The images of the sequence `ids` (the whole of it, the state's tokens and the next prompt's: an image's place a run of `image_token`, one image of `imgs` each in order, its tokens'
    /// grid (rows, columns) and the tower's rows (bf16 [tokens, hidden]; None for one the state holds already)): their rows staged in place of the token's embedding, and the rotary's
    /// tables turned to the sequence's M-RoPE positions from its first image on (the rows past `ids` continuing from its last position: the reply's). No image: the plain tables again.
    pub fn set_images(&mut self, ids: &[u32], image_token: u32, imgs: Vec<((usize, usize), Option<Vec<u16>>)>) -> Result<()> {
        // (a model split over GPUs: on every part, each with its own rotary tables; the first stages the rows, the last's MTP head reads them)
        if self.front.is_some() {
            let copy = imgs.clone();
            self.on_front_mut(|f| f.set_images(ids, image_token, copy))?;
        }
        let hid = self.w.cfg.hidden;
        // (no image: the token's own rows, whatever `ids` holds, and the plain tables where a prompt's positions were in them)
        let (pos, next, first) = if imgs.is_empty() {
            self.img_rows.borrow_mut().clear();
            (Vec::new(), 0, None)
        } else {
            let grids: Vec<(usize, usize)> = imgs.iter().map(|x| x.0).collect();
            let (pos, next) = crate::vision::positions(ids, image_token, &grids)?;
            let starts: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == image_token && (i == 0 || ids[i - 1] != image_token)).collect();
            let mut rows = Vec::new();
            for (((r, c), img), &start) in imgs.into_iter().zip(&starts) {
                if let Some(v) = img {
                    if v.len() != r * c * hid {
                        return err(format!("an image of {} rows for {} places", v.len() / hid, r * c));
                    }
                    rows.push((start, v));
                }
            }
            *self.img_rows.borrow_mut() = rows;
            (pos, next, starts.first().copied())
        };
        let Some(from) = self.rope_from.get().into_iter().chain(first).min() else { return Ok(()) };
        if from >= self.ctx {
            return err(format!("an image at row {from}, where a sequence may reach {}", self.ctx));
        }
        let half = self.w.cfg.rotary / 2;
        let at: Vec<[u32; 3]> = (from..self.ctx)
            .map(|r| {
                if r < pos.len() {
                    pos[r]
                } else if first.is_some() {
                    [next + (r - pos.len()) as u32; 3]
                } else {
                    [r as u32; 3]
                }
            })
            .collect();
        let (cs, sn) = crate::vision::rope_rows(self.w.cfg.theta, half, &at);
        let off = (from * half * 2) as u64;
        // SAFETY: rows from..ctx of the tables (ctx rows of `half`), in the stream's order; the pageable rows are read before the calls return.
        unsafe {
            copy_h2d_async(self.cos.ptr + off, cs.as_ptr() as *const _, cs.len() * 2, self.dev.stream)?;
            copy_h2d_async(self.sin.ptr + off, sn.as_ptr() as *const _, sn.len() * 2, self.dev.stream)?;
        }
        self.dev.sync()?;
        self.rope_from.set(first);
        Ok(())
    }

    /// The token chosen after the work queued (the largest logit).
    fn read_sample(&mut self) -> Result<Sample> {
        let at = self.host.ptr as usize + PREFILL_CHUNK.max(self.chunk) * (self.w.cfg.hidden + self.w.cfg.ple.as_ref().map_or(0, |p| p.dim)) * 2;
        // SAFETY: the pinned buffer's last 4,096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(at as *mut _, self.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((at + 8) as *mut _, self.val.ptr, 4, self.dev.stream)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*(at as *const i32), *((at + 8) as *const f32)) };
        let greedy = Sample { token: token as u32, logit, logprob: None };
        Ok(self.draw(self.logits.ptr, self.val.ptr, vec![greedy], &[])?.remove(0))
    }

    /// The tokens of `greedy.len()` rows of logits (bf16 at `logits`, each row's largest as f32 at `maxv`) where the request samples (`sampler`):
    /// drawn in row order, stopping after the first that is not its row's guess in `guesses` (the rows past it are not used); else `greedy`.
    fn draw(&self, logits: u64, maxv: u64, greedy: Vec<Sample>, guesses: &[u32]) -> Result<Vec<Sample>> {
        use crate::kern::TOPK;
        let mut sm = self.sampler.borrow_mut();
        let Some(sampler) = sm.as_mut() else { return Ok(greedy) };
        let (rows, vocab, st) = (greedy.len(), self.w.cfg.vocab, self.dev.stream);
        let (tok, val, sum, stage) = &self.samp;
        let need = sampler.top_need().filter(|_| vocab.is_multiple_of(8) && vocab >= TOPK);
        let host = stage.ptr as *mut u8;
        if let Some(nd) = need {
            // SAFETY: the engine's buffers, sized for SPEC_ROWS rows; the copies complete at the synchronize.
            unsafe {
                self.kern.topk_rows(st, logits, vocab as i32, rows as u32, tok.ptr, val.ptr)?;
                // (a grammar's mask for the first row, whose token follows what is said so far: its candidates made again with the mask put on the row)
                if let (Some(m), Some(g)) = (sampler.mask_handle(), &self.maskgpu) {
                    g.apply(&self.kern, &self.dev, &[Some(m)], logits, tok.ptr, val.ptr)?;
                }
                copy_d2h_async(host as *mut _, tok.ptr, 4 * TOPK * rows, st)?;
                copy_d2h_async(host.add(4 * TOPK * SPEC_ROWS) as *mut _, val.ptr, 4 * TOPK * rows, st)?;
                if nd.mass {
                    self.kern.exp_sum_rows(st, logits, vocab as i32, rows as u32, maxv, nd.temperature, sum.ptr)?;
                    copy_d2h_async(host.add(8 * TOPK * SPEC_ROWS) as *mut _, sum.ptr, 8 * rows, st)?;
                }
            }
        }
        self.dev.sync()?;
        let mut out = greedy;
        for r in 0..rows {
            // SAFETY: the pinned bytes were just copied: TOPK tokens (int32) and logits (f32) a row, then a sum (f64) a row.
            let top = need.map(|nd| unsafe {
                let cands = (0..TOPK)
                    .map(|j| (*(host.add(4 * (r * TOPK + j)) as *const i32) as u32, *(host.add(4 * TOPK * SPEC_ROWS + 4 * (r * TOPK + j)) as *const f32)))
                    .collect();
                crate::sampler::Top { cands, mass: nd.mass.then(|| *(host.add(8 * TOPK * SPEC_ROWS + 8 * r) as *const f64)) }
            });
            let row = || -> Result<Vec<f32>> {
                let mut h = vec![0u16; vocab];
                // SAFETY: row r of the logits, live on the engine's stream; synchronized before the read.
                unsafe { copy_d2h_async(h.as_mut_ptr() as *mut _, logits + (r * vocab * 2) as u64, vocab * 2, st)? };
                self.dev.sync()?;
                Ok(h.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
            };
            let t = sampler.sample_with(top.as_ref(), row)?;
            let logit = top.as_ref().and_then(|tp| tp.cands.iter().find(|c| c.0 == t)).map_or(out[r].logit, |c| c.1);
            out[r] = Sample { token: t, logit, logprob: None };
            if r < guesses.len() && t != guesses[r] {
                break;
            }
        }
        Ok(out)
    }

    /// The prompt through the model after the tokens already in the sequence, in chunks; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where a sequence may reach {}", self.n, tokens.len(), self.ctx));
        }
        if self.takes_whole(tokens.len()) {
            // (the buffers made for the prompt's rows, at most `whole` a part, places lent where the GPU has not the room; all freed after)
            let bufs = self.prompt_begin(self.whole.min(tokens.len()))?;
            let mut run = || -> Result<()> {
                let mut at = 0;
                while at < tokens.len() {
                    let n = self.whole.min(tokens.len() - at);
                    // SAFETY: live buffers of the engine and the prompt's; synchronized in the call before its staging.
                    unsafe { self.forward_whole(&tokens[at..at + n], at + n == tokens.len())? };
                    self.n += n;
                    at += n;
                }
                Ok(())
            };
            let r = run();
            self.prompt_end(bufs)?;
            r?;
            return self.read_sample();
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?;
            self.set_pos(&pos)?;
            let last = at + n == tokens.len();
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step, synchronized above.
            unsafe {
                self.stage(&tokens[at..at + n])?;
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
        self.dev.sync()?;
        self.set_pos(&[self.n as i32])?;
        // SAFETY: as `prefill`.
        unsafe {
            self.stage(&[token])?;
            if self.runs_in_parts() {
                self.forward_segmented(1, false, self.n)?;
            } else {
                self.forward(1, self.pos.ptr, self.n, true)?;
            }
        }
        self.n += 1;
        self.read_sample()
    }

    /// Whether a decode step runs in parts: asked for, no development trace hook, every layer's block able to (the parts' before this one too).
    fn runs_in_parts(&self) -> bool {
        self.segmented.get() && self.trace.is_none() && self.parts_able()
    }

    fn parts_able(&self) -> bool {
        (self.span.0..self.span.1).all(|l| self.w.layers[l].as_ref().is_some_and(|l| l.moe.as_ref().is_some_and(|b| b.segmented())))
            && self.front.as_ref().is_none_or(|f| f.parts_able())
    }

    /// Whether a verification of two rows keeps the GDN layers' states after its first in the snapshot's place (`gdn`, `verify`, `commit`): while every GDN layer's rule of a few rows is `qx_gdn_rec_gn`
    /// (GLYD_GDN_ST1=0: the copy before the pass and the rule run again).
    fn gdn_st1(&self) -> bool {
        self.w.cfg.gate_sigmoid
            && crate::tune!("GLYD_GDN_ST1").is_none_or(|v| v != "0")
            && self.w.layers.iter().flatten().filter(|l| l.kind == QxKind::Gdn).all(|l| matches!(l.mats.get(1), Some(Mat::Dense(..))))
    }

    /// The graphs of the parts forgotten (the settings they were made for have changed).
    pub fn forget_segments(&self) {
        self.segs.borrow_mut().clear();
        let _ = self.on_front(|f| {
            f.forget_segments();
            Ok(())
        });
    }

    /// The stretch `key` of GPU work, launched as the graph made of it (the first call captures it by running `f` with the stream capturing; where the GPU has not the room to make the graph, the stretch is
    /// queued as it is, and the next call tries again).
    ///
    /// # Safety
    /// As `forward`; `f` queues the same stretch each time.
    unsafe fn seg(&self, key: usize, f: impl Fn() -> Result<()>) -> Result<()> {
        if crate::dev::kprof_on() {
            // (the kernel census: the stretch queued as it is, each launch timed alone)
            return f();
        }
        let mut segs = self.segs.borrow_mut();
        if segs.len() <= key {
            segs.resize_with(key + 1, || None);
        }
        if segs[key].is_none() {
            match self.dev.capture(&f) {
                Ok(g) => segs[key] = Some(g),
                // (nothing of a capture runs: the stretch is queued now)
                Err(e) if e.msg.starts_with("cuGraphInstantiate") => return f(),
                Err(e) => return Err(e),
            }
        }
        // (a development trace: the graph's own time on the stream between the two markers)
        crate::tmark!(self.dev.stream, "segment: launched");
        segs[key].as_ref().expect("a graph").launch(self.dev.stream)?;
        crate::tmark!(self.dev.stream, "segment: graph ran");
        Ok(())
    }

    /// The decode step with the expert cache in parts: the first layer's always-on work and its block's front as one graph; for each layer the host's part (`MoeBlockQx::host`), then one graph of its back, the streams'
    /// update and the next layer's PLE, mixer half, Gated Residual and front (the last one's: the final mixer, the LM head and the argmax). One wait a layer, the GPU's work in a launch each. `start`: the tokens before the
    /// rows. A model split over GPUs: the part before this one runs its layers so first and sends its streams out in its last graph (`Hop`); this part's first graph takes them in, after the hop's event.
    unsafe fn forward_segmented(&self, n: usize, verify: bool, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let pos = self.pos.ptr;
        if n == 0 || n > SPEC_ROWS || (n > 1 && !verify) {
            return err(format!("a step in parts on {n} rows (a decode step's one, a verification's at most {SPEC_ROWS})"));
        }
        self.on_front(|f| f.forward_segmented(n, verify, start))?;
        let row = c.hc * c.hidden * 2;
        let layers: Vec<&LayerQx> =
            (self.span.0..self.span.1).map(|l| self.w.layers[l].as_ref().ok_or_else(|| Error("a layer is not loaded".into()))).collect::<Result<_>>()?;
        fn block(ly: &LayerQx) -> Result<&dyn MoeBlockQx> {
            ly.moe.as_deref().ok_or_else(|| Error("a layer without its block of experts".into()))
        }
        if self.segs.borrow().is_empty() {
            block(layers[0])?.prepare(self)?;
        }
        for ly in &layers {
            block(ly)?.sync_map()?;
        }
        // (the graphs of each count of rows and kind of pass apart)
        let base = ((n - 1) * 2 + verify as usize) * (layers.len() + 1);
        let mixer_and_front = |ly: &LayerQx| -> Result<()> {
            if let Some(p) = &ly.ple {
                self.ple_half(p, n)?;
            }
            // (the streams' update after the mixer made inside the next Gated Residual's launches)
            self.mixer_half_defer(ly, n, pos, start, self.x.ptr, true)?;
            self.gr_mix_post(&ly.gr[1], self.x.ptr, n, self.h.ptr, self.inj[1].ptr, Some((self.o.ptr, self.inj[0].ptr)))?;
            block(ly)?.front(self, self.h.ptr, n)
        };
        let from = self.front.as_deref().and_then(|f| f.hop.as_ref().map(|h| (f, h)));
        if let Some((_, h)) = from {
            h.ev.wait(s)?;
        }
        self.seg(base, || {
            match from {
                Some((f, h)) if h.peer => crate::dev::copy_d2d_async(self.x.ptr, f.x.ptr, n * row, s)?,
                Some((_, h)) => copy_h2d_async(self.x.ptr, h.buf.ptr, n * row, s)?,
                None => self.kg.streams(s, self.o.ptr, self.x.ptr, n as u32, c.hidden as i32)?,
            }
            mixer_and_front(layers[0])
        })?;
        for i in 0..layers.len() {
            let b = block(layers[i])?;
            b.host_then(self, n, &mut || {
                self.seg(base + i + 1, || {
                    b.back_post(self, n, self.x.ptr, self.inj[1].ptr)?;
                    if let Some(nx) = layers.get(i + 1) {
                        return mixer_and_front(nx);
                    }
                    if let Some(h) = &self.hop {
                        // (a part before another: its streams out to the hop's buffer for the next part to take in)
                        return if h.peer { Ok(()) } else { copy_d2h_async(h.buf.ptr, self.x.ptr, n * row, s) };
                    }
                    let (Some(mixer), Some(lm)) = (&self.w.mixer, &self.w.head) else { return err("the final mixer or the LM head is not loaded") };
                    if verify {
                        // (every row's pick, into the draft buffers: `forward_rows`'s)
                        let sp = self.spec.as_ref().ok_or_else(|| Error("the engine was made without room for a draft (OptsQx::spec)".into()))?;
                        self.gr_mix(mixer, self.x.ptr, n, sp.hl.ptr, 0)?;
                        self.lin_packed(lm, sp.hl.ptr, n, sp.lg.ptr)?;
                        return self.kern.argmax_rows(s, sp.lg.ptr, c.vocab as i32, n as u32, sp.tk.ptr, sp.vl.ptr);
                    }
                    self.gr_mix(mixer, self.x.ptr, 1, self.hlast.ptr, 0)?;
                    self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
                    self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)
                })
            })?;
        }
        if let Some(h) = &self.hop {
            h.ev.record(s)?;
        }
        Ok(())
    }

    /// The layers on the n rows staged, then for every row the final mixer, the LM head and the argmax (into the draft buffers): what a verified draft needs. The rows' last streams stay in `x`.
    unsafe fn forward_rows(&self, n: usize, pos: u64, start: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let sp = self.spec.as_ref().ok_or_else(|| Error("the engine was made without room for a draft (OptsQx::spec)".into()))?;
        self.forward(n, pos, start, false)?;
        let (Some(mixer), Some(lm)) = (&self.w.mixer, &self.w.head) else { return err("the final mixer or the LM head is not loaded") };
        self.gr_mix(mixer, self.x.ptr, n, sp.hl.ptr, 0)?;
        self.lin_packed(lm, sp.hl.ptr, n, sp.lg.ptr)?;
        self.kern.argmax_rows(s, sp.lg.ptr, c.vocab as i32, n as u32, sp.tk.ptr, sp.vl.ptr)
    }

    /// A draft verified in one pass: `tokens` (the last token the model has not seen, then the guesses) at the next positions; the token the model picks after each of them. The GDN layers' and PLE's states go forward
    /// over every row (as decode steps would) after being kept; the sequence is not advanced: `commit` says how many rows stay.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Sample>> {
        let n = tokens.len();
        if self.spec.is_none() {
            return err("the engine was made without room for a draft (OptsQx::spec)");
        }
        if n == 0 || n > SPEC_ROWS.min(self.chunk) {
            return err(format!("a draft of {n} rows, where 1 to {} are verified at once", SPEC_ROWS.min(self.chunk)));
        }
        if self.n + n + 1 > self.ctx {
            return err(format!("{} tokens and {n} more, where a sequence may reach {}", self.n, self.ctx));
        }
        #[cfg_attr(not(feature = "dev"), allow(unused_variables))]
        let s = self.dev.stream;
        self.keep_states(tokens)?;
        let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
        self.dev.sync()?;
        crate::tmark!(s, "verify: before");
        self.set_pos(&pos)?;
        let chunked = self.chunked;
        self.each_part(&mut |e| {
            e.save.set(true);
            e.chunked = false;
        });
        // SAFETY: live buffers of the engine, sized for a chunk; synchronized above.
        let parts = self.runs_in_parts();
        let r = unsafe {
            self.stage(tokens).and_then(|()| if parts { self.forward_segmented(n, true, self.n) } else { self.forward_rows(n, self.pos.ptr, self.n) })
        };
        self.each_part(&mut |e| {
            e.save.set(false);
            e.chunked = chunked;
        });
        r?;
        self.pending.set(Some(n));
        crate::tmark!(s, "verify: last part");
        self.dev.sync()?;
        let sp = self.spec.as_ref().ok_or_else(|| Error("no draft buffers".into()))?;
        let (mut tk, mut vl) = (vec![0i32; n], vec![0f32; n]);
        sp.tk.read(0, &mut tk)?;
        sp.vl.read(0, &mut vl)?;
        let greedy: Vec<Sample> = tk.iter().zip(&vl).map(|(&t, &v)| Sample { token: t as u32, logit: v, logprob: None }).collect();
        self.draw(sp.lg.ptr, sp.vl.ptr, greedy, &tokens[1..])
    }

    /// Of the `rows` of the last `verify`, the first `kept` stay: the GDN layers go back to where they were after the `kept`th (their states from before the pass and the recurrence run again over the kept rows' saved
    /// inputs; the convolutions' states the last three of the old ones and those rows' inputs), PLE's convolution likewise (nine), the n-gram context to the kept tokens.
    pub fn commit(&mut self, rows: usize, kept: usize) -> Result<()> {
        if kept == 0 || kept > rows || rows > SPEC_ROWS {
            return err(format!("{kept} of {rows} rows kept"));
        }
        if self.pending.get() != Some(rows) {
            return err(format!("{kept} of {rows} rows kept, where the pass waiting to be settled is {:?}", self.pending.get()));
        }
        self.pending.set(None);
        if kept < rows {
            self.rewind(rows, kept)?;
        }
        self.n += kept;
        Ok(())
    }

    /// What a verification of `tokens` may go back to kept, on this part of the model and the parts before it: the GDN layers' and PLE's states as they are, the n-gram context and the tokens.
    fn keep_states(&mut self, tokens: &[u32]) -> Result<()> {
        self.on_front_mut(|f| f.keep_states(tokens))?;
        let (s, ng0, st1, n) = (self.dev.stream, self.ng_ctx, self.gdn_st1(), tokens.len());
        if let Some(sp) = &mut self.spec {
            sp.ng0 = ng0;
            sp.toks = tokens.to_vec();
            // SAFETY: the engine's own buffers, of equal sizes; the stream orders the copies before the pass.
            unsafe {
                // (two rows: the GDN layers' pass writes the state after the first row there itself, `gdn`)
                if n != 2 || !st1 {
                    crate::dev::copy_d2d_async(sp.ssm0.ptr, self.ssm.ptr, self.ssm.bytes, s)?;
                }
                crate::dev::copy_d2d_async(sp.conv0.ptr, self.convst.ptr, self.convst.bytes, s)?;
                crate::dev::copy_d2d_async(sp.ple0.ptr, self.plest.ptr, self.plest.bytes, s)?;
            }
        }
        Ok(())
    }

    /// `commit`'s going back to the first `kept` of `rows` rows, on this part of the model and the parts before it.
    fn rewind(&mut self, rows: usize, kept: usize) -> Result<()> {
        self.on_front_mut(|f| f.rewind(rows, kept))?;
        let c = &self.w.cfg;
        let sp = self.spec.as_ref().ok_or_else(|| Error("no draft buffers".into()))?;
        let s = self.dev.stream;
        let (cd, vd, hw) = (c.conv_dim(), c.value_dim(), c.hc * c.hidden);
        let (state_bytes, conv_bytes) = (c.nv * 128 * 128 * 4, 3 * cd * 2);
        // (two rows, the first kept: the states after it are where the pass left them, `gdn`'s st1)
        let st1 = rows == 2 && self.gdn_st1();
        for ly in self.w.layers.iter().flatten().filter(|l| l.kind == QxKind::Gdn) {
            let (mix, xc, ba) = sp.slot(ly.idx, c);
            let state = self.ssm.ptr + (ly.idx * state_bytes) as u64;
            let (conv, old) = (self.convst.ptr + (ly.idx * conv_bytes) as u64, sp.conv0.ptr + (ly.idx * conv_bytes) as u64);
            // SAFETY: the engine's own buffers; the stream orders the copies and the kernel.
            unsafe {
                crate::dev::copy_d2d_async(state, sp.ssm0.ptr + (ly.idx * state_bytes) as u64, state_bytes, s)?;
                if !st1 {
                    self.k35.delta(
                        s,
                        false,
                        xc,
                        cd as i32,
                        ba,
                        2 * c.nv as i32,
                        ly.small[1].ptr,
                        ly.small[2].ptr,
                        state,
                        self.core.ptr,
                        kept as i32,
                        c.nk as i32,
                        c.nv as i32,
                    )?;
                }
                for r in 0..3 {
                    let q = kept + r;
                    let src = if q < 3 { old + (q * cd * 2) as u64 } else { mix + ((q - 3) * (cd + vd) * 2) as u64 };
                    crate::dev::copy_d2d_async(conv + (r * cd * 2) as u64, src, cd * 2, s)?;
                }
            }
        }
        if c.ple.is_some() {
            for r in 0..9 {
                let q = kept + r;
                let src = if q < 9 { sp.ple0.ptr + (q * hw * 2) as u64 } else { sp.gvn.ptr + ((q - 9) * hw * 2) as u64 };
                // SAFETY: as above.
                unsafe { crate::dev::copy_d2d_async(self.plest.ptr + (r * hw * 2) as u64, src, hw * 2, s)? };
            }
        }
        let mut hist = sp.ng0.to_vec();
        hist.extend_from_slice(&sp.toks[..kept]);
        self.ng_ctx = [hist[hist.len() - 2], hist[hist.len() - 1]];
        crate::tmark!(s, "commit: rewind");
        self.dev.sync()
    }

    /// The embedding rows of `tokens` (rows `at`.. of the sequence; an image's rows in place of its tokens') into `dst` (bf16 [n, hidden]), through the pinned staging buffer.
    ///
    /// # Safety
    /// As `stage`.
    unsafe fn stage_embed(&self, at: usize, tokens: &[u32], dst: u64) -> Result<()> {
        let rb = self.w.cfg.hidden * 2;
        let buf = std::slice::from_raw_parts_mut(self.host.ptr as *mut u8, tokens.len() * rb);
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= self.w.cfg.vocab {
                return err(format!("token {t}: the vocabulary has {}", self.w.cfg.vocab));
            }
            buf[i * rb..(i + 1) * rb].copy_from_slice(self.w.embed.row(t as usize));
        }
        self.image_rows(at, tokens.len(), buf);
        copy_h2d_async(dst, buf.as_ptr() as *const _, tokens.len() * rb, self.dev.stream)
    }

    /// The rows of the images (`set_images`) among rows `at`..`at + n` of the sequence written over their places in `buf` (n rows of hidden bf16).
    fn image_rows(&self, at: usize, n: usize, buf: &mut [u8]) {
        let (hid, rb) = (self.w.cfg.hidden, self.w.cfg.hidden * 2);
        for (start, rows) in self.img_rows.borrow().iter() {
            for r in at.max(*start)..(start + rows.len() / hid).min(at + n) {
                buf[(r - at) * rb..(r - at + 1) * rb].copy_from_slice(crate::weights::as_bytes(&rows[(r - start) * hid..(r - start + 1) * hid]));
            }
        }
    }

    /// The MTP head on m rows: `prev` (device, [m, 4 hidden]) the streams each row continues (the main model's last streams of a position, or the head's own of its last draft step), `tokens` the token after each, the rows
    /// at positions pos0.. (the head's cache filled there). The draft for the last row (the token after its token); the head's streams of the last row kept in `mtp_prev` for the next step.
    ///
    /// # Safety
    /// `prev` holds m rows; the stream is idle (the staging buffer and the positions are written here).
    pub(crate) unsafe fn mtp_forward(&self, prev: u64, tokens: &[u32], pos0: usize) -> Result<u32> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let mh = self.w.mtp.as_ref().ok_or_else(|| Error("the engine has no MTP head (OptsQx::mtp)".into()))?;
        let (m, hw) = (tokens.len(), c.hc * c.hidden);
        if m == 0 || m > self.chunk {
            return err(format!("the MTP head on {m} rows, where a chunk is {}", self.chunk));
        }
        self.dev.sync()?;
        crate::tmark!(s, "mtp head: before");
        self.pos.write(0, &(0..m).map(|i| (pos0 + i) as i32).collect::<Vec<i32>>())?;
        // the embedding branch: normed, projected into `o`
        self.stage_embed(pos0 + 1, tokens, self.o.ptr)?;
        self.draft_bufs()?;
        let last = self.x.ptr + ((m - 1) * hw * 2) as u64;
        self.mtp_front(prev, m, pos0)?;
        self.mlp_half(&mh.layer, m)?;
        crate::dev::copy_d2d_async(self.mtp_prev.ptr, last, hw * 2, s)?;
        self.draft_head(last)?;
        crate::tmark!(s, "mtp head: layer and LM head");
        self.dev.sync()?;
        let mut t = [0i32];
        self.tok.read(0, &mut t)?;
        if self.draft_min_p.get() > 0.0 {
            let mut p = [1f32];
            if let Some(b) = self.dpb.borrow().as_ref() {
                b.read(0, &mut p)?;
            }
            self.draft_p.set(p[0]);
        }
        Ok(t[0] as u32)
    }

    /// The head's front on m rows (`mtp_forward`'s, its embedding branch staged in `o`): the streams it continues normed and projected, the embedding's projection added, the head's layer's mixer half.
    ///
    /// # Safety
    /// As `mtp_forward`.
    unsafe fn mtp_front(&self, prev: u64, m: usize, pos0: usize) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let mh = self.w.mtp.as_ref().ok_or_else(|| Error("the engine has no MTP head (OptsQx::mtp)".into()))?;
        let hid = c.hidden;
        self.k35.rmsnorm(s, self.o.ptr, mh.ne.ptr, self.h.ptr, m as u32, hid as i32, c.eps)?;
        self.lin(&mh.fe, self.h.ptr, m, self.o.ptr)?;
        // the streams: normed as one vector, each projected by fc_hidden (4 m rows of hidden), the embedding's projection added to each
        self.kq.norm4(s, prev, mh.nh.ptr, self.xn.ptr, m, hid, c.eps, true)?;
        self.lin(&mh.fh, self.xn.ptr, c.hc * m, self.x.ptr)?;
        self.kq.hc_post(s, self.x.ptr, self.o.ptr, self.ones4.ptr, m, hid)?;
        self.mixer_half(&mh.layer, m, self.pos.ptr, pos0)
    }

    /// The head's draft from its last row's streams at `last`: the final mixer, the LM head (its first `draft_vocab` rows and its special tokens', or all), the argmax and the draft's probability.
    ///
    /// # Safety
    /// As `mtp_forward`; `draft_bufs` made the draft's buffers.
    unsafe fn draft_head(&self, last: u64) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let mh = self.w.mtp.as_ref().ok_or_else(|| Error("the engine has no MTP head (OptsQx::mtp)".into()))?;
        let lm = self.w.head.as_ref().ok_or_else(|| Error("the LM head is not loaded".into()))?;
        self.gr_mix(&mh.mixer, last, 1, self.hlast.ptr, 0)?;
        let (pv, tail) = (self.draft_vocab.get(), c.vocab.saturating_sub(DRAFT_TAIL));
        if pv > 0
            && pv % 64 == 0
            && pv < tail
            && tail % 64 == 0
            && self.skinny_on.get()
            && lm.pack.rows as usize == c.vocab
            && crate::skinny::SkinnyKern::takes(&lm.pack)
        {
            let dl = self.dlg.borrow().as_ref().map_or(0, |b| b.ptr);
            self.skinny.gemv_rows(s, &lm.pack, self.hlast.ptr, dl, 1, 0, pv)?;
            self.skinny.gemv_rows(s, &lm.pack, self.hlast.ptr, dl, 1, tail, c.vocab)?;
            self.kern.argmax_rows(s, dl, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
            self.draft_prob(dl)
        } else {
            self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
            self.draft_prob(self.logits.ptr)
        }
    }

    /// The draft's buffers made (before a graph names them): the logits of part of the LM head (-inf between) and the draft's probability.
    fn draft_bufs(&self) -> Result<()> {
        let v = self.w.cfg.vocab;
        let mut dl = self.dlg.borrow_mut();
        if dl.is_none() {
            let b = Buf::new(v * 2)?;
            b.write(0, &vec![0xFF80u16; v])?;
            *dl = Some(b);
        }
        let mut p = self.dpb.borrow_mut();
        if p.is_none() {
            *p = Some(Buf::zeroed(4)?);
        }
        Ok(())
    }

    /// The draft's probability into `dpb` (where the drafts are kept by it: `draft_min_p`), from its logits lg (bf16 [vocab]) and argmax's value.
    ///
    /// # Safety
    /// lg holds the draft's logits; on the engine's stream.
    unsafe fn draft_prob(&self, lg: u64) -> Result<()> {
        if self.draft_min_p.get() <= 0.0 {
            return Ok(());
        }
        let p = self.dpb.borrow().as_ref().map_or(0, |b| b.ptr);
        self.kq.top_prob(self.dev.stream, lg, self.w.cfg.vocab, self.val.ptr, p)
    }

    /// `mtp_forward` on the rows of a prompt taken layer by layer: the main model's last streams of each row in `x` (replaced by the head's), `tokens` the token after each, at positions pos0..; the head's layer as
    /// `layer_whole` runs it. The draft for the last row; the head's streams of the last row kept in `mtp_prev`.
    ///
    /// # Safety
    /// `x` holds the rows (`forward_whole`'s); the stream is idle.
    unsafe fn mtp_whole(&self, tokens: &[u32], pos0: usize) -> Result<u32> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let mh = self.w.mtp.as_ref().ok_or_else(|| Error("the engine has no MTP head (OptsQx::mtp)".into()))?;
        // (a step at most 1,024 rows: the 4 n rows of the streams' product as the engine has always taken them)
        let (t, ch, hid, hw) = (tokens.len(), self.pchunk.get().min(1024), c.hidden, c.hc * c.hidden);
        if t == 0 || t > self.whole_now() {
            return err(format!("the MTP head on {t} rows taken layer by layer, where the buffers hold {}", self.whole_now()));
        }
        self.dev.sync()?;
        let (xw, _, _, _) = self.whole_ptrs()?;
        self.whole_pos(&(0..t).map(|i| (pos0 + i) as i32).collect::<Vec<i32>>())?;
        for j in (0..t).step_by(ch) {
            let n = ch.min(t - j);
            let x = xw + (j * hw * 2) as u64;
            // (the staging buffer's last copy done)
            self.dev.sync()?;
            self.stage_embed(pos0 + j + 1, &tokens[j..j + n], self.o.ptr)?;
            self.k35.rmsnorm(s, self.o.ptr, mh.ne.ptr, self.h.ptr, n as u32, hid as i32, c.eps)?;
            self.lin(&mh.fe, self.h.ptr, n, self.o.ptr)?;
            self.kq.norm4(s, x, mh.nh.ptr, self.xn.ptr, n, hid, c.eps, true)?;
            self.lin(&mh.fh, self.xn.ptr, c.hc * n, x)?;
            self.kq.hc_post(s, x, self.o.ptr, self.ones4.ptr, n, hid)?;
        }
        self.layer_whole(&mh.layer, t, pos0)?;
        let last = xw + ((t - 1) * hw * 2) as u64;
        crate::dev::copy_d2d_async(self.mtp_prev.ptr, last, hw * 2, s)?;
        let lm = self.w.head.as_ref().ok_or_else(|| Error("the LM head is not loaded".into()))?;
        self.gr_mix(&mh.mixer, last, 1, self.hlast.ptr, 0)?;
        self.lin_packed(lm, self.hlast.ptr, 1, self.logits.ptr)?;
        self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        self.dev.sync()?;
        let mut d = [0i32];
        self.tok.read(0, &mut d)?;
        Ok(d[0] as u32)
    }

    /// Greedy tokens as `generate` makes them, `max_new` of them, drafted by the MTP head: `k` guesses a pass (at most `SPEC_ROWS - 1`), verified in one pass, the guesses the model picks itself kept with its pick after them;
    /// the head is run on the rows a pass keeps (their last streams and the token after each) to fill its cache and draft again. Whatever the head guesses, the tokens are `generate`'s.
    pub fn generate_mtp(&mut self, prompt: &[u32], max_new: usize, k: usize, stop: &[u32]) -> Result<(Vec<Sample>, SpecStats)> {
        let mut r = self.mtp_begin(prompt, max_new, k, stop)?;
        while self.mtp_pass(&mut r)? {}
        Ok((r.out, r.stats))
    }

    /// `generate_mtp` up to its first pass: the prompt, the first token and the head's first drafts; then `mtp_pass` until it says the generation is done (between passes the engine may run another sequence: `swap_seq`).
    pub fn mtp_begin(&mut self, prompt: &[u32], max_new: usize, k: usize, stop: &[u32]) -> Result<MtpRun> {
        if self.w.mtp.is_none() || self.spec.is_none() {
            return err("the engine has no MTP head (OptsQx::mtp)");
        }
        let c_hw = self.w.cfg.hc * self.w.cfg.hidden;
        let k = k.clamp(1, SPEC_ROWS - 1);
        let t0 = std::time::Instant::now();
        let keep = self.keep.replace(0);
        let prompt = if keep > 0 && keep == self.n && keep < prompt.len() {
            &prompt[keep..]
        } else {
            self.reset()?;
            prompt
        };
        if prompt.is_empty() || self.n + prompt.len() + max_new + 2 > self.ctx {
            return err(format!("{} prompt tokens and {max_new} to make, where a sequence may reach {}", self.n + prompt.len(), self.ctx));
        }
        let mut first = Sample { token: 0, logit: 0.0, logprob: None };
        if self.takes_whole(prompt.len()) {
            // the prompt layer by layer in parts of `whole` rows, the head run layer by layer on each part's rows; the buffers made for them (places lent where needed) and freed before the drafts
            let bufs = self.prompt_begin(self.whole.min(prompt.len()))?;
            let mut run = || -> Result<u32> {
                let mut at = 0;
                while at < prompt.len() {
                    let n = self.whole.min(prompt.len() - at);
                    let last = at + n == prompt.len();
                    let start = self.n;
                    // SAFETY: live buffers of the engine and the prompt's; each call synchronizes before its staging.
                    unsafe {
                        self.forward_whole(&prompt[at..at + n], last)?;
                        if last {
                            first = self.read_sample()?;
                        }
                        let toks: Vec<u32> =
                            if last { prompt[at + 1..].iter().copied().chain([first.token]).collect() } else { prompt[at + 1..at + 1 + n].to_vec() };
                        let d = self.mtp_whole(&toks, start)?;
                        if last {
                            self.n = start + n;
                            return Ok(d);
                        }
                    }
                    self.n += n;
                    at += n;
                }
                err("an empty prompt")
            };
            let r = run();
            self.prompt_end(bufs)?;
            let d = r?;
            first.logprob = None;
            return self.mtp_start(first, d, k, max_new, stop, t0.elapsed().as_millis() as u32);
        }
        // the prompt in chunks, the head run on each chunk's rows (a position's last streams and the token after it; the last row waits for the model's pick)
        let mut at = 0;
        while at < prompt.len() {
            let n = self.chunk.min(prompt.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?;
            self.set_pos(&pos)?;
            let last = at + n == prompt.len();
            let start = self.n;
            // SAFETY: live buffers of the engine, sized for a chunk; synchronized above and in `mtp_forward`.
            unsafe {
                self.stage(&prompt[at..at + n])?;
                self.forward(n, self.pos.ptr, start, last)?;
                if last {
                    first = self.read_sample()?;
                }
                crate::dev::copy_d2d_async(self.hs.ptr, self.x.ptr, n * c_hw * 2, self.dev.stream)?;
                let toks: Vec<u32> = if last { prompt[at + 1..].iter().copied().chain([first.token]).collect() } else { prompt[at + 1..at + 1 + n].to_vec() };
                let d = self.mtp_forward(self.hs.ptr, &toks, start)?;
                if last {
                    first.logprob = None;
                    self.n = start + n;
                    // (`d` is the first guess: after the prompt's last token and the model's pick)
                    return self.mtp_start(first, d, k, max_new, stop, t0.elapsed().as_millis() as u32);
                }
            }
            self.n += n;
            at += n;
        }
        err("an empty prompt")
    }

    /// After the prompt: `first` the model's pick after it, `d1` the head's first guess after that; the head's drafts of the first pass.
    fn mtp_start(&mut self, first: Sample, d1: u32, k: usize, max_new: usize, stop: &[u32], prompt_ms: u32) -> Result<MtpRun> {
        let out = vec![first];
        self.emit_tokens(&out);
        let stats = SpecStats { prompt_ms, ..SpecStats::default() };
        // (a development trace keeps the passes' markers alone)
        #[cfg(feature = "dev")]
        crate::trace::restart();
        if self.kprof_passes.replace(false) {
            crate::dev::kprof_start();
        }
        let mut drafts = vec![d1];
        // (with `draft_min_p`: a draft beyond the first kept only while the head's probabilities of the drafts so far multiplied are at least that, the drafts ending at the first that leaves them below)
        let min_p = self.draft_min_p.get();
        let mut sure = self.draft_p.get();
        // SAFETY: the head's streams of its last step are in `mtp_prev`; every call below synchronizes before its staging.
        unsafe {
            while drafts.len() < k.min(max_new.saturating_sub(1)).min(self.ctx.saturating_sub(self.n + 3)).max(1) {
                let pos = self.n - 1 + drafts.len();
                let d = self.mtp_forward(self.mtp_prev.ptr, &[drafts[drafts.len() - 1]], pos)?;
                sure *= self.draft_p.get();
                if min_p > 0.0 && sure < min_p {
                    break;
                }
                drafts.push(d);
            }
        }
        Ok(MtpRun { out, stats, drafts, k, max_new, stop: stop.to_vec() })
    }

    /// One pass of a drafted generation (`mtp_begin`'s): the drafts verified, the tokens kept sent to the sink, the head's next drafts. False where the generation is done (max_new, a stop token, the context's end,
    /// `halt`): no pass was run then, or this one ended it.
    pub fn mtp_pass(&mut self, r: &mut MtpRun) -> Result<bool> {
        let c_hw = self.w.cfg.hc * self.w.cfg.hidden;
        let MtpRun { out, stats, drafts, k, max_new, stop } = r;
        let (k, max_new, min_p) = (*k, *max_new, self.draft_min_p.get());
        if !(out.len() < max_new && !stop.contains(&out[out.len() - 1].token) && self.n + 2 <= self.ctx && !self.halt.get()) {
            return Ok(false);
        }
        let mut guesses = std::mem::take(drafts);
        guesses.truncate((max_new - out.len()).min(self.ctx - self.n - 2));
        let mut rows = vec![out[out.len() - 1].token];
        rows.extend_from_slice(&guesses);
        let n0 = self.n;
        let picks = self.verify(&rows)?;
        let right = guesses.iter().zip(&picks).take_while(|(g, p)| **g == p.token).count();
        // (the tokens this pass adds: its right picks, up to a stop token or max_new; the state keeps exactly the rows read before them, so it
        // holds the prompt and every token made but the last, which a conversation's next request reuses)
        let (mut m, mut last) = (0, out[out.len() - 1].token);
        for p in &picks[..=right] {
            if out.len() + m >= max_new || stop.contains(&last) {
                break;
            }
            m += 1;
            last = p.token;
        }
        self.commit(rows.len(), m.max(1))?;
        stats.passes += 1;
        stats.drafted += guesses.len();
        stats.accepted += right;
        let before = out.len();
        out.extend_from_slice(&picks[..m]);
        self.emit_tokens(&out[before..]);
        if out.len() >= max_new || stop.contains(&out[out.len() - 1].token) {
            return Ok(false);
        }
        // the head on the kept rows: each one's last streams and the token after it (the model's pick, for the last)
        let mut toks: Vec<u32> = rows[1..=right].to_vec();
        toks.push(picks[right].token);
        // SAFETY: `x` holds the pass's rows' last streams (verify's forward_rows leaves them); copied before the head uses `x`.
        unsafe {
            crate::dev::copy_d2d_async(self.hs.ptr, self.x.ptr, toks.len() * c_hw * 2, self.dev.stream)?;
            let mut d = self.mtp_forward(self.hs.ptr, &toks, n0)?;
            drafts.push(d);
            let mut sure = self.draft_p.get();
            while drafts.len() < k.min(max_new - out.len()).min(self.ctx.saturating_sub(self.n + 3)).max(1) {
                d = self.mtp_forward(self.mtp_prev.ptr, &[d], n0 + right + drafts.len())?;
                sure *= self.draft_p.get();
                if min_p > 0.0 && sure < min_p {
                    break;
                }
                drafts.push(d);
            }
        }
        Ok(true)
    }

    /// `tokens` to the streamed request's sink, if there is one (`emit`).
    pub fn emit_tokens(&self, tokens: &[Sample]) {
        if let Some(f) = self.emit.borrow_mut().as_mut() {
            if !tokens.is_empty() {
                f(&tokens.iter().map(|s| s.token).collect::<Vec<_>>());
            }
        }
    }

    /// Greedy tokens: the prompt, then up to `max_new` tokens (stopping at an end token unless `ignore_eos`).
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

// ---- a layer's experts on the GPU (the development checks', a model small enough) ----------------------------------------------------------------------------------------------------------

/// The core's layer for the model's mixture-of-experts layers: softmax over bf16 logits, the top k renormalised, SwiGLU without a clamp.
pub fn moe_cfg(c: &ConfigQx) -> MoeCfg {
    MoeCfg {
        experts: c.experts,
        router_hidden: c.hidden,
        hidden: c.hidden,
        inter: c.inter,
        router: Router { renorm: c.norm_topk, ..Router::softmax(c.top_k) },
        act: Act::SwiGlu { limit: None },
    }
}

/// Expert `e`'s slice of a 3-D expert tensor [experts, rows, cols] (bf16) of `ck`: (the file's tensor, rows, cols).
fn expert_slice(ck: &Checkpoint, name: &str, e: usize) -> Result<(PathBuf, TensorInfo, usize, usize)> {
    let (f, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    if t.dtype != "BF16" || t.shape.len() != 3 || e as u64 >= t.shape[0] {
        return err(format!("{name}: {} {:?}, where bf16 [experts, rows, cols] with expert {e} is wanted", t.dtype, t.shape));
    }
    let (rows, cols) = (t.shape[1] as usize, t.shape[2] as usize);
    let bytes = (rows * cols * 2) as u64;
    Ok((f.path.clone(), TensorInfo { dtype: "BF16".into(), shape: vec![rows as u64, cols as u64], offset: t.offset + e as u64 * bytes, bytes }, rows, cols))
}

/// Expert `e`'s matrix of a 3-D tensor as bf16 words.
pub fn read_expert(ck: &Checkpoint, name: &str, e: usize) -> Result<Vec<u16>> {
    let (path, t, rows, cols) = expert_slice(ck, name, e)?;
    let f = glyd_gpu::safetensors::File { path: path.clone(), tensors: Vec::new(), metadata: Vec::new() };
    let mut b = vec![0u8; t.bytes as usize];
    f.read_into(&t, &mut b).map_err(|x| Error(format!("{name}[{e}]: {x}")))?;
    let _ = (rows, cols);
    Ok(b.as_chunks::<2>().0.iter().map(|x| u16::from_le_bytes(*x)).collect())
}

/// A mixture-of-experts layer with every expert on the GPU in a block layout.
pub struct QxMoe {
    layer: MoeLayer,
}

impl QxMoe {
    /// Layer `l`'s block (`l` = layers: the MTP head's): the router and the shared expert's gate bf16, the shared expert and the experts packed at `level` (Int6, Int5 or Int4).
    pub fn load(l: usize, c: &ConfigQx, ck: &Checkpoint, level: Layout) -> Result<QxMoe> {
        if !matches!(level, Layout::Int6 | Layout::Int5 | Layout::Int4) {
            return err(format!("the experts are held at the kestrel, swift and wren levels: {level:?} is asked"));
        }
        let p = format!("{}.mlp", c.layer_name(l));
        let router = tensor(ck, &format!("{p}.gate.weight"))?;
        let gate = tensor(ck, &format!("{p}.shared_expert_gate.weight"))?;
        let packs = c.layer_packs(l);
        let [s_up, s_down] = &packs[packs.len() - 2..] else { return err("a layer without its shared expert") };
        let shared = Shared {
            inter: c.shared_inter,
            up: upload_packed(read_pack(ck, None, s_up, level)?)?,
            down: upload_packed(read_pack(ck, None, s_down, level)?)?,
            gate: Some(gate),
        };
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let (mut up, mut down): (Vec<glyd_gpu::pack::Tiered>, Vec<glyd_gpu::pack::Tiered>) = (Vec::new(), Vec::new());
        let (gu, dn) = (format!("{p}.experts.gate_up_proj"), format!("{p}.experts.down_proj"));
        ordered(
            2 * c.experts,
            threads,
            2 * threads,
            &|k| {
                let (name, rows, cols) = if k % 2 == 0 { (&gu, 2 * c.inter, c.hidden) } else { (&dn, c.hidden, c.inter) };
                let w = read_expert(ck, name, k / 2)?;
                let packed = match level {
                    Layout::Int6 => glyd_gpu::pack::pack_int6(&w, rows, cols),
                    Layout::Int5 => glyd_gpu::pack::pack_int5(&w, rows, cols),
                    _ => glyd_gpu::pack::pack_int4(&w, rows, cols),
                };
                packed.ok_or_else(|| Error(format!("{name}[{}]: not packable at {level:?}", k / 2)))
            },
            |k, t| {
                (if k % 2 == 0 { &mut up } else { &mut down }).push(t);
                Ok(())
            },
        )?;
        let (ju, jd) = (Joined::of(&up.iter().collect::<Vec<_>>())?, Joined::of(&down.iter().collect::<Vec<_>>())?);
        let w = MoeWeights { router, bias: None, escale: None, up: ExpertStack::upload(&ju)?, down: Down::Block(ExpertStack::upload(&jd)?) };
        Ok(QxMoe { layer: MoeLayer::new(moe_cfg(c), w, Some(shared))? })
    }

    /// The block's layer, for the description executor (`dynamic::ResidentMoe`).
    pub fn into_layer(self) -> MoeLayer {
        self.layer
    }

    /// The router's choices of the last call of `n` rows (ids int64, weights fp32, [n, k] each), from the shared scratch.
    pub fn routed(e: &EngineQx, n: usize, k: usize) -> Result<(Vec<i64>, Vec<f32>)> {
        let sc = e.moe_scratch.get().ok_or_else(|| Error("no expert has run".into()))?;
        e.dev.sync()?;
        let (mut ids, mut wts) = (vec![0i64; n * k], vec![0f32; n * k]);
        sc.ids.read(0, &mut ids)?;
        sc.wts.read(0, &mut wts)?;
        Ok((ids, wts))
    }
}

/// The factory `EngineQx::new` takes for experts on the GPU at `level`.
pub fn gpu_experts(level: Layout) -> impl Fn(usize, &ConfigQx, &Checkpoint, Layout) -> Result<Box<dyn MoeBlockQx>> {
    move |l, c, ck, _| Ok(Box::new(QxMoe::load(l, c, ck, level)?))
}

impl MoeBlockQx for QxMoe {
    unsafe fn forward(&self, e: &EngineQx, h: u64, n: usize, out: u64) -> Result<()> {
        if e.moe_scratch.get().is_none() {
            let _ = e.moe_scratch.set(MoeScratch::new(&e.dev, &e.lib, &[&self.layer], e.chunk())?);
        }
        let sc = e.moe_scratch.get().ok_or_else(|| Error("the experts' scratch is not made".into()))?;
        self.layer.forward(&e.lib, sc, e.dev.stream, h, n, out)
    }

    fn graphable(&self) -> bool {
        true
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The n-gram contexts follow transformers' `_shift_right_ignore_eos`: the token before (the end token at the start), and the one before that unless the token before is the end token.
    #[test]
    fn the_ngram_contexts() {
        let g = Ngram {
            shards: Vec::new(),
            q4: false,
            per_shard: 1,
            heads: 2,
            per_order: 1,
            row: 1,
            mult: [3, 5, 7],
            size: vec![11, 13],
            offset: vec![0, 11],
            eos: 9,
        };
        assert_eq!(g.contexts([9, 9], &[1, 2, 3]), [(1, 9, 9), (2, 1, 9), (3, 2, 1)]);
        assert_eq!(g.contexts([4, 5], &[9, 6, 7]), [(9, 5, 4), (6, 9, 9), (7, 6, 9)]);
        assert_eq!(g.contexts([9, 9], &[]), []);
        // bigram heads hash two tokens, trigram heads three, each in its own range
        assert_eq!(g.ids(1, 2, 3), vec![(3 ^ 10) % 11, ((3 ^ 10 ^ 21) % 13) + 11]);
    }

    /// Every item filled once with its own number, by one thread or many.
    #[test]
    fn rows_fill_by_threads() {
        for n in [1usize, 63, 64, 1000, 5923] {
            let mut out = vec![0u8; n * 3];
            fill_rows(&mut out, 3, |i, o| o.copy_from_slice(&[(i % 251) as u8, (i / 251) as u8, 7]));
            assert!(out.chunks(3).enumerate().all(|(i, o)| o == [(i % 251) as u8, (i / 251) as u8, 7]), "{n}");
        }
    }
}
