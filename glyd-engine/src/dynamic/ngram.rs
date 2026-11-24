//! The hashed n-gram embedding kind (`runtime::kinds::Ngram`, Qwen3.8-Flash-Next's per-layer embedding) as the executor's operations: `NgramRows`, each row's rows of the
//! table from its token and the two before it in its sequence (the hash on the GPU, the sequence's last tokens a state), into In; then the product of In by the key
//! and value matrices stacked (an `Op::Linear` into Gu); then `NgramAdd`, the per-stream gate of the value, its norm, the dilated convolution (its inputs a state)
//! and the sum into the streams S. The kernels are kernels/qwen4x.cu's (`qx_ngram_rows`, `qx_ple_gate`, `qx_ple_conv`): four streams, bigrams and trigrams, a
//! convolution of 4 taps 3 apart. The table is held on the GPU as saved (a model whose table does not fit is not loaded here: src/dynamic/qwen4x.rs).

use super::{Dyn, Part, Slot};
use crate::ffi::{Error, Result};

/// The rows of the table into In: the table (a matrix of the part, held as saved: bf16 [rows, row]), the hash's numbers (small int64 tensors: the `order`
/// multipliers, each table's size, each table's offset), the sequence's last `order` - 1 tokens (state `hist`, int32), `per` tables an order (`row` values a
/// table's row), the end token `eos` (a position before the sequence reads as it); `map`: each token first mapped to its compressed vocabulary's id (small int32
/// [vocabulary], Engram's); `segments`: a segment ends at `eos`, the tokens before it read as `eos` (Condor's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramRows {
    pub table: usize,
    pub hash: [usize; 3],
    pub hist: usize,
    pub per: usize,
    pub row: usize,
    pub eos: u32,
    pub order: usize,
    pub map: Option<usize>,
    pub segments: bool,
}

/// The dilated convolution of an n-gram block's gated value: its input's norm (small [4 hidden]), its taps (small, [4 hidden, `taps_n`], `dilation` positions
/// apart) and its last inputs (state `state`, bf16 [(taps_n - 1) dilation, 4 hidden]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramConv {
    pub norm: usize,
    pub taps: usize,
    pub state: usize,
    pub taps_n: usize,
    pub dilation: usize,
}

/// The gated add into S of Gu's keys (4 hidden: a stream's each) and value (hidden) a row: the streams' norm as queries `q` and the keys' `k` (small, [4 hidden]
/// each; (1 + w) where `plus_one`, else w), each stream's gate sigmoid(signed sqrt(max(|<k_s, q_s>| / sqrt(hidden), 1e-6))); with `conv`, the gated value normed
/// and convolved added to it (Condor's), else the gated value alone (Engram's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramAdd {
    pub q: usize,
    pub k: usize,
    pub conv: Option<NgramConv>,
    pub plus_one: bool,
    pub eps: f32,
}

impl Dyn {
    /// `Op::NgramRows` on n rows of one sequence.
    pub(super) unsafe fn ngram_rows(&self, p: &Part, x: &NgramRows, n: usize) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an n-gram embedding in a model loaded without its kernels".into()))?;
        let tok = self.tok.as_ref().ok_or_else(|| Error("an n-gram embedding without the tokens' buffer".into()))?;
        let e = p.slot(Slot::In) + self.rows_before(p.d.width(Slot::In));
        let crate::kern35::Mat::Dense(table, ..) = &p.mats[x.table] else { return Err(Error("the n-gram table is not held as saved".into())) };
        if (x.order, x.map, x.segments) != (3, None, true) {
            return Err(Error(format!(
                "an n-gram block of order {} (a token map: {}; segments: {}): the kernel takes Condor's trigrams alone yet",
                x.order,
                x.map.is_some(),
                x.segments
            )));
        }
        let hash = x.hash.map(|i| p.small[i].ptr);
        k.ngram_rows(self.dev.stream, tok.ptr + self.rows_before(2), self.state(x.hist), hash, table.ptr, e, n, 2 * x.per, x.per, x.row, x.eos)?;
        self.mark("n-gram embedding: rows")
    }

    /// `Op::NgramAdd` on n rows of one sequence.
    pub(super) unsafe fn ngram_add(&self, p: &Part, x: &NgramAdd, n: usize) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an n-gram embedding in a model loaded without its kernels".into()))?;
        let (s, d, eps) = (self.dev.stream, p.d.hidden, p.d.eps);
        let w = 4 * d;
        let (kv, st) = (p.slot(Slot::Gu) + self.rows_before(5 * d), p.slot(Slot::S) + self.rows_before(w));
        let (gv, gvn) = (p.slot(Slot::D) + self.rows_before(w), p.slot(Slot::M) + self.rows_before(w));
        let Some(cv) = x.conv.filter(|c| (c.taps_n, c.dilation, x.plus_one) == (4, 3, true)) else {
            return Err(Error("an n-gram block without Condor's convolution of 4 taps 3 apart: the kernels take it alone yet".into()));
        };
        let (q, kn, c) = (p.small[x.q].ptr, p.small[x.k].ptr, p.small[cv.norm].ptr);
        k.ple_gate(s, kv, st, kn, q, c, gv, gvn, n, d, eps)?;
        k.ple_conv(s, gvn, gv, p.small[cv.taps].ptr, self.state(cv.state), st, n, w)?;
        self.mark("n-gram embedding: gate and convolution")
    }
}
