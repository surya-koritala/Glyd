//! The hashed n-gram embedding kind (`runtime::kinds::Ngram`, Qwen3.8-Flash-Next's per-layer embedding) as the executor's operations: `NgramRows`, each row's rows of the
//! table from its token and the two before it in its sequence (the hash on the GPU, the sequence's last tokens a state), into In; then the product of In by the key
//! and value matrices stacked (an `Op::Linear` into Gu); then `NgramAdd`, the per-stream gate of the value, its norm, the dilated convolution (its inputs a state)
//! and the sum into the streams S. The kernels are kernels/qwen4x.cu's (`qx_ngram_rows`, `qx_ple_gate`, `qx_ple_conv`): four streams, bigrams and trigrams, a
//! convolution of 4 taps 3 apart. The table is held on the GPU as saved (a model whose table does not fit is not loaded here: src/dynamic/qwen4x.rs).

use super::{Dyn, Part, Slot};
use crate::ffi::{Error, Result};

/// The rows of the table into In: the table (a matrix of the part, held as saved: bf16 [rows, row]), the hash's numbers (small tensors, int64 as the checkpoint
/// holds them: the 3 multipliers, each table's size, each table's offset), the sequence's last two tokens (state `hist`, 8 bytes), `per` tables an order (bigrams,
/// trigrams: 2 `per` tables of `row` values), the end token `eos`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramRows {
    pub table: usize,
    pub hash: [usize; 3],
    pub hist: usize,
    pub per: usize,
    pub row: usize,
    pub eos: u32,
}

/// The gated add into S of Gu's key (4 hidden) and value (hidden) a row: the norms' weights (small, [4 hidden] each: the streams' as queries, the keys', the
/// convolution input's), the convolution's taps (small, [4 hidden, 4]) and its last 9 inputs (state `conv`, bf16 [9, 4 hidden]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramAdd {
    pub norms: [usize; 3],
    pub taps: usize,
    pub conv: usize,
}

impl Dyn {
    /// `Op::NgramRows` on n rows of one sequence.
    pub(super) unsafe fn ngram_rows(&self, p: &Part, x: &NgramRows, n: usize) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an n-gram embedding in a model loaded without its kernels".into()))?;
        let tok = self.tok.as_ref().ok_or_else(|| Error("an n-gram embedding without the tokens' buffer".into()))?;
        let e = p.slot(Slot::In) + self.rows_before(p.d.width(Slot::In));
        let crate::kern35::Mat::Dense(table, ..) = &p.mats[x.table] else { return Err(Error("the n-gram table is not held as saved".into())) };
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
        let [q, kn, c] = x.norms.map(|i| p.small[i].ptr);
        k.ple_gate(s, kv, st, kn, q, c, gv, gvn, n, d, eps)?;
        k.ple_conv(s, gvn, gv, p.small[x.taps].ptr, self.state(x.conv), st, n, w)?;
        self.mark("n-gram embedding: gate and convolution")
    }
}
