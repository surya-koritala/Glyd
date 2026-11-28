//! The hashed n-gram embedding kind (`runtime::kinds::Ngram`: Qwen3.8-Flash-Next's per-layer embedding, DeepSeek-V4.1's Engram) as the executor's operations:
//! `NgramRows`, each row's rows of the table from its token and the `order` - 1 before it in its sequence (the hash on the GPU, the sequence's last ids a state;
//! each token its compressed vocabulary's id first where the form has one), into In; then the product of In by the key and value matrices stacked (an
//! `Op::Linear` into Gu); then `NgramAdd`, the per-stream gate of the value and its sum into the streams S: with the segmented form's norm of the gated value and
//! its dilated convolution (its inputs a state) between, or the gated value alone (Engram's). The kernels are kernels/qwen4x.cu's (`qx_ngram_rows`, `qx_ple_gate`,
//! `qx_ple_conv`): four streams, orders 2 to 4, a convolution reaching up to 32 rows back. The table is held on the GPU as saved, or on the host where its kind
//! says (`kinds::TableAt`: the pass's rows gathered by the host, the same ids, and copied in: src/dynamic/tables.rs).

use super::{Dyn, Part, Slot};
use crate::ffi::{Error, Result};

/// The rows of the table into In: the table (a matrix of the part, held as saved: bf16 [rows, row]), the hash's numbers (small int64 tensors: the `order`
/// multipliers, each table's size, each table's offset), the sequence's last `order` - 1 tokens (state `hist`, int32), `per` tables an order (`row` values a
/// table's row), the end token `eos` (a position before the sequence reads as it); `map`: each token first mapped to its compressed vocabulary's id (small int32
/// [vocabulary], Engram's); `segments`: a segment ends at `eos`, the tokens before it read as `eos` (Ibis's).
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
/// and convolved added to it (Ibis's), else the gated value alone (Engram's).
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
        let map = x.map.map_or(0, |m| p.small[m].ptr);
        // (a table on the host: the rows staged with the pass's tokens copied in, src/dynamic/tables.rs; then the sequence's last ids moved on, as the GPU's hash does)
        if std::ptr::eq(p, &self.big) && self.ngram_rows_host(x, e, n)? {
            k.ngram_hist(self.dev.stream, tok.ptr + self.rows_before(2), self.state(x.hist), n, x.order, map)?;
            return self.mark("n-gram embedding: rows");
        }
        let crate::kern35::Mat::Dense(table, ..) = &p.mats[x.table] else { return Err(Error("the n-gram table is not held as saved".into())) };
        let hash = x.hash.map(|i| p.small[i].ptr);
        let h = (x.order - 1) * x.per;
        k.ngram_rows(
            self.dev.stream,
            tok.ptr + self.rows_before(2),
            self.state(x.hist),
            hash,
            table.ptr,
            e,
            n,
            h,
            x.per,
            x.row,
            x.eos,
            (x.order, map, x.segments),
        )?;
        self.mark("n-gram embedding: rows")
    }

    /// `Op::NgramAdd` on n rows of one sequence: the segmented form's (the gate's (1 + w) weights, its rounding, the convolution) or Engram's (plain fp32 weights,
    /// no convolution).
    pub(super) unsafe fn ngram_add(&self, p: &Part, x: &NgramAdd, n: usize) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an n-gram embedding in a model loaded without its kernels".into()))?;
        let (s, d) = (self.dev.stream, p.d.hidden);
        let w = 4 * d;
        let (kv, st) = (p.slot(Slot::Gu) + self.rows_before(5 * d), p.slot(Slot::S) + self.rows_before(w));
        let (gv, gvn) = (p.slot(Slot::D) + self.rows_before(w), p.slot(Slot::M) + self.rows_before(w));
        let (q, kn) = (p.small[x.q].ptr, p.small[x.k].ptr);
        match x.conv {
            Some(cv) if x.plus_one => {
                k.ple_gate(s, kv, st, kn, q, p.small[cv.norm].ptr, gv, gvn, n, d, p.d.eps, false)?;
                k.ple_conv(s, gvn, gv, p.small[cv.taps].ptr, self.state(cv.state), st, n, w, (cv.taps_n, cv.dilation))?;
            }
            None if !x.plus_one => k.ple_gate(s, kv, st, kn, q, 0, gv, gvn, n, d, x.eps, true)?,
            _ => {
                return Err(Error(format!(
                    "an n-gram block of {x:?}: the kernels take the segmented form (the convolution, (1 + w) norms) or Engram's (plain norms, no convolution)"
                )))
            }
        }
        self.mark("n-gram embedding: gate and convolution")
    }

    /// `Op::NgramRows` on the pass's first `rows` rows, each a decode row of its own sequence (`Dyn::lead`): one launch for all of them, each row's last tokens its own
    /// sequence's state (`rows::RowTabs`), moved on past the row. A table on the host: the rows staged with the pass's tokens (the lead's first, a row each) copied in, a
    /// node of the step's graph where it is one, and the launch moves the states on alone (a graph's padding rows, whose staged rows are no sequence's, zeroed by it).
    pub(super) unsafe fn ngram_rows_lead(&self, p: &Part, x: &NgramRows, rows: usize) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an n-gram embedding in a model loaded without its kernels".into()))?;
        let tok = self.tok.as_ref().ok_or_else(|| Error("an n-gram embedding without the tokens' buffer".into()))?;
        let e = p.slot(Slot::In);
        let table = if std::ptr::eq(p, &self.big) && self.ngram_rows_host(x, e, rows)? {
            0
        } else {
            let crate::kern35::Mat::Dense(table, ..) = &p.mats[x.table] else { return Err(Error("the n-gram table is not held as saved".into())) };
            table.ptr
        };
        let hash = x.hash.map(|i| p.small[i].ptr);
        let map = x.map.map_or(0, |m| p.small[m].ptr);
        let h = (x.order - 1) * x.per;
        k.ngram_lead(self.dev.stream, tok.ptr, self.lead_states(x.hist), hash, table, e, rows, h, x.per, x.row, x.eos, (x.order, map, x.segments))?;
        self.mark("n-gram embedding: rows")
    }

    /// `Op::NgramAdd` on the pass's first `rows` rows, each a decode row of its own sequence (`Dyn::lead`): the gate on their rows, then (the segmented form's) the
    /// convolution, one launch for all of them, each row's last inputs its own sequence's state (`rows::RowTabs`).
    pub(super) unsafe fn ngram_add_lead(&self, p: &Part, x: &NgramAdd, rows: usize) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an n-gram embedding in a model loaded without its kernels".into()))?;
        let (s, d) = (self.dev.stream, p.d.hidden);
        let w = 4 * d;
        let (kv, st, gv, gvn) = (p.slot(Slot::Gu), p.slot(Slot::S), p.slot(Slot::D), p.slot(Slot::M));
        let (q, kn) = (p.small[x.q].ptr, p.small[x.k].ptr);
        match x.conv {
            Some(cv) if x.plus_one => {
                k.ple_gate(s, kv, st, kn, q, p.small[cv.norm].ptr, gv, gvn, rows, d, p.d.eps, false)?;
                k.ple_conv_rows(s, gvn, gv, p.small[cv.taps].ptr, self.lead_states(cv.state), st, rows, w, (cv.taps_n, cv.dilation))?;
            }
            None if !x.plus_one => k.ple_gate(s, kv, st, kn, q, 0, gv, gvn, rows, d, x.eps, true)?,
            _ => {
                return Err(Error(format!(
                    "an n-gram block of {x:?}: the kernels take the segmented form (the convolution, (1 + w) norms) or Engram's (plain norms, no convolution)"
                )))
            }
        }
        self.mark("n-gram embedding: gate and convolution")
    }
}
