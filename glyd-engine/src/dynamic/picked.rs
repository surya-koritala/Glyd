//! The sparse attention whose indexer is mean-pooled (`runtime::kinds::IndexForm::Mean`, Qwen3.8-Flash-Next's QSA) on the executor: `Op::IdxMean`, the indexer's queries,
//! its raw keys and its pools' keys (each pool of 4 tokens the mean of their raw keys, normed and rotated at its first position); `Op::IdxPick` with `IdxKeys::Pools`, every
//! indexer's pick (kernels/mla.cu `idx_score`, `idx_pick`), unweighted: the best pools' tokens and the open pool's; then the attention kind's preparation (`Op::Attn` with
//! `picked`) of the queries, and of the keys and values into its ring, and the attention over the picked tokens with the in-query gate (kernels/qwen4x.cu `qx_attn_sel`).
//! The keys and values, the raw keys and the pools' keys are rings (the pools' `Ring::entries` of 4 tokens), paged as every attention's (R4b): each read and written
//! through the sequence's block table where it has one.

use super::attn::AttnOp;
use super::{Dyn, IdxKeys, IdxPick, ModelDesc, Op, Part, Slot};
use crate::dev::Buf;
use crate::ffi::{err, Error, Result};

/// The picks' room for a chunk's rows: the pools' scores [rows, emax] fp32, each row's picked tokens [rows, sk] and their count, and a head weight of one for each
/// of the indexer's heads (the unweighted scores read the same row for every query).
pub(super) struct PickRun {
    pub scores: Buf,
    pub emax: usize,
    pub sel: Buf,
    pub sk: usize,
    pub cnt: Buf,
    pub ones: Buf,
}

impl PickRun {
    /// The room of the description's unweighted picks for chunks of `ch` rows (None: it has none).
    pub(super) fn new(d: &ModelDesc, ch: usize) -> Result<Option<PickRun>> {
        let picks: Vec<&IdxPick> =
            d.layers.iter().flatten().filter_map(|op| if let Op::IdxPick(x) = op { Some(x) } else { None }).filter(|x| !x.weighted).collect();
        let Some(first) = picks.first() else { return Ok(None) };
        if picks.iter().any(|x| x.ratio != first.ratio || x.top_k != first.top_k || x.heads != first.heads || !x.expand || !x.tail) {
            return err("unweighted indexers of other pools or picks in one model: not an operation of this executor");
        }
        let (emax, sk) = (d.max_pos / first.ratio + 1, first.top_k * first.ratio + first.ratio - 1);
        let ones = Buf::new(first.heads * 2)?;
        ones.write(0, &vec![0x3f80u16; first.heads])?;
        Ok(Some(PickRun { scores: Buf::new(ch * emax * 4)?, emax, sel: Buf::new(ch * sk * 4)?, sk, cnt: Buf::new(ch * 4)?, ones }))
    }
}

impl Dyn {
    /// `Op::IdxMean` on n rows of one sequence at positions `pos`.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn idx_mean(
        &self,
        p: &Part,
        (heads, dim, block): (usize, usize, usize),
        q_norm: usize,
        k_norm: usize,
        rope: usize,
        keys: usize,
        pools: usize,
        n: usize,
        pos: u64,
    ) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("an indexer in a model loaded without its kernels".into()))?;
        let s = self.dev.stream;
        let (cs, sn) = (self.ropes[rope].0.ptr, self.ropes[rope].1.ptr);
        let half = p.d.ropes[rope].pairs;
        let qk = p.slot(Slot::Qr) + self.rows_before((heads + 1) * dim);
        let iq = p.slot(Slot::Iq) + self.rows_before(heads * dim);
        let (ik, pk) = ((self.cache(keys).0, self.table(keys)), (self.cache(pools).0, self.table(pools)));
        let r3 = self.rope3(p);
        k.ix_prep_at(s, qk, p.small[q_norm].ptr, cs, sn, half, pos, r3, iq, ik, n, heads, p.d.eps, dim)?;
        k.ix_cmp_at(s, ik, p.small[k_norm].ptr, cs, sn, half, pos, r3, pk, n, p.d.eps, dim, block)?;
        self.mark("indexer: queries and pools")
    }

    /// `Op::IdxPick` of pooled keys (`IdxKeys::Pools`) of layer `l` on n rows of one sequence at positions `pos`: into the picks' room at the sequence's rows
    /// (and each row's pick kept where a check traces them, `trace_row_picks`).
    pub(super) unsafe fn idx_pick_pools(&self, p: &Part, l: usize, x: &IdxPick, n: usize, pos: u64) -> Result<()> {
        let r = self.picks.as_ref().ok_or_else(|| Error("an indexer's pick in a model loaded without its room".into()))?;
        let IdxKeys::Pools(pools) = x.keys else { return err("not an indexer of pooled keys") };
        let row0 = self.rows_before(1) as usize / 2;
        let at = super::mla::PickAt {
            q: p.slot(Slot::Iq) + self.rows_before(x.heads * x.dim()),
            w: if x.weighted { p.slot(Slot::Iw) + self.rows_before(x.heads) } else { r.ones.ptr },
            keys: self.cache(pools).0,
            stride: x.dim(),
            tab: self.table(pools),
            pos,
            rows: n,
            scores: r.scores.ptr,
            emax: r.emax,
            sel: r.sel.ptr + (row0 * r.sk * 4) as u64,
            sk: r.sk,
            cnt: r.cnt.ptr + (row0 * 4) as u64,
            cand: super::mla::CandAt::default(),
        };
        self.idx_pick(x, &at)?;
        self.trace_row_picks(p, l, x.ratio, n, row0)?;
        self.mark("indexer: picks")
    }

    /// `Op::Attn` with `picked` on n rows of one sequence at positions `pos`: the queries, keys and values prepared (the keys and values into its ring, through the
    /// sequence's block table where it has one), then the attention of each row over its picks, gated, into Att.
    pub(super) unsafe fn attn_picked(&self, p: &Part, a: &AttnOp, n: usize, pos: u64) -> Result<()> {
        let k = self.kqx.as_ref().ok_or_else(|| Error("a sparse attention in a model loaded without its kernels".into()))?;
        let r = self.picks.as_ref().ok_or_else(|| Error("a sparse attention in a model loaded without its picks' room".into()))?;
        let s = self.dev.stream;
        let ((kc, vc), tab) = (self.cache(a.ring), self.table(a.ring));
        let mut x = a.args(n, p.d.eps);
        (x.kc, x.vc, x.cap, x.tab) = (kc, vc, self.ring_cap[a.ring] as i32, tab);
        x.src = p.slot(a.src) + self.rows_before(a.stride);
        x.qo = p.slot(Slot::Qo) + self.rows_before(a.nq * a.d);
        x.qpos = pos;
        if let Some((q, kk)) = a.weights() {
            (x.qw, x.kw) = (p.small[q].ptr, p.small[kk].ptr);
        }
        if let Some(rp) = a.rope {
            (x.cs, x.sn) = (self.ropes[rp.table].0.ptr, self.ropes[rp.table].1.ptr);
            (x.rpos, x.rax) = self.rope3(p);
        }
        self.ka.prep(s, &x, a.nq + a.nk, (crate::kern::NextPf::default(), 0))?;
        let row0 = self.rows_before(1) as usize / 2;
        let src = p.slot(a.src) + self.rows_before(a.stride);
        let out = p.slot(Slot::Att) + self.rows_before(a.nq * a.d);
        let (sel, cnt) = (r.sel.ptr + (row0 * r.sk * 4) as u64, r.cnt.ptr + (row0 * 4) as u64);
        k.attn_sel_at(s, x.qo, a.nq * a.d, src, a.stride, (kc, vc, tab), sel, cnt, r.sk, out, a.nq * a.d, a.nq, a.nk, 0, 1, n)?;
        self.mark("attention (the indexer's picks)")
    }
}
