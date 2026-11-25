//! The attention layer kind (`runtime::kinds::Attn`) as one operation of the executor (`Op::Attn`), for every dense-attention layer of every family (#135, #141).
//! What a family's layer is, is the operation's parameters (`AttnOp`): its heads and their width, where its q, k, v and gate are in the projection's rows, its norms,
//! rotary, window, scale, soft-cap and gate, and its ring of keys and values; its rounding points the kind's (`kinds::AttnRounding`). One set of kernels runs them all
//! (kernels/attn.cu): the preparation, then a decode row's attention (the keys in parts, combined) or a prompt chunk's, and the gate. A prompt chunk of heads of
//! 64 and 128 takes the kind's own FlashAttention-2 kernel (attn_pf64, attn_pf128: a token's query heads in one block, tiles of 64 keys), over 12-bit pages
//! (src/dynamic/kvpages.rs) on the pages unpacked to a bf16 ring, so the same bits as over a bf16 cache; heads of 256 and 512 the GPU library's prompt kernel,
//! one for both caches: over bf16 rings or pages `glyd_gpu_kv_prefill_bf16`, over 12-bit pages `glyd_gpu_kv_prefill`, the same tiles and sums, so the same
//! bits. A decode row likewise (`LibDecode`): over 12-bit pages `kv_decode_from`, over a bf16 cache `glyd_gpu_kv_decode_bf16`, one kernel, plan and merge, where
//! the shape is the pages' (heads of 128 to 512, up to 8 query heads a kv head or one kv head, no soft-cap); others (Muse Spark's heads of 64, Nemotron-H's 16
//! query heads over 2) the engine's parts kernel.
//!
//! The arithmetic of each family is the one its own operations had, bit for bit where the kernel is the same (the preparation, the decode and the gate of Gemma 4,
//! the Qwen3.5 layout and Nemotron-H); a prompt chunk's is now the library's kernel for all (Gemma 4's g4_fa256, the Qwen3.5 layout's warp a head, Nemotron-H's
//! dense-engine kernel and Muse Spark's key at a time before): the same function, sums in another order. Muse Spark's decode is the parts kernel too.

use super::{Dyn, Part, Slot};
use crate::dev::Dev;
use crate::ffi::{err, Result, Stream};
use crate::gemma4::G4Pf;
use crate::gpu_ops::GpuOps;
use crate::launch;
use crate::runtime::kinds;

/// Keys a tile of the warp kernels (kernels/attn.cu AT_KB).
const KB: usize = 32;
/// The parts a decode row's keys are split into.
pub const SPLITS: usize = 16;
/// The warps of a decode row's attention block: the G that score its row and more that only load the key and value tiles, so a short context is not one warp
/// pair's loads in a row (GLY-312: 0.75 ms of a gemma-4-12B step at 30 tokens with 2 warps).
const PART_WARPS: usize = 16;
/// Query heads to a key-value head the warp kernels take (a block of at most 512 threads).
pub const MAX_G: usize = 16;
/// Query heads to a key-value head a call of the library's decode takes.
pub const MAX_DG: usize = 8;

/// The GPU library's decode for the attention layers whose shapes it takes (`AttnOp::lib_decode`), over a bf16 cache (glyd_gpu_kv_decode_bf16) as over 12-bit pages
/// (kv_decode_from): one kernel, plan and merge, so the two caches give a decode row the same bits. Its splits (the 12-bit pages' too), its workspace, and each kind of
/// call's (window, calls a row) rows' lengths and first tokens, made once a pass (`prepare`).
pub(super) struct LibDecode {
    pub splits: i64,
    ws: crate::dev::Buf,
    kinds: Vec<(usize, usize)>,
    lens: Vec<crate::dev::Buf>,
    first: Vec<crate::dev::Buf>,
    /// The widest call's kv heads, query heads a call's sequence and head width (the workspace's shape).
    shape: (usize, usize, usize),
}

impl LibDecode {
    /// For the description's attention layers (`ops`) on `rows` rows a pass at most, `drows` a decode call, a context of `ctx`; None where none takes it. The plan's
    /// floor is set (the 12-bit pages' DECODE_FLOOR) where GLYD_KV_FLOOR does not set it.
    pub fn new<'a>(lib: &crate::ffi::Lib, ops: impl Iterator<Item = &'a AttnOp>, rows: usize, drows: usize, ctx: usize) -> Result<Option<LibDecode>> {
        let (mut heads, mut g, mut d, mut parts, mut kinds) = (0, 0, 0, 0, Vec::new());
        for a in ops.filter(|a| a.lib_decode()) {
            let p = a.dparts();
            (heads, g, d, parts) = (heads.max(a.nk), g.max(a.nq / a.nk / p), d.max(a.d), parts.max(p));
            if !kinds.contains(&(a.window, p)) {
                kinds.push((a.window, p));
            }
        }
        if kinds.is_empty() {
            return Ok(None);
        }
        if std::env::var_os("GLYD_KV_FLOOR").is_none() {
            lib.kv_floor_set(super::kvpages::DECODE_FLOOR)?;
        }
        let splits = lib.kv_decode_splits(1, heads as i64, ctx.div_ceil(16) as i64)?;
        let ws = crate::dev::Buf::zeroed(lib.kv_decode_workspace((drows * parts) as i64, heads as i64, g as i64, d as i64, splits)?)?;
        let lens = kinds.iter().map(|k| crate::dev::Buf::new(4 * rows * k.1)).collect::<Result<_>>()?;
        let first = kinds.iter().map(|k| crate::dev::Buf::new(4 * rows * k.1)).collect::<Result<_>>()?;
        Ok(Some(LibDecode { splits, ws, kinds, lens, first, shape: (heads, g, d) }))
    }

    /// The workspace made for a call of `drows` rows at least (a step's rows that decode, all in one call: `Dyn::attn_lead`), each row's query heads in the most calls' sequences
    /// a kind has. The splits stay: a sequence's bits do not depend on the batch.
    pub fn grow(&mut self, lib: &crate::ffi::Lib, drows: usize) -> Result<()> {
        let parts = self.kinds.iter().map(|k| k.1).max().unwrap_or(1);
        let (h, g, d) = self.shape;
        let need = lib.kv_decode_workspace((drows * parts) as i64, h as i64, g as i64, d as i64, self.splits)?;
        if need > self.ws.bytes {
            self.ws = crate::dev::Buf::zeroed(need)?;
        }
        Ok(())
    }
}

/// Dynamic shared memory of a warp kernel's block of head `d` and `rg` warps that score.
pub(crate) fn rows_smem(d: usize, rg: usize) -> u32 {
    ((KB * (d / 2 + 1) + KB * d / 2 + rg * d / 2) * 4) as u32
}

/// A prompt block's rows of the kind's own prompt kernel (kernels/attn.cu PF_ROWS: a key-value head's query heads, token by token) and its keys a tile (PF_KT).
const PF_ROWS: usize = 64;
const PF_KT: usize = 64;

/// Dynamic shared memory of a block of the kind's own prompt kernel (attn_pf_t): K and V [PF_KT][d + 8] bf16.
fn pf_smem(d: usize) -> u32 {
    (2 * PF_KT * (d + 8) * 2) as u32
}

/// The q and k norms of a layer (their weights' small tensors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QkNorm {
    None,
    /// rmsnorm(x) w
    Weight {
        q: usize,
        k: usize,
    },
    /// rmsnorm(x) (1 + w)
    Plus1 {
        q: usize,
        k: usize,
    },
    /// rmsnorm(x), no weight
    Plain,
}

/// The rotary of a layer: rotary table `table` (bf16 cos and sin [positions, pairs]), the pairs (i, i + span / 2) of a head's first `span` dims, the first
/// `pairs` of them turned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeAt {
    pub table: usize,
    pub span: usize,
    pub pairs: usize,
}

/// An output gate o = norm(o) sigmoid(g): head h's gate values at column `off` + h `step` of the projection's rows; the head's weightless norm first where `norm`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateAt {
    pub off: usize,
    pub step: usize,
    pub norm: bool,
}

/// `Op::Attn`: the preparation of the projection's rows (slot `src`, rows `stride` wide: query head h at h `q_step`, the key heads from `k_off`, the value heads
/// from `v_off` or none: K = V, the keys' projection normed without a weight), then the causal attention of the queries over ring `ring` (`window` 0: every
/// position) into Att, gated where `gate`. `writes`: the layer writes its keys and values (a draft model's layers read the big model's ring and write none).
#[derive(Clone, Debug, PartialEq)]
pub struct AttnOp {
    pub src: Slot,
    pub stride: usize,
    pub q_step: usize,
    pub k_off: usize,
    pub v_off: Option<usize>,
    pub d: usize,
    pub nq: usize,
    pub nk: usize,
    pub writes: bool,
    pub ring: usize,
    pub window: usize,
    pub norm: QkNorm,
    pub v_norm: bool,
    pub rope: Option<RopeAt>,
    pub q_scale: f32,
    pub scale: f32,
    pub softcap: f32,
    pub gate: Option<GateAt>,
    pub round: bool,
    pub sbf16: bool,
    /// The attention over the tokens the indexer's pick before it took (`Op::IdxPick`, its picks' run: src/dynamic/picked.rs), its keys and values in its ring as
    /// any attention's: a sparse attention (`Attn::indexer`). Heads of 256, the gate in the query.
    pub picked: bool,
}

impl AttnOp {
    /// The operation of layer kind `a` whose projection (queries, an in-query gate interleaved with them, keys, values, an own gate: in that order) is in `src`, its
    /// keys and values ring `ring`, its rotary table `table`, its q and k norms' weights the small tensors `qk`. Its arithmetic is the kind's rounding points
    /// (`kinds::AttnRounding`): Eager's and Sdpa's rope, norms, queries' scale and gate in bf16 arithmetic (`round`), Eager's scores the bf16 of their dot
    /// products and of their scale (`sbf16`), Sdpa's scale on the queries before the rope (the scores then unscaled); Flash's one rounding at the end of each.
    /// F32 (transformers in float32) runs as Flash: the executor's activations are bf16.
    pub fn of(a: &kinds::Attn, src: Slot, ring: usize, table: Option<usize>, qk: Option<(usize, usize)>) -> Result<AttnOp> {
        let (d, nq, nk) = (a.head_dim, a.heads, a.kv_heads);
        if ![64, 128, 256, 512].contains(&d) || a.v_head_dim != d {
            return err(format!("attention heads of {d} (values {}): the kernels take 64, 128, 256 or 512, the values as wide", a.v_head_dim));
        }
        if nk == 0 || nq % nk != 0 || nq / nk > MAX_G {
            return err(format!("{nq} query heads over {nk} key-value heads: the kernels take up to {MAX_G} a key-value head"));
        }
        if a.sinks || a.bias || a.indexer.is_some() {
            return err("attention with sinks, a bias or an indexer: not an operation of this executor yet");
        }
        let in_query = a.gate.is_some_and(|g| g.from == kinds::GateFrom::InQuery);
        let q_step = if in_query { 2 * d } else { d };
        let k_off = nq * q_step;
        let v_off = (!a.k_eq_v).then_some(k_off + nk * d);
        let after = k_off + nk * d * if a.k_eq_v { 1 } else { 2 };
        let gate = match a.gate {
            None => None,
            Some(g) if g.act != kinds::Act::Sigmoid => return err(format!("an output gate of {:?}: the kernels take sigmoid", g.act)),
            Some(g) if g.from == kinds::GateFrom::InQuery => Some(GateAt { off: d, step: 2 * d, norm: g.norm.is_some() }),
            Some(g) => Some(GateAt { off: after, step: d, norm: g.norm.is_some() }),
        };
        let stride = after + if matches!(a.gate, Some(g) if g.from == kinds::GateFrom::Own) { nq * d } else { 0 };
        let rms = |n: &Option<kinds::Norm>| -> Result<()> {
            match n {
                Some(n) if n.kind != kinds::NormKind::Rms => err(format!("a q, k or v norm of {:?}: the kernels take RMS norms", n.kind)),
                _ => Ok(()),
            }
        };
        rms(&a.q_norm)?;
        rms(&a.k_norm)?;
        rms(&a.v_norm)?;
        let norm = match (a.q_norm.map(|n| n.weight), qk) {
            (None, _) => QkNorm::None,
            (Some(kinds::NormWeight::None), _) => QkNorm::Plain,
            (Some(kinds::NormWeight::Plain), Some((q, k))) => QkNorm::Weight { q, k },
            (Some(kinds::NormWeight::PlusOne), Some((q, k))) => QkNorm::Plus1 { q, k },
            (w, _) => return err(format!("q and k norms of {w:?} (weights {qk:?}): not an operation of this executor")),
        };
        if a.k_norm.map(|n| n.weight) != a.q_norm.map(|n| n.weight) || a.v_norm.is_some_and(|n| n.weight != kinds::NormWeight::None) {
            return err("q and k norms of different kinds, or a v norm with a weight: not an operation of this executor");
        }
        let rope = match (&a.rope, table) {
            (None, _) => None,
            (Some(r), Some(table)) if r.style == kinds::RopeStyle::Half && r.dim <= d && r.freqs <= r.dim / 2 => {
                Some(RopeAt { table, span: r.dim, pairs: r.freqs })
            }
            (Some(r), t) => return err(format!("a rotary of {r:?} (table {t:?}): the kernels turn pairs (i, i + dim / 2) of a table's")),
        };
        use kinds::AttnRounding::*;
        let (round, sbf16, prescale) = match a.rounding {
            F32 | Flash => (false, false, false),
            Eager => (true, true, false),
            Sdpa => (true, false, true),
        };
        let (q_scale, scale) = if prescale { (a.scale, 1.0) } else { (1.0, a.scale) };
        Ok(AttnOp {
            src,
            stride,
            q_step,
            k_off,
            v_off,
            d,
            nq,
            nk,
            writes: true,
            ring,
            window: a.window.unwrap_or(0),
            norm,
            v_norm: a.v_norm.is_some(),
            rope,
            q_scale,
            scale,
            softcap: a.softcap.unwrap_or(0.0),
            gate,
            round,
            sbf16,
            picked: false,
        })
    }

    /// The operation of a sparse attention (`a` with an indexer: Condor's, `IndexForm::Mean`) whose keys and values are ring `ring`, over the indexer's picks
    /// (`picked`); the rest as `of`. The kernel of the picks takes heads of 256 and an in-query gate.
    pub fn picked(a: &kinds::Attn, src: Slot, ring: usize, table: Option<usize>, qk: Option<(usize, usize)>) -> Result<AttnOp> {
        if a.head_dim != 256 || !a.gate.is_some_and(|g| g.from == kinds::GateFrom::InQuery && g.norm.is_none()) || a.window.is_some() || a.softcap.is_some() {
            return err(format!(
                "a sparse attention of heads of {} (a window {:?}, a soft-cap {:?}): the kernel of the picks takes heads of 256 and an in-query gate",
                a.head_dim, a.window, a.softcap
            ));
        }
        let mut x = AttnOp::of(&kinds::Attn { indexer: None, ..a.clone() }, src, ring, table, qk)?;
        x.picked = true;
        Ok(x)
    }

    /// The kernels' arguments of this operation on n rows, the pointers null (`Dyn::attn` sets them).
    pub(crate) fn args(&self, n: usize, eps: f32) -> AttnArgs {
        let (q, k, norm) = match self.norm {
            QkNorm::None => (0, 0, 0),
            QkNorm::Weight { q, k } => (q, k, 1),
            QkNorm::Plus1 { q, k } => (q, k, 2),
            QkNorm::Plain => (0, 0, 3),
        };
        let _ = (q, k);
        let g = self.gate.unwrap_or(GateAt { off: 0, step: 0, norm: false });
        AttnArgs {
            d: self.d as i32,
            stride: self.stride as i32,
            q_step: self.q_step as i32,
            k_off: self.k_off as i32,
            v_off: self.v_off.map_or(-1, |v| v as i32),
            norm,
            v_norm: self.v_norm as i32,
            span: self.rope.map_or(0, |r| r.span as i32),
            pairs: self.rope.map_or(0, |r| r.pairs as i32),
            n: n as i32,
            nq: self.nq as i32,
            nk: self.nk as i32,
            window: self.window as i32,
            os: (self.nq * self.d) as i32,
            gs: self.stride as i32,
            g_step: g.step as i32,
            gnorm: g.norm as i32,
            round: self.round as i32,
            sbf16: self.sbf16 as i32,
            S: SPLITS as i32,
            q_scale: self.q_scale,
            eps,
            scale: self.scale,
            sl2: sl2(self.scale),
            softcap: self.softcap,
            ..AttnArgs::default()
        }
    }

    /// Whether a decode row of this attention takes the GPU library's decode (`LibDecode`): heads of 128, 256 or 512, up to 8 query heads a kv head or one kv head
    /// (its query heads in calls of 8), no soft-cap; the shapes the 12-bit pages take.
    pub(crate) fn lib_decode(&self) -> bool {
        let g = self.nq / self.nk;
        self.d % 128 == 0 && self.d <= 512 && self.d != 384 && (g <= MAX_DG || (self.nk == 1 && g % MAX_DG == 0)) && self.softcap == 0.0
    }

    /// The decode's calls a row's query heads take (8 at most each).
    pub(crate) fn dparts(&self) -> usize {
        (self.nq / self.nk).div_ceil(MAX_DG)
    }

    /// The q and k norms' small tensors, where they have weights.
    pub(super) fn weights(&self) -> Option<(usize, usize)> {
        match self.norm {
            QkNorm::Weight { q, k } | QkNorm::Plus1 { q, k } => Some((q, k)),
            _ => None,
        }
    }
}

/// The scores' factor in log2 units, as the kernels take it.
pub fn sl2(scale: f32) -> f32 {
    (scale as f64 * std::f64::consts::LOG2_E) as f32
}

/// kernels/attn.cu's AttnArgs, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
#[allow(non_snake_case)]
pub(crate) struct AttnArgs {
    pub src: u64,
    pub qw: u64,
    pub kw: u64,
    pub cs: u64,
    pub sn: u64,
    pub qpos: u64,
    pub kpos: u64,
    pub qo: u64,
    pub kc: u64,
    pub vc: u64,
    pub tab: u64,
    pub out: u64,
    pub gate: u64,
    pub part: u64,
    /// Row t's table at tab + t tstride entries (a step's rows that decode, each its sequence's: `rows::RowTabs`); 0: one table for every row.
    pub tstride: i64,
    pub d: i32,
    pub stride: i32,
    pub q_step: i32,
    pub k_off: i32,
    pub v_off: i32,
    pub norm: i32,
    pub v_norm: i32,
    pub span: i32,
    pub pairs: i32,
    pub cap: i32,
    pub n: i32,
    pub nq: i32,
    pub nk: i32,
    pub window: i32,
    pub os: i32,
    pub gs: i32,
    pub g_step: i32,
    pub gnorm: i32,
    pub round: i32,
    pub sbf16: i32,
    pub S: i32,
    pub q_scale: f32,
    pub eps: f32,
    pub scale: f32,
    pub sl2: f32,
    pub softcap: f32,
}

/// kernels/attn.cu's kernels, each by its type (`crate::kt`).
pub(crate) struct KernAttn {
    prep64: crate::kt::attn_prep64,
    prep128: crate::kt::attn_prep128,
    prep256: crate::kt::attn_prep256,
    prep512: crate::kt::attn_prep512,
    rows64: crate::kt::attn_rows64,
    rows128: crate::kt::attn_rows128,
    rows256: crate::kt::attn_rows256,
    rows512: crate::kt::attn_rows512,
    parts64: crate::kt::attn_parts64,
    parts128: crate::kt::attn_parts128,
    parts256: crate::kt::attn_parts256,
    parts512: crate::kt::attn_parts512,
    merge: crate::kt::attn_merge,
    gate: crate::kt::attn_gate,
    pf64: crate::kt::attn_pf64,
    pf128: crate::kt::attn_pf128,
    /// Whether prompt chunks take the tensor-core kernels (the kind's own for heads of 64 and 128, the library's for 256 and 512; a development build's GLYD_ATTN_FA=0:
    /// the warp kernels).
    pub fa: bool,
}

impl KernAttn {
    pub fn load(dev: &Dev) -> Result<KernAttn> {
        let k = KernAttn {
            prep64: dev.typed()?,
            prep128: dev.typed()?,
            prep256: dev.typed()?,
            prep512: dev.typed()?,
            rows64: dev.typed()?,
            rows128: dev.typed()?,
            rows256: dev.typed()?,
            rows512: dev.typed()?,
            parts64: dev.typed()?,
            parts128: dev.typed()?,
            parts256: dev.typed()?,
            parts512: dev.typed()?,
            merge: dev.typed()?,
            gate: dev.typed()?,
            pf64: dev.typed()?,
            pf128: dev.typed()?,
            // (on: against transformers bf16 on gemma-4-31B QAT, wikitext 32 x 512, KLD 0.0192 where the warp kernels' is 0.0225, and a 28K prompt 2x faster)
            fa: {
                // (the switch back, a development build's alone: a release build reads no GLYD_ variable it does not list)
                #[cfg(feature = "dev")]
                let on = std::env::var("GLYD_ATTN_FA").map_or(true, |v| v != "0");
                #[cfg(not(feature = "dev"))]
                let on = true;
                on
            },
        };
        for (kern, d) in [(&*k.rows256, 256), (&*k.parts256, 256), (&*k.rows512, 512), (&*k.parts512, 512), (&*k.rows128, 128), (&*k.parts128, 128)] {
            kern.allow_smem(rows_smem(d, MAX_G))?;
        }
        Ok(k)
    }

    /// The preparation of `a.n` rows (`heads`: nq, and nk more where the keys and values are written), with `pf`'s prefetch on `pfb` more blocks a row and head.
    ///
    /// # Safety
    /// `a`'s buffers live and of the sizes its fields give.
    pub(crate) unsafe fn prep(&self, s: Stream, a: &AttnArgs, heads: usize, pf: (G4Pf, u32)) -> Result<()> {
        let (n, d) = (a.n as u32, a.d as usize);
        let grid = [n, heads as u32, 1 + pf.1.div_ceil(n * heads as u32)];
        let block = [(d / 2) as u32, 1, 1];
        match d {
            64 => launch!(self.prep64, grid, block, s; *a, pf.0),
            128 => launch!(self.prep128, grid, block, s; *a, pf.0),
            256 => launch!(self.prep256, grid, block, s; *a, pf.0),
            512 => launch!(self.prep512, grid, block, s; *a, pf.0),
            _ => err(format!("attention heads of {d}")),
        }
    }

    /// A decode row's attention for each of `a.n` rows (each at its own position): the keys in SPLITS parts into `a.part`, then combined (and gated) into `a.out`.
    ///
    /// # Safety
    /// As `prep`; `a.part` holds n nq SPLITS (d + 2) floats.
    pub(crate) unsafe fn decode(&self, s: Stream, a: &AttnArgs) -> Result<()> {
        let (n, d, nq, nk) = (a.n as u32, a.d as usize, a.nq as u32, a.nk as u32);
        let g = (nq / nk) as usize;
        let (grid, block, smem) = ([n, nk, SPLITS as u32], [(32 * g.max(PART_WARPS)) as u32, 1, 1], rows_smem(d, g));
        let parts = match d {
            64 => self.parts64.sig(),
            128 => self.parts128.sig(),
            256 => self.parts256.sig(),
            512 => self.parts512.sig(),
            _ => return err(format!("attention heads of {d}")),
        };
        launch!(parts, grid, block, smem smem, s; *a)?;
        launch!(self.merge, [n, nq, 1], [d as u32, 1, 1], s; *a)
    }

    /// Whether a prompt chunk of heads of `d` takes the kind's own prompt kernel (attn_pf_t), else the GPU library's.
    pub(crate) fn own_prompt(d: usize) -> bool {
        d <= 128
    }

    /// A prompt chunk's attention of `a.n` rows into `a.out` over a bf16 cache, then the gate: the kind's own kernel for heads of 64 and 128 (a block of PF_ROWS
    /// rows of a key-value head's query heads), the GPU library's for heads of 256 and 512 (`lib`'s glyd_gpu_kv_prefill_bf16, the one its 12-bit pages' prompt
    /// attention is: `start` the rows' first position, as it takes them), the warp kernels where the tensor-core ones are off.
    ///
    /// # Safety
    /// As `prep`; the rows are at positions `start` .. `start + a.n` (`a.kpos` holds them).
    pub(crate) unsafe fn prompt(&self, lib: &crate::ffi::Lib, s: Stream, a: &AttnArgs, start: usize) -> Result<()> {
        let (n, d, g, nk) = (a.n as usize, a.d as usize, (a.nq / a.nk) as usize, a.nk as i64);
        if !self.fa {
            return self.rows(s, a);
        }
        if KernAttn::own_prompt(d) {
            let grid = [(n * g).div_ceil(PF_ROWS) as u32, nk as u32, 1];
            let pf = if d == 64 { self.pf64.sig() } else { self.pf128.sig() };
            launch!(pf, grid, [(32 * PF_ROWS / 16) as u32, 1, 1], smem pf_smem(d), s; *a)?;
        } else {
            let w = (a.nq * a.d) as i64;
            lib.kv_prefill_bf16(
                a.qo,
                w,
                (a.kc, a.vc, a.tab, a.cap as i64),
                (n as i64, start as i64, a.window as i64),
                (nk, (a.nq / a.nk) as i64, d as i64),
                (a.scale as f64, a.sbf16 != 0, a.softcap as f64),
                a.out,
                a.os as i64,
                s,
            )?;
        }
        self.gate(s, a)
    }

    /// The warp kernels' attention of `a.n` rows (R rows a block), then the gate.
    ///
    /// # Safety
    /// As `prep`.
    pub(crate) unsafe fn rows(&self, s: Stream, a: &AttnArgs) -> Result<()> {
        let (n, d, nk) = (a.n as usize, a.d as usize, a.nk as u32);
        let g = (a.nq / a.nk) as usize;
        let r = (16 / g).max(1);
        let (grid, block, smem) = ([n.div_ceil(r) as u32, nk, 1], [(32 * r * g) as u32, 1, 1], rows_smem(d, r * g));
        let rows = match d {
            64 => self.rows64.sig(),
            128 => self.rows128.sig(),
            256 => self.rows256.sig(),
            512 => self.rows512.sig(),
            _ => return err(format!("attention heads of {d}")),
        };
        launch!(rows, grid, block, smem smem, s; *a, r as i32)?;
        self.gate(s, a)
    }

    /// The gate of `a.n` rows of `a.out` in place, where `a` has one.
    ///
    /// # Safety
    /// As `prep`.
    pub(crate) unsafe fn gate(&self, s: Stream, a: &AttnArgs) -> Result<()> {
        if a.gate == 0 {
            return Ok(());
        }
        launch!(self.gate, [a.n as u32, a.nq as u32, 1], [a.d as u32, 1, 1], s; *a)
    }
}

impl Dyn {
    /// A pass's rows' lengths and first tokens for each kind of the library's decode calls, once for every layer (`n` rows at `pos`): one launch a kind for all the
    /// rows, whatever sequences they are (a row's entries depend on its position alone, and a part's rows are at its row0 in them as in `pos`).
    pub(super) unsafe fn lib_decode_prepare(&self, n: usize, pos: (u64, u64)) -> Result<()> {
        let Some(ld) = &self.ldec else { return Ok(()) };
        for (i, &(window, sp)) in ld.kinds.iter().enumerate() {
            self.kern.kv12_rows(self.dev.stream, pos.1, n, (0, 0), window, sp, 0, ld.lens[i].ptr, ld.first[i].ptr)?;
        }
        Ok(())
    }

    /// A decode call of the library over the bf16 cache `x` names (its table, or its ring of `x.cap` places): the n rows of the part an operation runs on now, each
    /// row's query heads in `a.dparts()` calls' sequences.
    unsafe fn lib_decode(&self, a: &AttnOp, n: usize, x: &AttnArgs) -> Result<()> {
        self.lib_decode_at(a, n, x, (x.tab, 0), self.row0())
    }

    /// `lib_decode` of the n rows from row `row0` of the pass, the call's sequence b (row b / sp's query heads' part b % sp) through table `tab.0` + b `tab.1` (0: every
    /// sequence the one table).
    unsafe fn lib_decode_at(&self, a: &AttnOp, n: usize, x: &AttnArgs, tab: (u64, usize), row0: usize) -> Result<()> {
        let ld = self.ldec.as_ref().expect("a library decode");
        let sp = a.dparts();
        let k = ld.kinds.iter().position(|&k| k == (a.window, sp)).expect("a kind of the description's");
        let at = 4 * (row0 * sp) as u64;
        self.dcalls.set(self.dcalls.get() + 1);
        let stride = (a.nq * a.d / sp) as i64;
        self.lib.kv_decode_bf16(
            x.qo,
            stride,
            (x.kc, x.vc, tab.0, tab.1 as i64, x.cap as i64),
            (ld.lens[k].ptr + at, ld.first[k].ptr + at, (n * sp) as i64, ld.splits),
            (a.nk as i64, (a.nq / a.nk / sp) as i64, a.d as i64),
            (a.scale as f64, a.sbf16),
            x.out,
            stride,
            (ld.ws.ptr, ld.ws.bytes),
            self.dev.stream,
        )
    }

    /// `Op::Attn` (op `i` of layer `l` of part `p`) on n rows at positions `pos`: the preparation (unless the pass made it for every row already: `prepped`), then
    /// the attention.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn attn(&self, p: &Part, l: usize, i: usize, a: &AttnOp, n: usize, pos: (u64, u64), head: usize) -> Result<()> {
        if !self.prepped.get() {
            self.attn_prep(p, l, i, a, n, pos, head)?;
        }
        self.attn_read(p, a, n, pos)
    }

    /// `Op::Attn` (op `i` of layer `l` of part `p`) on the pass's first `rows.0` rows (of `rows.1`), each a decode row of its own sequence (`Dyn::lead`): their preparation in one
    /// launch (unless the pass made it for every row already: `prepped`), their attention in one launch of the parts kernel or one call of the library's decode, each row through its
    /// own block table (`rows::RowTabs`, made at the pass's start). Each row's kernels and sums are its own pass's: the same bits (the decode's plan is a sequence's own, and
    /// the 12-bit pages' decode is the bf16 cache's).
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn attn_lead(&self, p: &Part, l: usize, i: usize, a: &AttnOp, rows: (usize, usize), pos: (u64, u64), head: usize) -> Result<()> {
        let (s, k) = (self.dev.stream, rows.0);
        let (tab, t, dup) = self.row_tab(a.ring);
        let (kc, vc) = self.cache(a.ring);
        if !self.prepped.get() {
            let mut x = a.args(k, p.d.eps);
            (x.kc, x.vc, x.cap, x.tab, x.tstride) = (kc, vc, self.ring_cap[a.ring] as i32, tab, (dup * t) as i64);
            (x.src, x.qo, x.qpos) = (p.slot(a.src), p.slot(Slot::Qo), pos.0);
            if let Some((q, kw)) = a.weights() {
                (x.qw, x.kw) = (p.small[q].ptr, p.small[kw].ptr);
            }
            if let Some(r) = a.rope {
                (x.cs, x.sn) = (self.ropes[r.table].0.ptr, self.ropes[r.table].1.ptr);
            }
            let (nx, m) = Self::next_linear(p, l, i, rows.1, head);
            let pf = self.pf(nx, m, self.pf_kib[0], (a.d / 2) as u32);
            self.ka.prep(s, &x, a.nq + if a.writes { a.nk } else { 0 }, pf)?;
            self.mark("attention: preparation (norms, rotary, cache write)")?;
        }
        let mut x = a.args(k, p.d.eps);
        (x.qo, x.out, x.kpos) = (p.slot(Slot::Qo), p.slot(Slot::Att), pos.1);
        if let Some(g) = a.gate {
            x.gate = p.slot(a.src) + (g.off * 2) as u64;
        }
        if let Some(rp) = self.pages12(p, a.ring) {
            self.kv12_decode_at(a.ring, k, a.nq, x.qo, x.out, (tab, dup / rp.parts * t), 0)?;
            self.ka.gate(s, &x)?;
            return self.mark(if a.window > 0 { "attention (sliding, 12-bit pages)" } else { "attention (global, 12-bit pages)" });
        }
        (x.kc, x.vc, x.cap) = (kc, vc, self.ring_cap[a.ring] as i32);
        if self.ldec.is_some() && std::ptr::eq(p, &self.big) && a.lib_decode() {
            self.lib_decode_at(a, k, &x, (tab, dup / a.dparts() * t), 0)?;
            self.ka.gate(s, &x)?;
        } else {
            // (the parts kernel, a block a row, each row's keys through its own table; its merge gates)
            (x.tab, x.tstride, x.part) = (tab, (dup * t) as i64, self.part.ptr);
            self.dcalls.set(self.dcalls.get() + 1);
            self.ka.decode(s, &x)?;
        }
        self.mark(if a.window > 0 { "attention (sliding)" } else { "attention (global)" })
    }

    /// The preparation: the queries into Qo, the keys and values into the ring (or a 12-bit ring's staging rows, then its pages).
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn attn_prep(&self, p: &Part, l: usize, i: usize, a: &AttnOp, n: usize, pos: (u64, u64), head: usize) -> Result<()> {
        let mut x = a.args(n, p.d.eps);
        // (a ring of 12-bit pages: the rows' keys and values to its staging rows, each row at its own place, and from there into the pages)
        let twelve = a.writes && self.pages12(p, a.ring).is_some();
        (x.kc, x.vc, x.cap) = match &self.lossless {
            Some(lx) if twelve => (lx.rows_k.ptr, lx.rows_v.ptr, -1),
            _ => {
                let (kc, vc) = self.cache(a.ring);
                (kc, vc, self.ring_cap[a.ring] as i32)
            }
        };
        x.tab = self.table(a.ring);
        x.src = p.slot(a.src) + self.rows_before(a.stride);
        x.qo = p.slot(Slot::Qo) + self.rows_before(a.nq * a.d);
        x.qpos = pos.0;
        if let Some((q, k)) = a.weights() {
            (x.qw, x.kw) = (p.small[q].ptr, p.small[k].ptr);
        }
        if let Some(r) = a.rope {
            (x.cs, x.sn) = (self.ropes[r.table].0.ptr, self.ropes[r.table].1.ptr);
        }
        let (nx, rows) = Self::next_linear(p, l, i, n, head);
        let pf = self.pf(nx, rows, self.pf_kib[0], (a.d / 2) as u32);
        self.ka.prep(self.dev.stream, &x, a.nq + if a.writes { a.nk } else { 0 }, pf)?;
        if twelve {
            self.kv12_write(a.ring, n)?;
        }
        self.mark("attention: preparation (norms, rotary, cache write)")
    }

    /// The attention of the prepared queries over the ring into Att: a decode row's (n up to the step's rows, or a verify pass's) in parts, a prompt chunk's whole.
    unsafe fn attn_read(&self, p: &Part, a: &AttnOp, n: usize, pos: (u64, u64)) -> Result<()> {
        let s = self.dev.stream;
        let mut x = a.args(n, p.d.eps);
        x.qo = p.slot(Slot::Qo) + self.rows_before(a.nq * a.d);
        x.out = p.slot(Slot::Att) + self.rows_before(a.nq * a.d);
        x.kpos = pos.1;
        if let Some(g) = a.gate {
            x.gate = p.slot(a.src) + self.rows_before(a.stride) + (g.off * 2) as u64;
        }
        let decode = n <= self.split_rows || self.verifying.get();
        // (a ring of 12-bit pages: decode rows read the pages, then the gate; a prompt chunk's are unpacked to a bf16 ring the kernels below read, with no table)
        let twelve = self.pages12(p, a.ring).is_some();
        if twelve && n <= self.split_rows {
            self.kv12_decode(a.ring, n, a.nq, x.qo, x.out)?;
            self.ka.gate(s, &x)?;
            return self.mark(if a.window > 0 { "attention (sliding, 12-bit pages)" } else { "attention (global, 12-bit pages)" });
        }
        if twelve && !decode && self.ka.fa && !KernAttn::own_prompt(a.d) {
            // (a prompt chunk of heads of 256 or 512 over 12-bit pages: the library's prompt attention reads them where they are, and the chunk's own keys and values
            // from the staging rows the preparation wrote; the same kernel and sums as over a bf16 cache, so the same bits)
            self.kv12_prefill(a, n, x.qo, x.out)?;
            self.ka.gate(s, &x)?;
            return self.mark(if a.window > 0 { "attention (sliding, 12-bit pages)" } else { "attention (global, 12-bit pages)" });
        }
        (x.kc, x.vc, x.cap, x.tab) = if twelve {
            // (a verify pass's rows, the warp kernels, or the kind's own prompt kernel: the pages unpacked to a bf16 ring, whose values are the bf16 cache's, so the
            // same bits)
            let (kc, vc, cap) = self.kv12_stage(a.ring, self.start.get(), n)?;
            self.mark("attention: 12-bit pages unpacked to bf16")?;
            (kc, vc, cap as i32, 0)
        } else {
            let (kc, vc) = self.cache(a.ring);
            (kc, vc, self.ring_cap[a.ring] as i32, self.table(a.ring))
        };
        if decode {
            // (a verify pass's rows by the decode step's kernel, each row at its own position: a step's sums)
            if self.ldec.is_some() && std::ptr::eq(p, &self.big) && a.lib_decode() {
                // (the library's decode, the 12-bit pages' too: the same bits over either cache)
                self.lib_decode(a, n, &x)?;
                self.ka.gate(s, &x)?;
                return self.mark(if a.window > 0 { "attention (sliding)" } else { "attention (global)" });
            }
            x.part = match &self.vpart {
                Some(v) if self.verifying.get() && n > 1 => v.ptr,
                _ => self.part.ptr,
            };
            self.dcalls.set(self.dcalls.get() + 1);
            self.ka.decode(s, &x)?;
        } else {
            self.ka.prompt(&self.lib, s, &x, self.start.get())?;
        }
        self.mark(if a.window > 0 { "attention (sliding)" } else { "attention (global)" })
    }
}

/// attncheck's prompt rows in two calls (the kind's own prompt kernel: a row's bits are its position's and window's alone).
#[cfg(feature = "dev")]
const SPLIT: &str = "the same rows in two calls (the one call's bits)";

/// `glyd-engine attncheck`: the attention kind's kernels against an fp64 attention on the CPU, on random queries, keys, values and gates, with each family's head width
/// and arithmetic (Gemma 4's sliding and global heads with their scores in bf16, Nemotron-H's heads of 128, Muse Spark's window and normed gate, a soft-capped head):
/// a prompt's rows on the tensor cores and a warp a head, and each row as a decode step (its keys in SPLITS parts). The bound is on the largest difference, relative
/// to the largest value (the kernels round their probabilities to bf16 for the values, the CPU does not). Exit status 1 where one is past it.
#[cfg(feature = "dev")]
pub fn check() -> Result<bool> {
    use crate::dev::Buf;
    let dev = Dev::new(0)?;
    let (k, lib) = (KernAttn::load(&dev)?, crate::ffi::Lib::find()?);
    println!("{} (sm_{}{}): the attention kind's kernels against the CPU", dev.name, dev.cc.0, dev.cc.1);
    let mut seed = 0x2026_1123_0e00_0001u64;
    let mut rnd = |m: usize, lo: f32, hi: f32| -> Vec<f32> {
        (0..m)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                crate::muse::bf(lo + (hi - lo) * ((seed >> 40) as f32 / (1u64 << 24) as f32))
            })
            .collect()
    };
    let up = |v: &[f32]| -> Result<Buf> {
        let b = Buf::new(v.len() * 2)?;
        b.write(0, &v.iter().map(|x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>())?;
        Ok(b)
    };
    let bf = |x: f64| crate::muse::bf(x as f32) as f64;
    // (what, head, query heads, kv heads, window, the scores in bf16, scale, soft-cap, gate (normed), round, first position, rows)
    type Case = (&'static str, usize, usize, usize, usize, bool, f32, f32, Option<bool>, bool, usize, usize);
    let cases: [Case; 8] = [
        ("Gemma 4's sliding layers (head 256, window 64, scores in bf16)", 256, 16, 8, 64, true, 1.0, 0.0, None, true, 100, 70),
        ("Gemma 4's global layers (head 512, scores in bf16)", 512, 16, 2, 0, true, 1.0, 0.0, None, true, 40, 37),
        ("Nemotron-H's (head 128)", 128, 32, 2, 0, false, 128f32.powf(-0.5), 0.0, None, false, 124, 62),
        ("the Qwen3.5 layout's (head 256, gated)", 256, 16, 4, 0, false, 1.0 / 16.0, 0.0, Some(false), false, 50, 20),
        ("Muse Spark's (head 64, window 32, the gate normed)", 64, 8, 2, 32, false, 0.125, 0.0, Some(true), true, 90, 40),
        ("a soft-capped head of 128 (cap 5)", 128, 8, 2, 0, false, 128f32.powf(-0.5), 5.0, None, false, 60, 33),
        ("Nemotron-H's (head 128), a whole chunk", 128, 32, 2, 0, false, 128f32.powf(-0.5), 0.0, None, false, 128, 64),
        ("Qwen3-8B's (head 128, 32 query heads over 8)", 128, 32, 8, 0, false, 128f32.powf(-0.5), 0.0, None, false, 150, 100),
    ];
    let mut ok = true;
    for (what, d, nq, nk, window, sbf16, scale, softcap, gnorm, round, p0, n) in cases {
        let cap = p0 + n;
        let (q, kk, vv, g) = (rnd(n * nq * d, -2.0, 2.0), rnd(cap * nk * d, -2.0, 2.0), rnd(cap * nk * d, -2.0, 2.0), rnd(n * nq * d, -4.0, 4.0));
        let mut want = vec![0f64; n * nq * d];
        for t in 0..n {
            let p = p0 + t;
            let lo = if window > 0 { (p + 1).saturating_sub(window) } else { 0 };
            for h in 0..nq {
                let kv = h / (nq / nk);
                let qr = &q[(t * nq + h) * d..][..d];
                let sc: Vec<f64> = (lo..=p)
                    .map(|j| {
                        let dot: f64 = qr.iter().zip(&kk[(j * nk + kv) * d..][..d]).map(|(a, b)| *a as f64 * *b as f64).sum();
                        let s = if sbf16 { bf(bf(dot) * scale as f64) } else { dot * scale as f64 };
                        if softcap > 0.0 {
                            softcap as f64 * (s / softcap as f64).tanh()
                        } else {
                            s
                        }
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = e.iter().sum();
                let mut o: Vec<f64> = (0..d).map(|c| e.iter().enumerate().map(|(i, w)| w * vv[((lo + i) * nk + kv) * d + c] as f64).sum::<f64>() / l).collect();
                if let Some(nrm) = gnorm {
                    o.iter_mut().for_each(|x| *x = bf(*x));
                    if nrm {
                        let r = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-6).sqrt();
                        o.iter_mut().for_each(|x| *x = bf(*x * r));
                    }
                    for (c, x) in o.iter_mut().enumerate() {
                        *x *= 1.0 / (1.0 + (-g[(t * nq + h) * d + c] as f64).exp());
                    }
                }
                want[(t * nq + h) * d..][..d].copy_from_slice(&o);
            }
        }
        // (the keys and values as a ring, position p at p, and as pages: block b of the positions in the table's block nb - 1 - b)
        let nb = cap.div_ceil(16);
        let tab: Vec<i32> = (0..nb).map(|b| (nb - 1 - b) as i32).collect();
        let paged = |x: &[f32]| -> Vec<f32> {
            let w = nk * d;
            let mut o = vec![0f32; nb * 16 * w];
            for p in 0..cap {
                let r = tab[p / 16] as usize * 16 + p % 16;
                o[r * w..(r + 1) * w].copy_from_slice(&x[p * w..(p + 1) * w]);
            }
            o
        };
        let (qb, gb, pb, tb) = (up(&q)?, up(&g)?, Buf::new(4 * n)?, Buf::new(4 * nb)?);
        pb.write(0, &(0..n).map(|t| (p0 + t) as i32).collect::<Vec<i32>>())?;
        tb.write(0, &tab)?;
        let part = Buf::zeroed(n * nq * SPLITS * (d + 2) * 4)?;
        let peak = want.iter().fold(0f64, |m, x| m.max(x.abs()));
        for pages in [false, true] {
            let (kb, vb) = if pages { (up(&paged(&kk))?, up(&paged(&vv))?) } else { (up(&kk)?, up(&vv)?) };
            let mut a = AttnArgs {
                d: d as i32,
                n: n as i32,
                nq: nq as i32,
                nk: nk as i32,
                window: window as i32,
                os: (nq * d) as i32,
                gs: (nq * d) as i32,
                g_step: d as i32,
                gnorm: (gnorm == Some(true)) as i32,
                round: round as i32,
                sbf16: sbf16 as i32,
                S: SPLITS as i32,
                q_scale: 1.0,
                eps: 1e-6,
                scale,
                sl2: sl2(scale),
                softcap,
                ..AttnArgs::default()
            };
            (a.qo, a.kc, a.vc, a.cap, a.kpos, a.part) = (qb.ptr, kb.ptr, vb.ptr, cap as i32, pb.ptr, part.ptr);
            if pages {
                a.tab = tb.ptr;
            }
            if gnorm.is_some() {
                a.gate = gb.ptr;
            }
            // (the library's decode where the shape takes it: each row a sequence of its own length and window, its query heads in calls of 8 at most)
            let g = nq / nk;
            let lib_dec = d % 128 == 0 && (g <= MAX_DG || nk == 1) && softcap == 0.0;
            let sp = g.div_ceil(MAX_DG);
            let lens: Vec<i32> = (0..n).flat_map(|t| std::iter::repeat_n((p0 + t + 1) as i32, sp)).collect();
            let firsts: Vec<i32> =
                (0..n).flat_map(|t| std::iter::repeat_n(if window > 0 { (p0 + t + 1).saturating_sub(window) as i32 } else { 0 }, sp)).collect();
            let (lb, fb) = (Buf::new(4 * lens.len())?, Buf::new(4 * firsts.len())?);
            lb.write(0, &lens)?;
            fb.write(0, &firsts)?;
            let splits = lib.kv_decode_splits(1, nk as i64, cap.div_ceil(16) as i64)?;
            let ws_bytes = if lib_dec { lib.kv_decode_workspace((n * sp) as i64, nk as i64, (g / sp) as i64, d as i64, splits)? } else { 16 };
            let ws = Buf::zeroed(ws_bytes)?;
            let mut hows = vec!["a prompt's rows on the tensor cores", "a prompt's rows, a warp a head", "each row a decode step"];
            if lib_dec {
                hows.push("each row a decode step of the library's");
            }
            // (the kind's own prompt kernel: the same rows in two calls, the second's first row not at a tile's, give the one call's bits)
            let split = KernAttn::own_prompt(d).then_some(n / 2 - 3);
            if split.is_some() {
                hows.push(SPLIT);
            }
            let mut whole: Vec<u16> = Vec::new();
            let mut alone: Vec<(&str, Vec<u16>)> = Vec::new();
            for &how in &hows {
                let out = Buf::zeroed(n * nq * d * 2)?;
                let x = AttnArgs { out: out.ptr, ..a };
                // SAFETY: buffers made here of the sizes the kernels read and write; read after a synchronize.
                unsafe {
                    match how {
                        "a prompt's rows on the tensor cores" => k.prompt(&lib, dev.stream, &x, p0)?,
                        SPLIT => {
                            let h = split.unwrap_or(0);
                            let row = (h * nq * d * 2) as u64;
                            k.prompt(&lib, dev.stream, &AttnArgs { n: h as i32, ..x }, p0)?;
                            let rest = AttnArgs {
                                n: (n - h) as i32,
                                qo: x.qo + row,
                                out: x.out + row,
                                kpos: x.kpos + 4 * h as u64,
                                gate: if x.gate == 0 { 0 } else { x.gate + row },
                                ..x
                            };
                            k.prompt(&lib, dev.stream, &rest, p0 + h)?
                        }
                        "a prompt's rows, a warp a head" => k.rows(dev.stream, &x)?,
                        "each row a decode step" => k.decode(dev.stream, &x)?,
                        _ => {
                            let stride = (nq * d / sp) as i64;
                            lib.kv_decode_bf16(
                                x.qo,
                                stride,
                                (x.kc, x.vc, x.tab, 0, x.cap as i64),
                                (lb.ptr, fb.ptr, (n * sp) as i64, splits),
                                (nk as i64, (g / sp) as i64, d as i64),
                                (scale as f64, sbf16),
                                x.out,
                                stride,
                                (ws.ptr, ws_bytes),
                                dev.stream,
                            )?;
                            k.gate(dev.stream, &x)?
                        }
                    }
                }
                dev.sync()?;
                let mut got = vec![0u16; n * nq * d];
                out.read(0, &mut got)?;
                let worst = got.iter().zip(&want).map(|(g, w)| (f32::from_bits((*g as u32) << 16) as f64 - w).abs()).fold(0f64, f64::max) / peak.max(1e-30);
                // (the two calls: the one call's bits, every value)
                let differ = if how == SPLIT { got.iter().zip(&whole).filter(|(g, w)| g != w).count() } else { 0 };
                if how == "a prompt's rows on the tensor cores" {
                    whole = got.clone();
                }
                let pass = worst <= 1e-2 && differ == 0;
                ok &= pass;
                println!(
                    "{} {what}: {how}{}, {} values, largest difference {worst:.2e} of the largest value (bound 1.0e-2){}",
                    if pass { "ok  " } else { "FAIL" },
                    if pages { ", the keys in pages" } else { "" },
                    got.len(),
                    if how == SPLIT { format!(", {differ} not the one call's bits") } else { String::new() }
                );
                alone.push((how, got));
            }
            if !pages {
                continue;
            }
            // (each row through a table of its own, as a step's rows of several sequences take theirs (`Dyn::attn_lead`, rows::RowTabs): the one table copied twice a row, found at
            // a stride; the last row a padding row (position -1). The other rows' bits are the decode's above, the padding row's zeros.)
            let rows_tab: Vec<i32> = (0..2 * n).flat_map(|_| tab.iter().copied()).collect();
            let mut kp: Vec<i32> = (0..n).map(|t| (p0 + t) as i32).collect();
            kp[n - 1] = -1;
            let (lens_p, firsts_p): (Vec<i32>, Vec<i32>) =
                lens.iter().zip(&firsts).enumerate().map(|(i, (&l, &f))| if i / sp == n - 1 { (0, 0) } else { (l, f) }).unzip();
            let (rtb, kpb, lpb, fpb) = (Buf::new(4 * rows_tab.len())?, Buf::new(4 * n)?, Buf::new(4 * lens_p.len())?, Buf::new(4 * firsts_p.len())?);
            rtb.write(0, &rows_tab)?;
            kpb.write(0, &kp)?;
            lpb.write(0, &lens_p)?;
            fpb.write(0, &firsts_p)?;
            for (how, base) in &alone {
                let library = *how == "each row a decode step of the library's";
                if !library && *how != "each row a decode step" {
                    continue;
                }
                let out = Buf::zeroed(n * nq * d * 2)?;
                let x = AttnArgs { out: out.ptr, tab: rtb.ptr, tstride: (2 * nb) as i64, kpos: kpb.ptr, ..a };
                // SAFETY: as above.
                unsafe {
                    if library {
                        let stride = (nq * d / sp) as i64;
                        lib.kv_decode_bf16(
                            x.qo,
                            stride,
                            (x.kc, x.vc, rtb.ptr, (2 / sp * nb) as i64, x.cap as i64),
                            (lpb.ptr, fpb.ptr, (n * sp) as i64, splits),
                            (nk as i64, (g / sp) as i64, d as i64),
                            (scale as f64, sbf16),
                            x.out,
                            stride,
                            (ws.ptr, ws_bytes),
                            dev.stream,
                        )?;
                        k.gate(dev.stream, &x)?
                    } else {
                        k.decode(dev.stream, &x)?
                    }
                }
                dev.sync()?;
                let mut got = vec![0u16; n * nq * d];
                out.read(0, &mut got)?;
                let w = (n - 1) * nq * d;
                let (differ, pad) = (got[..w].iter().zip(&base[..w]).filter(|(x, y)| x != y).count(), got[w..].iter().filter(|&&v| v & 0x7FFF != 0).count());
                let pass = differ == 0 && pad == 0;
                ok &= pass;
                println!(
                    "{} {what}: {how}, each row through its own table, the last a padding row: {differ} values not the bits above, {pad} of the padding row's not zero",
                    if pass { "ok  " } else { "FAIL" }
                );
            }
        }
    }
    println!("{}", if ok { "all of the attention kind's kernels within their bounds" } else { "an attention kernel past its bound" });
    Ok(ok)
}

/// `glyd-engine attndecode [--lens 64,170,1024,4096,16384] [--ctx 32768] [--reps 200]`: a decode row's attention over bf16 pages, microseconds a call, at gemma-4-12B's sliding
/// layers (8 kv heads of 256, 16 query heads, a window of 1,024, the scores in bf16) and global ones (1 of 512, 16 query heads in two calls' 8) and the Qwen3.5 layout's (2 of 256,
/// 8 query heads): the engine's parts kernel and its merge (the decode before the library's took these shapes) and the library's (glyd_gpu_kv_decode_bf16, with the plan the engine
/// makes for a context of --ctx). Each a CUDA graph of --reps calls, as a step's graph runs them; and how many of the library's outputs are not the parts kernel's bits (their
/// sums are in another order: some).
#[cfg(feature = "dev")]
pub fn time_decode(a: &crate::args::Args) -> Result<()> {
    use crate::dev::{Buf, Timer};
    let dev = Dev::new(0)?;
    let (k, lib) = (KernAttn::load(&dev)?, crate::ffi::Lib::find()?);
    let s = dev.stream;
    let (ctx, reps) = (a.num("ctx", 32768usize)?, a.num("reps", 200usize)?);
    let lens: Vec<usize> = a.flag("lens").unwrap_or("64,170,1024,4096,16384").split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x| x >= 1).collect();
    if std::env::var_os("GLYD_KV_FLOOR").is_none() {
        lib.kv_floor_set(super::kvpages::DECODE_FLOOR)?;
    }
    println!(
        "{} (sm_{}{}): a decode row's attention over bf16 pages, a context of {ctx} planned, {reps} calls a graph (us a call)",
        dev.name, dev.cc.0, dev.cc.1
    );
    // (what, kv heads, head, query heads, window, the scores in bf16, the model's most kv heads: the engine plans its splits for them)
    let shapes: [(&str, usize, usize, usize, usize, bool, usize); 3] = [
        ("gemma-4-12B sliding (8 x 256, 16 queries, window 1024)", 8, 256, 16, 1024, true, 8),
        ("gemma-4-12B global (1 x 512, 16 queries)", 1, 512, 16, 0, true, 8),
        ("the Qwen3.5 layout (2 x 256, 8 queries)", 2, 256, 8, 0, false, 2),
    ];
    let mut seed = 0x2026_1123_0d0c_0001u64;
    let mut bits = |m: usize| -> Vec<u16> {
        (0..m)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (0x3C00 | (seed >> 52) as u16 & 0x03FF) ^ ((seed >> 63) as u16) << 15
                // (a value of 2^-7 to 2, either sign)
            })
            .collect()
    };
    for (what, nk, d, nq, window, sbf16, heads) in shapes {
        let (g, sp) = (nq / nk, (nq / nk).div_ceil(MAX_DG));
        let splits = lib.kv_decode_splits(1, heads as i64, ctx.div_ceil(16) as i64)?;
        for &len in &lens {
            let blocks = len.div_ceil(16);
            // (the pages in reverse order of the table's)
            let tab: Vec<i32> = (0..blocks).map(|b| (blocks - 1 - b) as i32).collect();
            let up = |x: &[u16]| -> Result<Buf> {
                let b = Buf::new(2 * x.len())?;
                b.write(0, x)?;
                Ok(b)
            };
            let (qb, kb, vb) = (up(&bits(nq * d))?, up(&bits(blocks * 16 * nk * d))?, up(&bits(blocks * 16 * nk * d))?);
            let (tb, pb) = (Buf::new(4 * blocks)?, Buf::new(4)?);
            tb.write(0, &tab)?;
            pb.write(0, &[(len - 1) as i32])?;
            let first = if window > 0 { len.saturating_sub(window) } else { 0 };
            let (lb, fb) = (Buf::new(4 * sp)?, Buf::new(4 * sp)?);
            lb.write(0, &vec![len as i32; sp])?;
            fb.write(0, &vec![first as i32; sp])?;
            let part = Buf::zeroed(nq * SPLITS * (d + 2) * 4)?;
            let ws_bytes = lib.kv_decode_workspace(sp as i64, nk as i64, (g / sp) as i64, d as i64, splits)?;
            let ws = Buf::zeroed(ws_bytes)?;
            let (o_old, o_lib) = (Buf::zeroed(2 * nq * d)?, Buf::zeroed(2 * nq * d)?);
            let scale = 1f32;
            let x = AttnArgs {
                qo: qb.ptr,
                kc: kb.ptr,
                vc: vb.ptr,
                tab: tb.ptr,
                cap: (blocks * 16) as i32,
                kpos: pb.ptr,
                part: part.ptr,
                d: d as i32,
                n: 1,
                nq: nq as i32,
                nk: nk as i32,
                window: window as i32,
                os: (nq * d) as i32,
                sbf16: sbf16 as i32,
                S: SPLITS as i32,
                q_scale: 1.0,
                scale,
                sl2: sl2(scale),
                ..AttnArgs::default()
            };
            let stride = (nq * d / sp) as i64;
            // SAFETY: buffers made here of the sizes the kernels read and write, alive while the graphs run; read after a synchronize.
            let old = || unsafe { k.decode(s, &AttnArgs { out: o_old.ptr, ..x }) };
            let new = || unsafe {
                lib.kv_decode_bf16(
                    qb.ptr,
                    stride,
                    (kb.ptr, vb.ptr, tb.ptr, 0, 0),
                    (lb.ptr, fb.ptr, sp as i64, splits),
                    (nk as i64, (g / sp) as i64, d as i64),
                    (scale as f64, sbf16),
                    o_lib.ptr,
                    stride,
                    (ws.ptr, ws_bytes),
                    s,
                )
            };
            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                let graph = dev.capture(|| (0..reps).try_for_each(|_| f()))?;
                // SAFETY: the graph's buffers are alive (above).
                unsafe { graph.launch(s)? };
                dev.sync()?;
                let t = Timer::new()?;
                t.start(s)?;
                unsafe { graph.launch(s)? };
                Ok(t.stop(s)? as f64 * 1e3 / reps as f64)
            };
            let (t_old, t_new) = (time(&old)?, time(&new)?);
            let (mut y, mut z) = (vec![0u16; nq * d], vec![0u16; nq * d]);
            o_old.read(0, &mut y)?;
            o_lib.read(0, &mut z)?;
            let differ = y.iter().zip(&z).filter(|(p, q)| p != q).count();
            println!(
                "{what}, {len:>6} tokens: parts kernel + merge {t_old:>7.2}; library ({splits} splits) {t_new:>7.2} ({:+.0}%); {differ} of {} outputs not the parts kernel's bits",
                100.0 * (t_new / t_old - 1.0),
                y.len()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::{Act, Attn, AttnRounding, GateFrom, Norm, OutGate, Rope};

    fn kind(d: usize, gate: Option<GateFrom>) -> Attn {
        Attn {
            heads: 8,
            kv_heads: 2,
            head_dim: d,
            v_head_dim: d,
            window: None,
            sinks: false,
            rope: Some(Rope::plain(1e4, 64)),
            q_norm: Some(Norm::rms_plus_one(1e-6)),
            k_norm: Some(Norm::rms_plus_one(1e-6)),
            v_norm: None,
            gate: gate.map(|from| OutGate { act: Act::Sigmoid, from, norm: None }),
            softcap: None,
            bias: false,
            scale: (d as f32).powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::Flash,
            kv_from: None,
        }
    }

    /// The scale the kernels take is the one each family's own attention took: Gemma 4's log2(e), the Qwen3.5 layout's 256^-0.5 and Nemotron-H's 128^-0.5 in
    /// log2 units as their engines made them in f64.
    #[test]
    fn the_scores_factor_is_the_old_kernels() {
        let old = |d: f64| (1.0 / d.sqrt() * std::f64::consts::LOG2_E) as f32;
        assert_eq!(sl2(1.0), std::f32::consts::LOG2_E);
        for d in [64usize, 128, 256, 512] {
            assert_eq!(sl2((d as f32).powf(-0.5)), old(d as f64), "head {d}");
        }
    }

    /// Where a kind's q, k, v and gate are in the fused projection's rows: an in-query gate interleaved with the queries (the Qwen3.5 layout), an own gate after the
    /// values (Muse Spark), K = V with no value heads (Gemma 4's global layers).
    #[test]
    fn the_projections_layout_follows_the_kind() {
        let q = AttnOp::of(&kind(256, Some(GateFrom::InQuery)), Slot::Mix, 0, Some(0), Some((2, 3))).unwrap();
        assert_eq!((q.q_step, q.k_off, q.v_off, q.stride), (512, 8 * 512, Some(8 * 512 + 2 * 256), 8 * 512 + 4 * 256));
        assert_eq!(q.gate, Some(GateAt { off: 256, step: 512, norm: false }));
        assert_eq!(q.norm, QkNorm::Plus1 { q: 2, k: 3 });
        assert_eq!(q.rope, Some(RopeAt { table: 0, span: 64, pairs: 32 }));
        let m = AttnOp::of(&Attn { rounding: AttnRounding::Sdpa, ..kind(64, Some(GateFrom::Own)) }, Slot::Mix, 0, Some(0), Some((2, 3))).unwrap();
        assert_eq!((m.q_step, m.k_off, m.v_off, m.stride), (64, 8 * 64, Some(10 * 64), 20 * 64));
        assert_eq!(m.gate, Some(GateAt { off: 12 * 64, step: 64, norm: false }));
        // (Sdpa: the scale on the queries, before the rope; the scores unscaled)
        assert_eq!((m.q_scale, m.scale, m.round, m.sbf16), (0.125, 1.0, true, false));
        assert_eq!((q.q_scale, q.scale, q.round, q.sbf16), (1.0, 1.0 / 16.0, false, false));
        let mut g = kind(512, None);
        g.k_eq_v = true;
        let g = AttnOp::of(&Attn { rounding: AttnRounding::Eager, ..g }, Slot::Proj, 0, Some(1), Some((2, 3))).unwrap();
        assert_eq!((g.round, g.sbf16), (true, true));
        assert_eq!((g.v_off, g.stride), (None, 10 * 512));
        // (a head the kernels have no instance for, and more query heads to a key-value head than a block takes, are refused)
        assert!(AttnOp::of(&kind(96, None), Slot::Mix, 0, Some(0), Some((2, 3))).is_err());
        let mut wide = kind(128, None);
        (wide.heads, wide.kv_heads) = (64, 2);
        assert!(AttnOp::of(&wide, Slot::Mix, 0, Some(0), Some((2, 3))).is_err());
    }

    /// kernels/attn.cu's AttnArgs: 14 pointers and a long long, 21 ints, 5 floats, no padding.
    #[test]
    fn the_arguments_are_the_kernels_struct() {
        assert_eq!(std::mem::size_of::<AttnArgs>(), 15 * 8 + 26 * 4);
        assert_eq!(std::mem::offset_of!(AttnArgs, tstride), 14 * 8);
    }
}
