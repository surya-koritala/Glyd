//! The attention layer kind (`runtime::kinds::Attn`) as one operation of the executor (`Op::Attn`), for every dense-attention layer of every family (#135, #141).
//! What a family's layer is, is the operation's parameters (`AttnOp`): its heads and their width, where its q, k, v and gate are in the projection's rows, its norms,
//! rotary, window, scale, soft-cap and gate, and its ring of keys and values; its rounding points the kind's (`kinds::AttnRounding`). One set of kernels runs them all
//! (kernels/attn.cu): the preparation, then a decode row's attention (the keys in parts, combined) or a prompt chunk's, and the gate. A prompt chunk of heads of
//! 128 to 512 takes the GPU library's prompt kernel, one for both caches: over bf16 rings or pages `glyd_gpu_kv_prefill_bf16`, over 12-bit pages (src/dynamic/kvpages.rs)
//! `glyd_gpu_kv_prefill`, the same tiles and sums, so the same bits; heads of 64 take attn_fa64. A decode row likewise (`LibDecode`): over 12-bit pages
//! `kv_decode_from`, over a bf16 cache `glyd_gpu_kv_decode_bf16`, one kernel, plan and merge, where the shape is the pages' (heads of 128 to 512, up to 8 query
//! heads a kv head or one kv head, no soft-cap); others (Muse Spark's heads of 64, Nemotron-H's 16 query heads over 2) the engine's parts kernel.
//!
//! The arithmetic of each family is the one its own operations had, bit for bit where the kernel is the same (the preparation, the decode and the gate of Gemma 4,
//! the Qwen3.5 layout and Nemotron-H); a prompt chunk's is now the library's kernel for all (Gemma 4's g4_fa256, the Qwen3.5 layout's warp a head, Nemotron-H's
//! dense-engine kernel and Muse Spark's key at a time before): the same function, sums in another order. Muse Spark's decode is the parts kernel too.

use super::{Dyn, Part, Slot};
use crate::dev::Dev;
use crate::ffi::{err, Result, Stream};
use crate::gemma4::G4Pf;
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
        Ok(Some(LibDecode { splits, ws, kinds, lens, first }))
    }
}

/// Dynamic shared memory of a warp kernel's block of head `d` and `rg` warps that score.
pub(crate) fn rows_smem(d: usize, rg: usize) -> u32 {
    ((KB * (d / 2 + 1) + KB * d / 2 + rg * d / 2) * 4) as u32
}

/// Dynamic shared memory of a tensor-core block (attn_fa_t): Q [64][d + 8], K and V [32][d + 8] bf16.
fn fa_smem(d: usize) -> u32 {
    ((64 * (d + 8) + 2 * 32 * (d + 8)) * 2) as u32
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
        })
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
    fn weights(&self) -> Option<(usize, usize)> {
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
    fa64: crate::kt::attn_fa64,
    /// Whether prompt chunks take the tensor-core kernels (the library's for heads of 128 to 512, attn_fa64 for 64; a development build's GLYD_ATTN_FA=0: the warp
    /// kernels).
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
            fa64: dev.typed()?,
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

    /// A prompt chunk's attention of `a.n` rows into `a.out` over a bf16 cache, then the gate: the GPU library's kernel for heads of 128 to 512 (`lib`'s
    /// glyd_gpu_kv_prefill_bf16, the one its 12-bit pages' prompt attention is: `start` the rows' first position, as it takes them), attn_fa64 for heads of 64, the warp
    /// kernels where the tensor-core ones are off.
    ///
    /// # Safety
    /// As `prep`; the rows are at positions `start` .. `start + a.n` (`a.kpos` holds them).
    pub(crate) unsafe fn prompt(&self, lib: &crate::ffi::Lib, s: Stream, a: &AttnArgs, start: usize) -> Result<()> {
        let (n, d, nq, nk) = (a.n as usize, a.d as usize, a.nq as u32, a.nk as i64);
        if !self.fa {
            return self.rows(s, a);
        }
        if d == 64 {
            launch!(self.fa64, [n.div_ceil(64) as u32, nq, 1], [128, 1, 1], smem fa_smem(64), s; *a)?;
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
    /// A pass's rows' lengths and first tokens for each kind of the library's decode calls, once for every layer (`n` rows at `pos`, each part's at its rows).
    pub(super) unsafe fn lib_decode_prepare(&self, n: usize, pos: (u64, u64)) -> Result<()> {
        let Some(ld) = &self.ldec else { return Ok(()) };
        let parts: Vec<(usize, usize)> = if self.parts.is_empty() { vec![(0, n)] } else { self.parts.iter().map(|q| (q.row0, q.rows)).collect() };
        for (row0, rows) in parts {
            for (i, &(window, sp)) in ld.kinds.iter().enumerate() {
                let at = 4 * (row0 * sp) as u64;
                self.kern.kv12_rows(self.dev.stream, pos.1 + 4 * row0 as u64, rows, 0, window, sp, 0, ld.lens[i].ptr + at, ld.first[i].ptr + at)?;
            }
        }
        Ok(())
    }

    /// A decode call of the library over the bf16 cache `x` names (its table, or its ring of `x.cap` places): the n rows of the part an operation runs on now, each
    /// row's query heads in `a.dparts()` calls' sequences.
    unsafe fn lib_decode(&self, a: &AttnOp, n: usize, x: &AttnArgs) -> Result<()> {
        let ld = self.ldec.as_ref().expect("a library decode");
        let sp = a.dparts();
        let k = ld.kinds.iter().position(|&k| k == (a.window, sp)).expect("a kind of the description's");
        let at = 4 * (self.row0() * sp) as u64;
        let stride = (a.nq * a.d / sp) as i64;
        self.lib.kv_decode_bf16(
            x.qo,
            stride,
            (x.kc, x.vc, x.tab, 0, x.cap as i64),
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
        self.mark("prep")
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
        if twelve && !decode && self.ka.fa {
            // (a prompt chunk over 12-bit pages: the library's prompt attention reads them where they are, and the chunk's own keys and values from the staging rows the
            // preparation wrote; the same kernel and sums as over a bf16 cache, so the same bits)
            self.kv12_prefill(a, n, x.qo, x.out)?;
            self.ka.gate(s, &x)?;
            return self.mark(if a.window > 0 { "attention (sliding, 12-bit pages)" } else { "attention (global, 12-bit pages)" });
        }
        (x.kc, x.vc, x.cap, x.tab) = if twelve {
            // (a verify pass's rows, or the warp kernels: the pages unpacked to a bf16 ring)
            let (kc, vc, cap) = self.kv12_stage(a.ring, self.start.get(), n)?;
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
            self.ka.decode(s, &x)?;
        } else {
            self.ka.prompt(&self.lib, s, &x, self.start.get())?;
        }
        self.mark(if a.window > 0 { "attention (sliding)" } else { "attention (global)" })
    }
}

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
    let cases: [Case; 7] = [
        ("Gemma 4's sliding layers (head 256, window 64, scores in bf16)", 256, 16, 8, 64, true, 1.0, 0.0, None, true, 100, 70),
        ("Gemma 4's global layers (head 512, scores in bf16)", 512, 16, 2, 0, true, 1.0, 0.0, None, true, 40, 37),
        ("Nemotron-H's (head 128)", 128, 32, 2, 0, false, 128f32.powf(-0.5), 0.0, None, false, 124, 62),
        ("the Qwen3.5 layout's (head 256, gated)", 256, 16, 4, 0, false, 1.0 / 16.0, 0.0, Some(false), false, 50, 20),
        ("Muse Spark's (head 64, window 32, the gate normed)", 64, 8, 2, 32, false, 0.125, 0.0, Some(true), true, 90, 40),
        ("a soft-capped head of 128 (cap 5)", 128, 8, 2, 0, false, 128f32.powf(-0.5), 5.0, None, false, 60, 33),
        ("Nemotron-H's (head 128), a whole chunk", 128, 32, 2, 0, false, 128f32.powf(-0.5), 0.0, None, false, 128, 64),
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
            let hows: &[&str] = if lib_dec {
                &["a prompt's rows on the tensor cores", "a prompt's rows, a warp a head", "each row a decode step", "each row a decode step of the library's"]
            } else {
                &["a prompt's rows on the tensor cores", "a prompt's rows, a warp a head", "each row a decode step"]
            };
            for &how in hows {
                let out = Buf::zeroed(n * nq * d * 2)?;
                let x = AttnArgs { out: out.ptr, ..a };
                // SAFETY: buffers made here of the sizes the kernels read and write; read after a synchronize.
                unsafe {
                    match how {
                        "a prompt's rows on the tensor cores" => k.prompt(&lib, dev.stream, &x, p0)?,
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
                let pass = worst <= 1e-2;
                ok &= pass;
                println!(
                    "{} {what}: {how}{}, {} values, largest difference {worst:.2e} of the largest value (bound 1.0e-2)",
                    if pass { "ok  " } else { "FAIL" },
                    if pages { ", the keys in pages" } else { "" },
                    got.len()
                );
            }
        }
    }
    println!("{}", if ok { "all of the attention kind's kernels within their bounds" } else { "an attention kernel past its bound" });
    Ok(ok)
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

    /// kernels/attn.cu's AttnArgs: 14 pointers, 21 ints, 5 floats, no padding.
    #[test]
    fn the_arguments_are_the_kernels_struct() {
        assert_eq!(std::mem::size_of::<AttnArgs>(), 14 * 8 + 26 * 4);
    }
}
