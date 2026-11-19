//! A matrix in the library's lossless FP8 format (glyd_gpu.h's f8 functions: E4M3 bytes held bit for bit, about 6.7 bits a value) with its 128 x 128 block scales: an FP8 checkpoint's matrices held as
//! released, multiplied as they are (`f8_linear`, 1 to 64 rows a call; a longer chunk in calls of 64 rows). A layer's experts are a stack of such matrices packed as one (`F8Experts`, `f8_moe`).

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Lib, Result};
use crate::weights::{as_bytes, upload_bytes};

pub struct F8Mat {
    /// Layout H (glyd_gpu.h's f8h functions): an HBM GPU's, fewer instructions a value for about 0.4 more bits.
    pub h: bool,
    core: Buf,
    heads: Buf,
    var: Buf,
    scale: Buf,
    pub n: usize,
    pub k: usize,
    pub bytes: usize,
    /// The whole matrix's rows (of an expert's, for a stack) where this one holds this GPU's slice of them (src/dynamic/tp.rs: a product is gathered after
    /// it); 0: the whole.
    pub tp_rows: usize,
}

/// Rows a product takes a call.
const ROWS: usize = 64;

/// Whether a matrix on `dev` takes layout H: an HBM GPU's (compute capability 8.0, 9.0, 10.0: A100, H100, B200), where the product waits on the decode's instructions, not on memory
/// (research/dsv4/a100-stalls-2026-10-19); GLYD_F8_LAYOUT=h or c sets it.
pub fn layout_h(dev: &Dev) -> bool {
    match crate::tune!("GLYD_F8_LAYOUT").as_deref() {
        Some("h") => true,
        Some("c") => false,
        _ => {
            let (major, minor) = dev.cc_pair();
            minor == 0 && (8..=10).contains(&major)
        }
    }
}

impl F8Mat {
    /// W [n, k] of E4M3 bytes `q` with its fp32 scales of 128 x 128 blocks (`scales`: [ceil(n / 128), ceil(k / 128)]), packed on `dev` (the bytes uploaded, sized, packed, the upload freed); where
    /// `check`, unpacked again and compared with `q` byte for byte.
    pub fn pack(dev: &Dev, lib: &Lib, q: &[u8], scales: &[f32], n: usize, k: usize, check: bool) -> Result<F8Mat> {
        if q.len() != n * k || scales.len() != n.div_ceil(128) * k.div_ceil(128) {
            return err(format!("{} bytes and {} scales for an FP8 matrix of {n} x {k}", q.len(), scales.len()));
        }
        let (s, (ni, ki), h) = (dev.stream, (n as i64, k as i64), layout_h(dev));
        let w = upload_bytes(q)?;
        let patches = n.div_ceil(64) * k.div_ceil(128);
        let heads = Buf::new(patches * 8 * 4)?;
        let wsb = lib.f8_size_workspace(ni, ki)?;
        let ws = Buf::new(wsb.max(8))?;
        // SAFETY: live device buffers of the sizes glyd_gpu.h gives; f8_size waits for the stream, and the stream is synchronized before the upload is freed.
        let core = Buf::new(patches * if h { 7168 } else { 6144 })?;
        let var = unsafe {
            let words = lib.f8_size(h, w.ptr, ki, ni, ki, heads.ptr, ws.ptr, wsb, s)?;
            let var = Buf::new(words.max(1) as usize * 4)?;
            lib.f8_pack(h, w.ptr, ki, ni, ki, heads.ptr, core.ptr, var.ptr, words, s)?;
            var
        };
        if check {
            let back = Buf::new(n * k)?;
            // SAFETY: as above.
            unsafe { lib.f8_unpack(h, core.ptr, heads.ptr, var.ptr, ni, ki, back.ptr, ki, s)? };
            dev.sync()?;
            let mut got = vec![0u8; n * k];
            back.read(0, &mut got)?;
            if got != q {
                return err(format!("an FP8 matrix of {n} x {k} unpacked to other bytes than it was packed from"));
            }
        }
        dev.sync()?;
        let scale = upload_bytes(as_bytes(scales))?;
        let bytes = core.bytes + heads.bytes + var.bytes + scale.bytes;
        Ok(F8Mat { h, core, heads, var, scale, n, k, bytes, tp_rows: 0 })
    }

    /// The experts' product: this matrix the stack of `experts` matrices [n / experts, k], each pair of `plan` (pairs `t k`, `t` tokens of `k` choices, t at most 64) times its expert's into its row of
    /// y [pairs, n / experts] in the plan's order; x's row of a pair its token's (`gather`: x [t, k]) or its own (x [pairs, k], the plan's order).
    ///
    /// # Safety
    /// As `linear`; `done` at least `experts.min(t k)` ceil(n / experts / 128) zeroed counters, `ws` `moe_workspace`'s bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn moe(&self, lib: &Lib, experts: usize, x: u64, plan: u64, t: usize, k: usize, gather: bool, y: u64, ws: u64, ws_bytes: usize, done: u64, s: crate::ffi::Stream) -> Result<()> {
        if t > ROWS {
            return err(format!("{t} tokens in one experts' product: at most {ROWS}"));
        }
        let (slots, rows) = (experts.min(t * k) as i64, self.n / experts);
        // (this GPU's slice of each expert's rows: its products, then every GPU's gathered into y)
        let (c, out) = self.tp(s, t * k * rows)?;
        lib.f8_moe(self.h, self.core.ptr, self.heads.ptr, self.var.ptr, experts as i64, rows as i64, self.k as i64, self.scale.ptr, x, plan, k as i64, gather, t as i64, slots, out.unwrap_or(y), ws, ws_bytes, done, s)?;
        match (c, out) {
            (Some(c), Some(o)) => c.gather(s, o, t * k, rows, y, self.tp_rows),
            _ => Ok(()),
        }
    }

    /// The whole matrix's rows: `tp_rows` where it is a slice of them, else its own.
    pub fn whole_n(&self) -> usize {
        if self.tp_rows > 0 { self.tp_rows } else { self.n }
    }

    /// Where a slice's product of `values` bf16 goes before its gather (this GPU's group and room), or nothing for a whole matrix.
    fn tp(&self, s: crate::ffi::Stream, values: usize) -> Result<(Option<std::rc::Rc<crate::dynamic::tp::Comm>>, Option<u64>)> {
        if self.tp_rows == 0 {
            return Ok((None, None));
        }
        let c = crate::dynamic::tp::comm().ok_or_else(|| crate::ffi::Error("an FP8 matrix sliced across GPUs, on a thread of no group of them".into()))?;
        let t = c.temp(s, values * 2)?;
        Ok((Some(c), Some(t)))
    }

    /// The experts' product's workspace for up to `t` tokens of `k` choices.
    pub fn moe_workspace(&self, lib: &Lib, experts: usize, t: usize, k: usize) -> Result<usize> {
        let mut need = 0;
        for t in 1..=t.min(ROWS) {
            need = need.max(lib.f8_moe_workspace((self.n / experts) as i64, self.k as i64, t as i64, experts.min(t * k) as i64)?);
        }
        Ok(need)
    }

    /// The library's workspace for products of up to `m` rows.
    pub fn workspace(&self, lib: &Lib, m: usize) -> Result<usize> {
        lib.f8_linear_workspace(self.n as i64, self.k as i64, m.clamp(1, ROWS) as i64)
    }

    /// y [m, n] = x [m, k] W^T (bf16), in calls of up to 64 rows.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `done` ceil(n / 128) zeroed counters; `ws` at least `workspace` bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear(&self, lib: &Lib, x: u64, m: usize, y: u64, ws: u64, ws_bytes: usize, done: u64, s: crate::ffi::Stream) -> Result<()> {
        let (n, k) = (self.n as i64, self.k as i64);
        // (this GPU's slice of the rows: its products, then every GPU's gathered into y)
        let (c, out) = self.tp(s, m * self.n)?;
        let yy = out.unwrap_or(y);
        for r0 in (0..m).step_by(ROWS) {
            let mm = (m - r0).min(ROWS) as i64;
            lib.f8_linear(self.h, self.core.ptr, self.heads.ptr, self.var.ptr, n, k, 0, self.scale.ptr, (k + 127) / 128, x + (r0 as i64 * k * 2) as u64, mm, yy + (r0 as i64 * n * 2) as u64, ws, ws_bytes, done, s)?;
        }
        match (c, out) {
            (Some(c), Some(o)) => c.gather(s, o, m, self.n, y, self.tp_rows),
            _ => Ok(()),
        }
    }
}

/// A layer's experts in the FP8 format: the first matrices (gate and up of each, [2 inter, hidden]) and the second (down, [hidden, inter]) of all of them, each stacked and packed as one (`F8Mat::moe`).
pub struct F8Experts {
    pub up: F8Mat,
    pub down: F8Mat,
    pub experts: usize,
}

/// Tokens an experts' product takes a call (a longer chunk: in parts of this many).
pub const MOE_TOKENS: usize = ROWS;
