//! A matrix in the library's lossless FP8 format (glyd_gpu.h's f8 functions: E4M3 bytes held bit for bit, about 6.7 bits a value) with its 128 x 128 block scales: an FP8 checkpoint's matrices held as
//! released, multiplied as they are (`f8_linear`, 1 to 64 rows a call; a longer chunk in calls of 64 rows). A layer's experts are a stack of such matrices packed as one (`F8Experts`, `f8_moe`).

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Lib, Result};
use crate::gpu_ops::GpuOps;
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
    /// Where the matrix is the expert cache's places for a stack's experts (`F8Places::places`, R12): an expert's rows and patches, each place's var words.
    pub place: Option<F8Place>,
}

/// The places of an FP8 stack in the expert cache (`F8Places`): an expert's rows and its patches, and the var words of a place (place s's var part at s `var_words`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F8Place {
    pub rows: usize,
    pub patches: usize,
    pub var_words: usize,
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
        Ok(F8Mat { h, core, heads, var, scale, n, k, bytes, tp_rows: 0, place: None })
    }

    /// The experts' product: this matrix the stack of `experts` matrices [n / experts, k], each pair of `plan` (pairs `t k`, `t` tokens of `k` choices, t at most 64) times its expert's into its row of
    /// y [pairs, n / experts] in the plan's order; x's row of a pair its token's (`gather`: x [t, k]) or its own (x [pairs, k], the plan's order).
    ///
    /// # Safety
    /// As `linear`; `done` at least `experts.min(t k)` ceil(n / experts / 128) zeroed counters, `ws` `moe_workspace`'s bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn moe(
        &self,
        lib: &Lib,
        experts: usize,
        x: u64,
        plan: u64,
        t: usize,
        k: usize,
        gather: bool,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        s: crate::ffi::Stream,
    ) -> Result<()> {
        if t > ROWS {
            return err(format!("{t} tokens in one experts' product: at most {ROWS}"));
        }
        let (slots, rows) = (experts.min(t * k) as i64, self.expert_rows(experts));
        // (this GPU's slice of each expert's rows: its products, then every GPU's gathered into y)
        let (c, out) = self.tp(s, t * k * rows)?;
        lib.f8_moe(
            self.h,
            self.core.ptr,
            self.heads.ptr,
            self.var.ptr,
            experts as i64,
            rows as i64,
            self.k as i64,
            self.scale.ptr,
            x,
            plan,
            k as i64,
            gather,
            t as i64,
            slots,
            out.unwrap_or(y),
            ws,
            ws_bytes,
            done,
            s,
        )?;
        match (c, out) {
            (Some(c), Some(o)) => c.gather(s, o, t * k, rows, y, self.tp_rows),
            _ => Ok(()),
        }
    }

    /// An expert's rows in the stack: the places' expert's, else the matrix's rows shared by `experts`.
    fn expert_rows(&self, experts: usize) -> usize {
        self.place.map_or(self.n / experts.max(1), |p| p.rows)
    }

    /// The places the matrix holds where it is the expert cache's (`F8Places::places`); 0 otherwise.
    pub fn places(&self) -> usize {
        self.place.map_or(0, |p| self.n / p.rows)
    }

    /// Where the matrix is the expert cache's places: its heads' device address, their number, a place's and its var words (for `MoeKern::f8_rebase`).
    pub fn place_heads(&self) -> Option<(u64, usize, usize, usize)> {
        self.place.map(|p| (self.heads.ptr, self.places() * p.patches, p.patches, p.var_words))
    }

    /// The four parts as the GPU holds them, read back: the core's bytes, the heads, the var words, the scales' bytes.
    fn read_back(&self) -> Result<(Vec<u8>, Vec<u32>, Vec<u32>, Vec<u8>)> {
        let (mut core, mut heads, mut var, mut scale) =
            (vec![0u8; self.core.bytes], vec![0u32; self.heads.bytes / 4], vec![0u32; self.var.bytes / 4], vec![0u8; self.scale.bytes]);
        self.core.read(0, &mut core)?;
        self.heads.read(0, &mut heads)?;
        self.var.read(0, &mut var)?;
        self.scale.read(0, &mut scale)?;
        Ok((core, heads, var, scale))
    }

    /// The whole matrix's rows: `tp_rows` where it is a slice of them, else its own.
    pub fn whole_n(&self) -> usize {
        if self.tp_rows > 0 {
            self.tp_rows
        } else {
            self.n
        }
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
            need = need.max(lib.f8_moe_workspace(self.expert_rows(experts) as i64, self.k as i64, t as i64, experts.min(t * k) as i64)?);
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
            lib.f8_linear(
                self.h,
                self.core.ptr,
                self.heads.ptr,
                self.var.ptr,
                n,
                k,
                0,
                self.scale.ptr,
                (k + 127) / 128,
                x + (r0 as i64 * k * 2) as u64,
                mm,
                yy + (r0 as i64 * n * 2) as u64,
                ws,
                ws_bytes,
                done,
                s,
            )?;
        }
        match (c, out) {
            (Some(c), Some(o)) => c.gather(s, o, m, self.n, y, self.tp_rows),
            _ => Ok(()),
        }
    }
}

/// The matrix of `members` stacked by rows as a checkpoint releases it in FP8, each member E4M3 with its fp32 scales of 128 x 128 blocks beside it
/// (`formats::scales_beside`): its rows and columns, where every member is so held and their blocks stack (each member but the last whole blocks of rows);
/// None otherwise. The header alone is read.
pub fn released_shape(ck: &glyd_gpu::safetensors::Checkpoint, members: &[String]) -> Option<(usize, usize)> {
    let (mut rows, mut cols) = (0, 0);
    for (i, n) in members.iter().enumerate() {
        crate::formats::scales_beside(ck, n)?;
        let (_, t) = ck.get(n)?;
        let (r, c) = (t.shape[0] as usize, t.shape[1] as usize);
        if (cols != 0 && c != cols) || (i + 1 < members.len() && !r.is_multiple_of(128)) {
            return None;
        }
        (rows, cols) = (rows + r, c);
    }
    (rows > 0).then_some((rows, cols))
}

/// Rows `r0 .. r0 + rn` of the matrix of `members` that `released_shape` takes: their E4M3 bytes and fp32 scales ([ceil(rn / 128), ceil(cols / 128)]), `r0`
/// and, but at the end, `rn` whole blocks of 128 rows.
pub fn released(ck: &glyd_gpu::safetensors::Checkpoint, members: &[String], (r0, rn): (usize, usize)) -> Result<(Vec<u8>, Vec<f32>)> {
    let (mut q, mut sc) = (Vec::new(), Vec::new());
    for n in members {
        let s = crate::formats::scales_beside(ck, n).ok_or_else(|| crate::ffi::Error(format!("{n}: no fp32 block scales beside it")))?;
        q.extend(ck.read(n).map_err(|e| crate::ffi::Error(format!("{n}: {e}")))?);
        sc.extend(ck.read(&s).map_err(|e| crate::ffi::Error(format!("{s}: {e}")))?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
    }
    let cols = members.first().and_then(|n| ck.get(n)).map_or(0, |(_, t)| t.shape[1] as usize);
    let gc = cols.div_ceil(128);
    if r0 % 128 != 0 || (r0 + rn) * cols > q.len() {
        return err(format!("rows {r0} .. {} of an FP8 matrix of {} rows: a slice starts at a whole block of 128 rows", r0 + rn, q.len() / cols.max(1)));
    }
    Ok((q[r0 * cols..(r0 + rn) * cols].to_vec(), sc[r0 / 128 * gc..(r0 + rn).div_ceil(128) * gc].to_vec()))
}

/// A layer's experts in the FP8 format: the first matrices (gate and up of each, [2 inter, hidden]) and the second (down, [hidden, inter]) of all of them, each stacked and packed as one (`F8Mat::moe`).
pub struct F8Experts {
    pub up: F8Mat,
    pub down: F8Mat,
    pub experts: usize,
}

/// Tokens an experts' product takes a call (a longer chunk: in parts of this many).
pub const MOE_TOKENS: usize = ROWS;

/// The raw bytes of experts `F8Places::pack` packs at a time on the GPU.
const PACK_BYTES: usize = 256 << 20;

/// A stack's experts in the FP8 format for the expert cache (R12, src/tiering.rs): each expert's core, heads, var and scales at strides of their own, every expert end to end, as the store keeps them in the computer's memory and
/// copies one into a place (`places`). An expert's heads hold its var parts' bases from the start of its own var, which is padded with zeros to `var_words` (more than any expert's): a place's heads are made its own var's
/// (base modulo `var_words`, plus the place's start) by `moe_f8_rebase` (kernels/moe.cu) after the store's copies. Packed on the GPU (`F8Mat::pack`) some experts at a time and read back: a patch is packed on its own, so an
/// expert's bytes are the same as in the stack's one pack, but for the bases.
pub struct F8Places {
    pub h: bool,
    pub experts: usize,
    pub rows: usize,
    pub k: usize,
    pub patches: usize,
    pub var_words: usize,
    pub core: Vec<u8>,
    pub heads: Vec<u8>,
    pub var: Vec<u8>,
    pub scale: Vec<u8>,
}

impl F8Places {
    /// `experts` matrices [rows, k] (rows a multiple of 128), `chunk(a, b)` giving experts a..b's E4M3 bytes and fp32 scales end to end (as `F8Mat::pack` takes the stack [(b - a) rows, k]), packed on `dev` some at a time
    /// (`check`: as `F8Mat::pack`'s) and read back.
    #[allow(clippy::type_complexity)]
    pub fn pack(
        dev: &Dev,
        lib: &Lib,
        experts: usize,
        rows: usize,
        k: usize,
        chunk: &mut dyn FnMut(usize, usize) -> Result<(Vec<u8>, Vec<f32>)>,
        check: bool,
    ) -> Result<F8Places> {
        if experts == 0 || rows == 0 || !rows.is_multiple_of(128) || k == 0 {
            return err(format!("{experts} FP8 experts of {rows} x {k}: the places take experts of rows a multiple of 128"));
        }
        let per = (PACK_BYTES / (rows * k)).clamp(1, experts);
        let mut h = false;
        let chunks = (0..experts).step_by(per).map(|a| {
            let b = (a + per).min(experts);
            let (q, sc) = chunk(a, b)?;
            let m = F8Mat::pack(dev, lib, &q, &sc, (b - a) * rows, k, check)?;
            h = m.h;
            m.read_back()
        });
        let mut p = F8Places::of_chunks(false, experts, rows, k, chunks)?;
        p.h = h;
        Ok(p)
    }

    /// The places' bytes from the packs of the experts some at a time, in order (each a stack's core, heads, var words and scales, as `F8Mat::read_back` gives them): each expert's parts cut out, its heads' bases made
    /// its own var's, its var padded to `var_words`.
    #[allow(clippy::type_complexity)]
    fn of_chunks(
        h: bool,
        experts: usize,
        rows: usize,
        k: usize,
        chunks: impl Iterator<Item = Result<(Vec<u8>, Vec<u32>, Vec<u32>, Vec<u8>)>>,
    ) -> Result<F8Places> {
        let kt = k.div_ceil(128);
        let (patches, scales) = (rows / 64 * kt, rows / 128 * kt * 4);
        let (mut core, mut heads, mut scale, mut parts) = (Vec::new(), Vec::new(), Vec::new(), Vec::with_capacity(experts));
        for got in chunks {
            let (c, hd, v, s) = got?;
            let (a, n) = (parts.len(), hd.len() / (patches * 8).max(1));
            let cb = if n == 0 { 0 } else { c.len() / (n * patches) };
            if n == 0 || hd.len() != n * patches * 8 || !matches!(cb, 6144 | 7168) || c.len() != n * patches * cb || s.len() != n * scales || a + n > experts {
                return err(format!("experts from {a} of {rows} x {k} in FP8: {} + {} + {} bytes, not their patches'", c.len(), hd.len() * 4, s.len()));
            }
            core.extend_from_slice(&c);
            scale.extend_from_slice(&s);
            for j in 0..n {
                let mut hs = hd[j * patches * 8..(j + 1) * patches * 8].to_vec();
                let (start, end) = (hs[0] as usize, if j + 1 < n { hd[(j + 1) * patches * 8] as usize } else { v.len() });
                if start > end || end > v.len() || hs.chunks(8).any(|x| (x[0] as usize) < start || x[0] as usize > end) {
                    return err(format!("expert {} of {rows} x {k} in FP8: its var parts are not its own words {start}..{end}", a + j));
                }
                hs.chunks_mut(8).for_each(|x| x[0] -= start as u32);
                heads.extend(hs.iter().flat_map(|w| w.to_le_bytes()));
                parts.push(v[start..end].to_vec());
            }
        }
        if parts.len() != experts {
            return err(format!("{} of {experts} FP8 experts packed", parts.len()));
        }
        // (more than any expert's words, so that a base modulo them is the expert's own)
        let var_words = (parts.iter().map(Vec::len).max().unwrap_or(0) + 8).next_multiple_of(8);
        let mut var = vec![0u8; experts * var_words * 4];
        for (e, p) in parts.iter().enumerate() {
            var[e * var_words * 4..][..p.len() * 4].copy_from_slice(as_bytes(p));
        }
        Ok(F8Places { h, experts, rows, k, patches, var_words, core, heads, var, scale })
    }

    /// An expert's bytes in each part: the core, the var, the heads, the scales.
    pub fn strides(&self) -> [u64; 4] {
        let e = self.experts as u64;
        [self.core.len() as u64 / e, self.var.len() as u64 / e, self.heads.len() as u64 / e, self.scale.len() as u64 / e]
    }

    /// `slots` places on the GPU for the experts: the matrix the experts' product runs over, the store copying an expert into each (`packs`). The heads and the var are zeros until then.
    pub fn places(&self, slots: usize) -> Result<F8Mat> {
        let [c, v, hd, s] = self.strides().map(|x| x as usize);
        let (core, heads, var, scale) = (Buf::new(slots * c)?, Buf::zeroed(slots * hd)?, Buf::zeroed(slots * v + 32)?, Buf::new(slots * s)?);
        let bytes = core.bytes + heads.bytes + var.bytes + scale.bytes;
        let place = Some(F8Place { rows: self.rows, patches: self.patches, var_words: self.var_words });
        Ok(F8Mat { h: self.h, core, heads, var, scale, n: slots * self.rows, k: self.k, bytes, tp_rows: 0, place })
    }

    /// The two packs of the store for these experts and their places `m` (`places`): the core and the var, the heads and the scales.
    pub fn packs<'a>(&'a self, m: &F8Mat) -> [crate::offload::PackIn<'a>; 2] {
        let [c, v, hd, s] = self.strides();
        [
            crate::offload::PackIn { bytes: [&self.core, &self.var], stride: [c, v], dev: [m.core.ptr, m.var.ptr] },
            crate::offload::PackIn { bytes: [&self.heads, &self.scale], stride: [hd, s], dev: [m.heads.ptr, m.scale.ptr] },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Experts packed two at a time (each pack's var parts end to end from 0, as a stack's): cut into their own bytes, then copied into places in another order and their heads made the place's as `moe_f8_rebase` does, each
    /// head's base points at the same words in the places' var as in the expert's pack, and the cores and scales are the expert's.
    #[test]
    fn an_experts_heads_point_at_its_own_words_in_any_place() {
        let (experts, rows, k) = (5usize, 128usize, 256usize);
        let (patches, scales) = (rows / 64 * 2, rows / 128 * 2 * 4);
        // (expert x's patch p: var part of 3 x + p % 4 words, each word x << 16 | p << 8 | w; its core bytes x, its scales x + 1)
        let words = |x: usize, p: usize| 3 * x + p % 4;
        let mut want = Vec::new();
        let chunks: Vec<Result<(Vec<u8>, Vec<u32>, Vec<u32>, Vec<u8>)>> = (0..experts)
            .step_by(2)
            .map(|a| {
                let b = (a + 2).min(experts);
                let (mut c, mut hd, mut v, mut s) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for x in a..b {
                    c.extend(std::iter::repeat_n(x as u8, patches * 6144));
                    s.extend(std::iter::repeat_n(x as u8 + 1, scales));
                    for p in 0..patches {
                        hd.extend([v.len() as u32, 7, 8, 9, 10, 11, 12, 13]);
                        let part: Vec<u32> = (0..words(x, p)).map(|w| (x << 16 | p << 8 | w) as u32).collect();
                        want.push(part.clone());
                        v.extend(part);
                    }
                }
                v.extend([0u32; 8]);
                Ok((c, hd, v, s))
            })
            .collect();
        let pl = F8Places::of_chunks(false, experts, rows, k, chunks.into_iter()).unwrap();
        let [cs, vs, hs, ss] = pl.strides().map(|x| x as usize);
        assert_eq!((cs, hs, ss), (patches * 6144, patches * 32, scales));
        assert!(pl.var_words > (0..patches).map(|p| words(experts - 1, p)).sum::<usize>());
        // (three places: experts 4, 1, 3 copied in, as the store copies them, then every place's heads set, twice: a set head stays)
        let slots = 3;
        let (mut core, mut heads, mut var, mut scale) = (vec![0u8; slots * cs], vec![0u8; slots * hs], vec![0u8; slots * vs + 32], vec![0u8; slots * ss]);
        for (place, x) in [(0, 4usize), (1, 1), (2, 3)] {
            core[place * cs..][..cs].copy_from_slice(&pl.core[x * cs..][..cs]);
            var[place * vs..][..vs].copy_from_slice(&pl.var[x * vs..][..vs]);
            heads[place * hs..][..hs].copy_from_slice(&pl.heads[x * hs..][..hs]);
            scale[place * ss..][..ss].copy_from_slice(&pl.scale[x * ss..][..ss]);
        }
        for _ in 0..2 {
            for i in 0..slots * patches {
                let b = u32::from_le_bytes(heads[i * 32..][..4].try_into().unwrap()) as usize;
                heads[i * 32..][..4].copy_from_slice(&((b % pl.var_words + i / patches * pl.var_words) as u32).to_le_bytes());
            }
        }
        let word = |i: usize| u32::from_le_bytes(var[i * 4..][..4].try_into().unwrap());
        for (place, x) in [(0, 4usize), (1, 1), (2, 3)] {
            assert!(core[place * cs..][..cs].iter().all(|&b| b == x as u8) && scale[place * ss..][..ss].iter().all(|&b| b == x as u8 + 1));
            for p in 0..patches {
                let h = &heads[(place * patches + p) * 32..][..32];
                let base = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
                assert_eq!(u32::from_le_bytes(h[4..8].try_into().unwrap()), 7, "the rest of a head as packed");
                let got: Vec<u32> = (0..words(x, p)).map(|w| word(base + w)).collect();
                assert_eq!(got, want[x * patches + p], "expert {x}'s patch {p} in place {place}");
            }
        }
        // (a pack of other patches than its experts' is refused)
        let short = vec![Ok((vec![0u8; 6144], vec![0u32; 8], vec![0u32; 8], vec![0u8; scales]))];
        assert!(F8Places::of_chunks(false, 1, rows, k, short.into_iter()).is_err());
    }
}
