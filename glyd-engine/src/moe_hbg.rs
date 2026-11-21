//! Hummingbird's routed experts on the GPU for every mixture of experts (docs/design/hummingbird.md, #429): `HbgExperts`, a layer's experts as codebook rows (glyd-gpu's codebook.rs: codes of 2 or 3 bits, a bf16 scale for each run of 32 values,
//! a bf16 codebook of the row's own) in the basis of one rotation R = (I x H_b) diag(signs) of the layer, for any width W (the experts' input: the hidden size, or the latent where the MoE has one) in runs of 32 and a rotation block b
//! (a power of two from 32 to 1,024 that divides W; src/hb_codec.rs makes the rows). An expert is `gate_up` [U, W] (U = 2 I for a gated MLP, the gate's rows first; I otherwise) and `down_t` [I, W], the rows R w_n and R d_n. The layer's
//! experts are in two groups, one for each bit width (group 0: 2 bits, 1: 3 bits), each a stack of its experts end to end, an expert's two matrices at the same stride in every place of the group (`HbgPlace`, as moe_hb.rs's `HbPlace`
//! with a row count for each matrix); two small tables, `group_of` [E] (u8) and `local_of` [E] (u16), say an expert's group and place.
//!
//! A layer's products (kernels/moe_hbg.cu, `HbgKern`): x' = R x once a token (`hbg_rot_in`); for each pair of the plan (the library's `moe_way` over the router's ids: the pairs sorted by expert), the first product over the pair's
//! expert's `gate_up` rows into the [pairs, U] rows in the plan's order that the library's first product writes, rounded to bf16 (`hbg_up2`, `hbg_up3`: <R w, R x> = <w, x>, so each family's own activation kernel runs on them as it
//! does); the second product over the `down_t` rows (`hbg_down2`, `hbg_down3`: the sum over the I neurons of the activation times the row, dense, fp32, in the rotated basis) into the parts [pairs, W]; then, where the family's sum is linear
//! in the experts' outputs, the weighted sum of a token's parts and R^T once, one rounding to bf16 (`hbg_sum_rot`, with the host's `extra` sums where the expert cache's host computed some pairs), or, where it is not (Muse Spark
//! norms each expert's output before weighing it), R^T of each pair (`hbg_unrot`). `MoeLayer::new_hb` (src/moe.rs) holds an `HbgExperts` as the layer's routed experts and runs these in place of the library's grouped products.
//! No atomics: every launch gives the same bits every time, and an expert gives the same bits wherever it sits in memory.
//!
//! The host's twin is `HbgHost` (the same bytes, in vectors), whose experts the CPU computes (src/cpu_hbg.rs) with the same roundings, for the check of the kernels and for the pairs of the experts the GPU has no place for.

use crate::cpu_hb::CbView;
use crate::dev::{Buf, Dev, Kernel};
use crate::ffi::{err, Error, Result, Stream};
use crate::hb_codec::{LayerRows, MAX_BLOCK, MIN_BLOCK, RUN};
use crate::launch;
use crate::moe::MoeCfg;
use crate::weights::upload_bytes;
use glyd_gpu::codebook::{cb_strides, CbRows};

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// Neurons a block of `hbg_up2` and `hbg_up3` takes, outputs a block of `hbg_down2` and `hbg_down3` takes (kernels/moe_hbg.cu defines the same: a test holds them equal).
pub const UP_ROWS: usize = 64;
pub const DOWN_OUT: usize = 128;
/// Floats a run of 32 values of the rotated input takes in the first product's shared memory, and the widest row it takes (8,192: 36,864 bytes).
pub const XPAD: usize = 36;
pub const MAX_WIDTH: usize = 8192;
/// Experts a layer has at most (the router's: `moe::MAX_EXPERTS`).
pub const MAX_EXPERTS: usize = 1024;

/// A bit width's group: 0 for two bits, 1 for three.
pub fn group_index(bits: u32) -> usize {
    bits as usize - 2
}

/// The factor H_b's normalisation is applied with, as `hb_codec::rotate` computes it in fp32 (so that the kernels' bits are the codec's).
pub fn scale_of(b: usize) -> f32 {
    1.0 / (b as f32).sqrt()
}

/// Whether rows of `width` values rotated in blocks of `rot` are ones the kernels and the host take: a width in runs of 32 up to `MAX_WIDTH`, the block a power of two from 32 to 1,024 that divides it.
pub fn check_rot(width: usize, rot: usize) -> Result<()> {
    if width == 0
        || !width.is_multiple_of(RUN)
        || width > MAX_WIDTH
        || !rot.is_power_of_two()
        || !(MIN_BLOCK..=MAX_BLOCK).contains(&rot)
        || !width.is_multiple_of(rot)
    {
        return err(format!(
            "rows of {width} values in rotation blocks of {rot}: a width in runs of {RUN} up to {MAX_WIDTH}, a block a power of two from {MIN_BLOCK} to {MAX_BLOCK} that divides it"
        ));
    }
    Ok(())
}

fn pad16(n: usize) -> usize {
    n.next_multiple_of(16)
}

/// The shape of a layer's experts, apart from their bytes: what `MoeLayer::new_hb` holds the layer's config to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HbgShape {
    pub experts: usize,
    pub width: usize,
    pub rot: usize,
    pub inter: usize,
    pub gated: bool,
}

impl HbgShape {
    /// Rows of an expert's first matrix: the gate's and the up's, or the up's.
    pub fn up_rows(&self) -> usize {
        if self.gated {
            2 * self.inter
        } else {
            self.inter
        }
    }

    /// Whether the experts are the layer `cfg`'s: its number, the width the experts work in (`hidden`: the latent where there is one), the intermediate size and a gate where the activation has one.
    pub fn check(&self, cfg: &MoeCfg) -> Result<()> {
        check_rot(self.width, self.rot)?;
        if self.experts == 0 || self.experts > MAX_EXPERTS || self.inter == 0 || !self.inter.is_multiple_of(8) {
            return err(format!("{} experts of {} neurons: 1 to {MAX_EXPERTS} experts, the neurons in runs of 8", self.experts, self.inter));
        }
        if (self.experts, self.width, self.inter, self.gated) != (cfg.experts, cfg.hidden, cfg.inter, cfg.act.gated()) {
            return err(format!(
                "hummingbird experts: {} of width {} and {} neurons ({}), where the layer has {} of width {} and {} neurons ({})",
                self.experts,
                self.width,
                self.inter,
                if self.gated { "gated" } else { "not gated" },
                cfg.experts,
                cfg.hidden,
                cfg.inter,
                if cfg.act.gated() { "gated" } else { "not gated" }
            ));
        }
        Ok(())
    }
}

/// Where the parts of an expert of `bits` bits are in a place (an expert's bytes in a group's stack): `data` holds the first matrix's codes and then down_t's, `meta` the first matrix's scales and codebook and then down_t's, each part to 16
/// bytes. The first matrix is `up_rows` neuron rows and down_t `down_rows` of `width` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HbgPlace {
    pub bits: u32,
    pub width: usize,
    pub up_rows: usize,
    pub down_rows: usize,
    /// bytes of `data` and of `meta` an expert takes
    pub data: usize,
    pub meta: usize,
    /// where down_t's codes start in `data`; where the first matrix's codebook, down_t's scales and down_t's codebook start in `meta`
    pub down_codes: usize,
    pub up_book: usize,
    pub down_scales: usize,
    pub down_book: usize,
}

impl HbgPlace {
    pub fn new(bits: u32, width: usize, up_rows: usize, down_rows: usize) -> Result<HbgPlace> {
        if !(2..=3).contains(&bits) || up_rows == 0 || down_rows == 0 || width == 0 || !width.is_multiple_of(RUN) {
            return err(format!(
                "an expert of {up_rows} and {down_rows} neuron rows of {width} values at {bits} bits: 2 or 3 bits and rows in runs of {RUN} are wanted"
            ));
        }
        let (c, s, b) = cb_strides(bits, width);
        let (uc, us, ub) = (pad16(up_rows * c), pad16(up_rows * s), pad16(up_rows * b));
        let (dc, ds, db) = (pad16(down_rows * c), pad16(down_rows * s), pad16(down_rows * b));
        Ok(HbgPlace {
            bits,
            width,
            up_rows,
            down_rows,
            data: uc + dc,
            meta: us + ub + ds + db,
            down_codes: uc,
            up_book: us,
            down_scales: us + ub,
            down_book: us + ub + ds,
        })
    }

    /// The bytes an expert takes in a stack (codes, scales and codebooks of its two matrices, the parts padded to 16 bytes).
    pub fn bytes(&self) -> u64 {
        (self.data + self.meta) as u64
    }

    /// An expert's two matrices as the CPU reads them, over the bytes of its place as they are (`data` and `meta`: its place's bytes, in a stack's host copy or the host tier: nothing is copied).
    pub fn views<'a>(&self, data: &'a [u8], meta: &'a [u8]) -> Result<HbgExpert<'a>> {
        if data.len() != self.data || meta.len() != self.meta {
            return err(format!(
                "{} + {} bytes, where an expert of {} and {} rows of {} at {} bits takes {} + {}",
                data.len(),
                meta.len(),
                self.up_rows,
                self.down_rows,
                self.width,
                self.bits,
                self.data,
                self.meta
            ));
        }
        let (c, s, b) = cb_strides(self.bits, self.width);
        let view = |rows: usize, codes: usize, scales: usize, book: usize| CbView {
            bits: self.bits,
            rows,
            cols: self.width,
            codes: &data[codes..codes + rows * c],
            scales: &meta[scales..scales + rows * s],
            codebook: &meta[book..book + rows * b],
        };
        Ok(HbgExpert { up: view(self.up_rows, 0, 0, self.up_book), down: view(self.down_rows, self.down_codes, self.down_scales, self.down_book) })
    }

    /// An expert's two matrices written into its place: `data` and `meta` are the place's bytes.
    fn put(&self, up: &CbRows, down: &CbRows, data: &mut [u8], meta: &mut [u8]) -> Result<()> {
        for (m, rows) in [(up, self.up_rows), (down, self.down_rows)] {
            if (m.bits, m.rows, m.cols) != (self.bits, rows, self.width) || !m.is_valid() {
                return err(format!(
                    "a matrix of {} x {} at {} bits, where the group's experts have {rows} x {} at {}",
                    m.rows, m.cols, m.bits, self.width, self.bits
                ));
            }
        }
        data[..up.codes.len()].copy_from_slice(&up.codes);
        data[self.down_codes..self.down_codes + down.codes.len()].copy_from_slice(&down.codes);
        meta[..up.scales.len()].copy_from_slice(&up.scales);
        meta[self.up_book..self.up_book + up.codebook.len()].copy_from_slice(&up.codebook);
        meta[self.down_scales..self.down_scales + down.scales.len()].copy_from_slice(&down.scales);
        meta[self.down_book..self.down_book + down.codebook.len()].copy_from_slice(&down.codebook);
        Ok(())
    }
}

/// An expert's two matrices, borrowed: `up` is the first (gate_up) [U, W], `down` is down_t [I, W].
#[derive(Clone, Copy, Debug)]
pub struct HbgExpert<'a> {
    pub up: CbView<'a>,
    pub down: CbView<'a>,
}

/// A group's experts end to end on the host (expert k's place at k times the place's `data` and `meta`).
pub struct HbgGroup {
    pub place: HbgPlace,
    pub experts: usize,
    pub data: Vec<u8>,
    pub meta: Vec<u8>,
}

/// A layer's experts on the host: what `HbgExperts` uploads and the CPU reads (src/cpu_hbg.rs): the two groups (index 0 the two-bit experts, 1 the three-bit ones; None where the layer has none of that width), each expert's group and its
/// number there, and the signs of the rotation.
pub struct HbgHost {
    pub shape: HbgShape,
    pub signs: Vec<i8>,
    pub groups: [Option<HbgGroup>; 2],
    pub group_of: Vec<u8>,
    pub local_of: Vec<u16>,
}

impl HbgHost {
    /// The layer's experts (`hb_codec`'s rows, each in the router's order) joined by bit width: every expert's matrices held to the layer's shape and to its own bits, the signs to +1 and -1, the rotation to a block the kernels take.
    pub fn of(rows: &LayerRows) -> Result<HbgHost> {
        let shape = HbgShape { experts: rows.experts.len(), width: rows.width, rot: rows.rot, inter: rows.inter, gated: rows.gated };
        check_rot(shape.width, shape.rot)?;
        if shape.experts == 0 || shape.experts > MAX_EXPERTS || shape.inter == 0 || !shape.inter.is_multiple_of(8) {
            return err(format!("{} experts of {} neurons: 1 to {MAX_EXPERTS} experts, the neurons in runs of 8", shape.experts, shape.inter));
        }
        if rows.signs.len() != rows.width || rows.signs.iter().any(|&s| s != 1 && s != -1) {
            return err(format!("{} signs for rows of {}: one +1 or -1 for each value", rows.signs.len(), rows.width));
        }
        let places = [HbgPlace::new(2, shape.width, shape.up_rows(), shape.inter)?, HbgPlace::new(3, shape.width, shape.up_rows(), shape.inter)?];
        let mut members: [Vec<&crate::hb_codec::ExpertRows>; 2] = [Vec::new(), Vec::new()];
        let (mut group_of, mut local_of) = (Vec::with_capacity(shape.experts), Vec::with_capacity(shape.experts));
        for (e, x) in rows.experts.iter().enumerate() {
            if !(2..=3).contains(&x.bits) {
                return err(format!("expert {e}: {} bits, where hummingbird's are 2 or 3", x.bits));
            }
            let g = group_index(x.bits);
            for (m, r) in [(&x.gate_up, shape.up_rows()), (&x.down_t, shape.inter)] {
                if (m.bits, m.rows, m.cols) != (x.bits, r, shape.width) || !m.is_valid() {
                    return err(format!(
                        "expert {e}: a matrix of {} x {} at {} bits, where the layer's are {r} x {} at {}",
                        m.rows, m.cols, m.bits, shape.width, x.bits
                    ));
                }
            }
            group_of.push(g as u8);
            local_of.push(u16::try_from(members[g].len()).map_err(|_| Error("more experts in a group than a place number holds".into()))?);
            members[g].push(x);
        }
        let mut groups = [None, None];
        for g in 0..2 {
            if members[g].is_empty() {
                continue;
            }
            let place = places[g];
            let (mut data, mut meta) = (vec![0u8; members[g].len() * place.data], vec![0u8; members[g].len() * place.meta]);
            for (k, x) in members[g].iter().enumerate() {
                place.put(&x.gate_up, &x.down_t, &mut data[k * place.data..(k + 1) * place.data], &mut meta[k * place.meta..(k + 1) * place.meta])?;
            }
            groups[g] = Some(HbgGroup { place, experts: members[g].len(), data, meta });
        }
        Ok(HbgHost { shape, signs: rows.signs.clone(), groups, group_of, local_of })
    }

    /// Expert `e`'s two matrices, over the group's bytes in place.
    pub fn expert(&self, e: usize) -> Result<HbgExpert<'_>> {
        let (Some(&g), Some(&k)) = (self.group_of.get(e), self.local_of.get(e)) else { return err(format!("expert {e} of {}", self.shape.experts)) };
        let grp = self.groups[g as usize].as_ref().ok_or_else(|| Error(format!("no group {g} in the layer")))?;
        let k = k as usize;
        grp.place.views(&grp.data[k * grp.place.data..(k + 1) * grp.place.data], &grp.meta[k * grp.place.meta..(k + 1) * grp.place.meta])
    }

    /// The bytes of the experts (the stacks'), as the planner counts them.
    pub fn bytes(&self) -> u64 {
        self.groups.iter().flatten().map(|g| g.experts as u64 * g.place.bytes()).sum()
    }
}

/// The three arrays of one matrix of a stack, as the kernels take them: the addresses of the first place's, and the bytes between places.
#[derive(Clone, Copy, Debug)]
pub struct HbgArrays {
    pub codes: u64,
    pub scales: u64,
    pub book: u64,
    pub codes_stride: i64,
    pub scales_stride: i64,
    pub book_stride: i64,
}

/// A group's experts (one bit width) in device memory: the codes in `data`, the scales and codebooks in `meta`.
pub struct HbgStack {
    pub place: HbgPlace,
    pub experts: usize,
    pub bytes: usize,
    data: Buf,
    meta: Buf,
}

impl HbgStack {
    /// The group's experts copied to the GPU.
    pub fn upload(g: &HbgGroup) -> Result<HbgStack> {
        let (data, meta) = (upload_bytes(&g.data)?, upload_bytes(&g.meta)?);
        Ok(HbgStack { place: g.place, experts: g.experts, bytes: data.bytes + meta.bytes, data, meta })
    }

    pub fn up_arrays(&self) -> HbgArrays {
        let p = &self.place;
        HbgArrays {
            codes: self.data.ptr,
            scales: self.meta.ptr,
            book: self.meta.ptr + p.up_book as u64,
            codes_stride: p.data as i64,
            scales_stride: p.meta as i64,
            book_stride: p.meta as i64,
        }
    }

    pub fn down_arrays(&self) -> HbgArrays {
        let p = &self.place;
        HbgArrays {
            codes: self.data.ptr + p.down_codes as u64,
            scales: self.meta.ptr + p.down_scales as u64,
            book: self.meta.ptr + p.down_book as u64,
            codes_stride: p.data as i64,
            scales_stride: p.meta as i64,
            book_stride: p.meta as i64,
        }
    }
}

/// A MoE layer's routed experts on the GPU in the hummingbird format: the two groups' stacks, the tables that send a pair to its group and place, the signs of the rotation, the shape.
pub struct HbgExperts {
    pub shape: HbgShape,
    groups: [Option<HbgStack>; 2],
    group_of: Buf,
    local_of: Buf,
    signs: Buf,
}

impl HbgExperts {
    /// A layer's experts (`hb_codec`'s rows, as the packer's reader gives them) copied to the GPU.
    pub fn upload(rows: &LayerRows) -> Result<HbgExperts> {
        HbgExperts::from_host(&HbgHost::of(rows)?)
    }

    /// A layer's experts on the host copied to the GPU (the host keeps its own for the CPU's products).
    pub fn from_host(h: &HbgHost) -> Result<HbgExperts> {
        let mut groups = [None, None];
        for (g, grp) in h.groups.iter().enumerate() {
            if let Some(grp) = grp {
                groups[g] = Some(HbgStack::upload(grp)?);
            }
        }
        Ok(HbgExperts {
            shape: h.shape,
            groups,
            group_of: upload_bytes(&h.group_of)?,
            local_of: upload_bytes(&h.local_of.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?,
            signs: upload_bytes(&h.signs.iter().map(|&s| s as u8).collect::<Vec<u8>>())?,
        })
    }

    /// Device bytes of the experts: the stacks, the tables and the signs.
    pub fn bytes(&self) -> usize {
        self.groups.iter().flatten().map(|g| g.bytes).sum::<usize>() + self.group_of.bytes + self.local_of.bytes + self.signs.bytes
    }

    /// Whether the layer has experts of group `g` (0: two bits, 1: three).
    pub fn has_group(&self, g: usize) -> bool {
        self.groups.get(g).is_some_and(Option::is_some)
    }

    pub fn group(&self, g: usize) -> Option<&HbgStack> {
        self.groups.get(g).and_then(Option::as_ref)
    }

    /// A value of the layer's shape with no bytes on the device (host tests of the layer's checks).
    #[cfg(test)]
    pub(crate) fn without_device(h: &HbgHost) -> HbgExperts {
        let none = || Buf { ptr: 0, bytes: 0 };
        HbgExperts {
            shape: h.shape,
            groups: h.groups.each_ref().map(|g| g.as_ref().map(|g| HbgStack { place: g.place, experts: g.experts, bytes: 0, data: none(), meta: none() })),
            group_of: none(),
            local_of: none(),
            signs: none(),
        }
    }
}

/// The bytes of the scratch a layer's products work in, for `tokens` tokens of `k` choices each: the rotated inputs (fp32), the parts of the second product (fp32, a row a pair), and the pairs' unrotated rows (bf16).
pub fn xr_bytes(tokens: usize, width: usize) -> usize {
    tokens * width * 4
}
pub fn part_bytes(pairs: usize, width: usize) -> usize {
    pairs * width * 4
}
pub fn rows_bytes(pairs: usize, width: usize) -> usize {
    pairs * width * 2
}

/// The block of a rotation kernel: threads of 4 values each, at least a warp (a run of b < 128 values is part of one), and the blocks a row takes.
fn rot_grid(width: usize, rot: usize) -> (u32, u32) {
    let threads = (rot / 4).max(32);
    (threads as u32, width.div_ceil(4 * threads) as u32)
}

/// The Hummingbird layer's kernels (kernels/moe_hbg.cu) with typed launches. Device addresses are `u64`.
pub struct HbgKern {
    rot: Kernel,
    up: [Kernel; 2],
    down: [Kernel; 2],
    inv: Kernel,
    sum: Kernel,
    unrot: Kernel,
}

impl HbgKern {
    pub fn load(dev: &Dev) -> Result<HbgKern> {
        Ok(HbgKern {
            rot: dev.kernel(kname!("hbg_rot_in"))?,
            up: [dev.kernel(kname!("hbg_up2"))?, dev.kernel(kname!("hbg_up3"))?],
            down: [dev.kernel(kname!("hbg_down2"))?, dev.kernel(kname!("hbg_down3"))?],
            inv: dev.kernel(kname!("hbg_inv"))?,
            sum: dev.kernel(kname!("hbg_sum_rot"))?,
            unrot: dev.kernel(kname!("hbg_unrot"))?,
        })
    }

    /// xr [tokens, W] (fp32) = R x for x [tokens, W] bf16 (rows `xs` elements apart): the layer's signs applied, then the Hadamard transform of each run of b values over sqrt b.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernel reads and writes (kernels/moe_hbg.cu).
    pub unsafe fn rot_in(&self, s: Stream, hb: &HbgExperts, x: u64, xs: usize, xr: u64, tokens: usize) -> Result<()> {
        let (w, b) = (hb.shape.width, hb.shape.rot);
        if xs < w {
            return err(format!("rows of {w} values {xs} apart"));
        }
        if tokens == 0 {
            return Ok(());
        }
        let (threads, blocks) = rot_grid(w, b);
        launch!(self.rot, [tokens as u32, blocks, 1], [threads, 1, 1], s; x, xs as i32, hb.signs.ptr, xr, w as i32, b as i32, scale_of(b))
    }

    /// The first product of group `g`'s experts (0: two bits, 1: three; nothing where the layer has none): gu [pairs in the plan, U] (bf16) from xr, for the pairs of `plan` (`Lib::moe_way`'s over `ids`, int64 [pairs] on the device, `k`
    /// choices a token) whose experts are in the group: the rows of the plan's order, as the library's first product writes them.
    ///
    /// # Safety
    /// As `rot_in`; `gu` has room for `pairs` rows of U values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn up(&self, s: Stream, g: usize, hb: &HbgExperts, xr: u64, plan: u64, ids: u64, gu: u64, pairs: usize, k: usize) -> Result<()> {
        if g > 1 {
            return err(format!("group {g}: 0 (two bits) or 1 (three)"));
        }
        let Some(st) = hb.group(g) else { return Ok(()) };
        if k == 0 {
            return err("0 choices a token");
        }
        if pairs == 0 {
            return Ok(());
        }
        let (w, rows, m) = (hb.shape.width, hb.shape.up_rows(), st.up_arrays());
        let smem = (w / RUN * XPAD * 4) as u32;
        launch!(self.up[g], [pairs as u32, rows.div_ceil(UP_ROWS) as u32, 1], [256, 1, 1], smem smem, s;
            xr, plan, ids, hb.group_of.ptr, hb.local_of.ptr, m.codes, m.scales, m.book, m.codes_stride, m.scales_stride, m.book_stride, gu, hb.shape.experts as i32, rows as i32, w as i32, k as i32)
    }

    /// The second product of group `g`'s experts: part [pairs in the plan, W] (fp32, the rotated basis) from `h` (the activation's rows [pairs, I] bf16, the plan's order), the sum over the neurons of each activation times the neuron's down_t row.
    ///
    /// # Safety
    /// As `up`; `part` has room for `pairs` rows of W floats.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn down(&self, s: Stream, g: usize, hb: &HbgExperts, h: u64, plan: u64, ids: u64, part: u64, pairs: usize) -> Result<()> {
        if g > 1 {
            return err(format!("group {g}: 0 (two bits) or 1 (three)"));
        }
        let Some(st) = hb.group(g) else { return Ok(()) };
        if pairs == 0 {
            return Ok(());
        }
        let (w, m) = (hb.shape.width, st.down_arrays());
        launch!(self.down[g], [pairs as u32, w.div_ceil(DOWN_OUT) as u32, 1], [256, 1, 1], s;
            h, plan, ids, hb.group_of.ptr, hb.local_of.ptr, m.codes, m.scales, m.book, m.codes_stride, m.scales_stride, m.book_stride, part, hb.shape.experts as i32, hb.shape.inter as i32, w as i32)
    }

    /// inv [pairs] (int32) = each pair's row in the plan, for the pairs in it (the others' entries are not written: `sum_rot` skips them by their ids).
    ///
    /// # Safety
    /// As `rot_in`.
    pub unsafe fn inverse(&self, s: Stream, hb: &HbgExperts, plan: u64, inv: u64, pairs: usize) -> Result<()> {
        if pairs == 0 {
            return Ok(());
        }
        launch!(self.inv, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; plan, inv, pairs as i32, hb.shape.experts as i32)
    }

    /// y [tokens, W] (bf16) = R^T of the weighted sum of each token's pairs' parts (`wts` fp32 [pairs]; `ids` the choices, int64 [pairs], an expert below E counting; `inv` from `inverse`) and of `extra` (fp32 [tokens, W] in the rotated basis,
    /// the host's weighted sums of the pairs it computed: 0, none): signs o (H v) over sqrt b on each run of b values, rounded to bf16 once.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sum_rot(
        &self,
        s: Stream,
        hb: &HbgExperts,
        part: u64,
        inv: u64,
        wts: u64,
        ids: u64,
        y: u64,
        extra: u64,
        tokens: usize,
        k: usize,
    ) -> Result<()> {
        let (w, b) = (hb.shape.width, hb.shape.rot);
        if k == 0 {
            return err("0 choices a token");
        }
        if tokens == 0 {
            return Ok(());
        }
        let (threads, blocks) = rot_grid(w, b);
        launch!(self.sum, [tokens as u32, blocks, 1], [threads, 1, 1], s; part, inv, wts, ids, hb.signs.ptr, y, extra, hb.shape.experts as i32, k as i32, w as i32, b as i32, scale_of(b))
    }

    /// rows [pairs in the plan, W] (bf16) = R^T of each pair's part, rounded to bf16 once: each expert's output in the basis of the model, for a family that norms it before it sums.
    ///
    /// # Safety
    /// As `rot_in`; `rows` has room for `pairs` rows of W values.
    pub unsafe fn unrotate(&self, s: Stream, hb: &HbgExperts, part: u64, plan: u64, rows: u64, pairs: usize) -> Result<()> {
        let (w, b) = (hb.shape.width, hb.shape.rot);
        if pairs == 0 {
            return Ok(());
        }
        let (threads, blocks) = rot_grid(w, b);
        launch!(self.unrot, [pairs as u32, blocks, 1], [threads, 1, 1], s; part, plan, hb.signs.ptr, rows, hb.shape.experts as i32, w as i32, b as i32, scale_of(b))
    }
}

/// Test helpers shared with src/cpu_hbg.rs's tests: a random generator and layers of random or coded experts.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::hb_codec::{bf16, block_of, code_expert, f32_of, signs, ExpertRows};
    use glyd_gpu::codebook::pack_codebook_rows;

    pub struct Rng(pub u64);

    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        pub fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        pub fn normal(&mut self) -> f32 {
            let (a, b) = (self.unit().max(1e-7), self.unit());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }

        /// A normal value of scale `s`, rounded to bf16.
        pub fn bf(&mut self, s: f32) -> f32 {
            f32_of(bf16(self.normal() * s))
        }
    }

    /// A matrix of random codes, scales (some negative) and codebooks: whatever the layout holds.
    pub fn random_matrix(rng: &mut Rng, bits: u32, rows: usize, cols: usize) -> CbRows {
        let codes: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
        let scales: Vec<u16> = (0..rows * cols / 32).map(|_| bf16((rng.unit() - 0.3) * 0.05)).collect();
        let book: Vec<u16> = (0..rows << bits).map(|_| bf16(rng.unit() * 2.0 - 1.0)).collect();
        pack_codebook_rows(&codes, &scales, &book, bits, rows, cols).unwrap()
    }

    /// A layer of experts of random rows, `bits` each (the experts' widths), in the basis of the layer's signs (layer number `block`) and rotation block (`block_of` the width, or `rot`).
    pub fn random_layer(rng: &mut Rng, width: usize, rot: usize, inter: usize, gated: bool, bits: &[u32], block: usize) -> LayerRows {
        let u = if gated { 2 * inter } else { inter };
        LayerRows {
            block,
            width,
            rot,
            inter,
            gated,
            signs: signs(block, width),
            experts: bits
                .iter()
                .map(|&b| ExpertRows { bits: b, gate_up: random_matrix(rng, b, u, width), down_t: random_matrix(rng, b, inter, width) })
                .collect(),
        }
    }

    /// A layer of experts coded by the codec (src/hb_codec.rs: rotated, rounded to nearest) from random release weights, `bits` each, with the release's matrices (f32, bf16 values: gate_up [U, W], down [W, I]) beside them.
    pub fn coded_layer(rng: &mut Rng, width: usize, inter: usize, gated: bool, bits: &[u32], block: usize) -> (LayerRows, Vec<(Vec<f32>, Vec<f32>)>) {
        let rot = block_of(width).expect("a width the codec rotates");
        let s = signs(block, width);
        let u = if gated { 2 * inter } else { inter };
        let (mut experts, mut release) = (Vec::new(), Vec::new());
        for &b in bits {
            let gate_up: Vec<f32> = (0..u * width).map(|_| rng.bf(0.05)).collect();
            let down: Vec<f32> = (0..width * inter).map(|_| rng.bf(0.05)).collect();
            let (e, _) = code_expert(&gate_up, &down, width, inter, &s, rot, b, None).unwrap();
            experts.push(e);
            release.push((gate_up, down));
        }
        (LayerRows { block, width, rot, inter, gated, signs: s, experts }, release)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::hb_codec::{block_of, rotate};
    use crate::moe::{Act, Router};

    const SRC: &str = include_str!("../kernels/moe_hbg.cu");

    /// The bytes of a place at a real shape (Kimi-K3's latent of 3,584, 2,048 neurons, gated) and at a small one with parts that are not whole 16 bytes: the parts in order and to 16 bytes, the sizes the codebook layout gives a row (3.53 and 2.51
    /// bits a weight at W = 4,096).
    #[test]
    fn a_place_and_its_bytes() {
        let (p3, p2) = (HbgPlace::new(3, 3584, 4096, 2048).unwrap(), HbgPlace::new(2, 3584, 4096, 2048).unwrap());
        // a row of 3,584 values: 1,344 + 224 + 16 bytes at three bits, 896 + 224 + 8 at two
        assert_eq!((p3.data, p3.meta), ((4096 + 2048) * 1344, (4096 + 2048) * 224 + 6144 * 16));
        assert_eq!((p2.data, p2.meta), ((4096 + 2048) * 896, (4096 + 2048) * 224 + 6144 * 8));
        assert_eq!(p3.bytes(), 6144 * (1344 + 224 + 16));
        assert_eq!(
            (p3.down_codes, p3.up_book, p3.down_scales, p3.down_book),
            (4096 * 1344, 4096 * 224, 4096 * 224 + 4096 * 16, 4096 * 224 + 4096 * 16 + 2048 * 224)
        );
        let bits_a_weight = |p: &HbgPlace| p.bytes() as f64 * 8.0 / (6144.0 * 3584.0);
        assert!((bits_a_weight(&p3) - 3.54).abs() < 0.01 && (bits_a_weight(&p2) - 2.51).abs() < 0.01, "{} {}", bits_a_weight(&p3), bits_a_weight(&p2));
        // parts not whole 16 bytes are padded: 5 rows of 64 values at 2 bits (16 bytes of codes a row: 80), scales 4 bytes a row (20 -> 32), codebook 8 bytes a row (40 -> 48), down_t 3 rows
        let q = HbgPlace::new(2, 64, 5, 3).unwrap();
        assert_eq!((q.down_codes, q.up_book, q.down_scales, q.down_book, q.data, q.meta), (80, 32, 80, 96, 80 + 48, 32 + 48 + 16 + 32));
        assert!([q.down_codes, q.up_book, q.down_scales, q.down_book, q.data, q.meta].iter().all(|x| x % 16 == 0));
        assert!(
            HbgPlace::new(4, 64, 8, 8).is_err()
                && HbgPlace::new(3, 48, 8, 8).is_err()
                && HbgPlace::new(3, 64, 0, 8).is_err()
                && HbgPlace::new(3, 64, 8, 0).is_err()
        );
    }

    /// The rotation's blocks and widths the kernels take: a block a power of two from 32 to 1,024 that divides the width, the width in runs of 32 up to 8,192; the scale is the codec's, bit for bit.
    #[test]
    fn rotations_the_kernels_take() {
        for (w, b) in [(64, 64), (96, 32), (128, 128), (384, 128), (2560, 512), (2688, 128), (3584, 512), (4096, 1024), (7168, 1024), (8192, 1024), (256, 32)] {
            check_rot(w, b).unwrap_or_else(|e| panic!("{w} {b}: {e}"));
        }
        for (w, b) in [(64, 128), (96, 64), (96, 96), (64, 16), (48, 16), (0, 32), (8224, 32), (2048, 2048), (384, 384), (3584, 1024)] {
            assert!(check_rot(w, b).is_err(), "{w} {b}");
        }
        // (hb_codec's own blocks are all taken)
        for w in [64, 96, 128, 2560, 2688, 3584, 4096, 7168] {
            check_rot(w, block_of(w).unwrap()).unwrap();
        }
        // scale_of is the codec's k: R e_0 is k everywhere
        for b in [32usize, 64, 128, 256, 512, 1024] {
            let mut x = vec![0f32; b];
            x[0] = 1.0;
            rotate(&mut x, &vec![1i8; b], b);
            assert!(x.iter().all(|v| v.to_bits() == scale_of(b).to_bits()), "{b}");
        }
        // the rotation kernels' blocks: a warp at least, 4 values a thread, a run of b < 128 part of a warp
        assert_eq!(
            (rot_grid(64, 64), rot_grid(96, 32), rot_grid(384, 128), rot_grid(3584, 512), rot_grid(7168, 1024)),
            ((32, 1), (32, 1), (32, 3), (128, 7), (256, 7))
        );
    }

    /// The experts of a layer are placed in two stacks by their bits, end to end, each at its place, and read back from the bytes where they are; the tables say each expert's group and place.
    #[test]
    fn experts_are_placed_in_two_groups_end_to_end() {
        let mut rng = Rng(0x5EED);
        for (width, rot, inter, gated) in [(256usize, 256usize, 64usize, true), (384, 128, 32, false), (64, 64, 32, true), (96, 32, 16, false)] {
            let bits = [3u32, 2, 2, 3, 2, 3, 3];
            let rows = random_layer(&mut rng, width, rot, inter, gated, &bits, 5);
            let host = HbgHost::of(&rows).unwrap();
            assert_eq!(host.shape, HbgShape { experts: 7, width, rot, inter, gated });
            assert_eq!((host.group_of.clone(), host.local_of.clone()), (vec![1, 0, 0, 1, 0, 1, 1], vec![0, 0, 1, 1, 2, 2, 3]));
            let (g2, g3) = (host.groups[0].as_ref().unwrap(), host.groups[1].as_ref().unwrap());
            assert_eq!((g2.experts, g3.experts), (3, 4));
            assert_eq!(
                (g2.data.len(), g2.meta.len(), g3.data.len(), g3.meta.len()),
                (3 * g2.place.data, 3 * g2.place.meta, 4 * g3.place.data, 4 * g3.place.meta)
            );
            assert_eq!(host.bytes(), 3 * g2.place.bytes() + 4 * g3.place.bytes());
            for (e, x) in rows.experts.iter().enumerate() {
                let v = host.expert(e).unwrap();
                for (got, want) in [(&v.up, &x.gate_up), (&v.down, &x.down_t)] {
                    assert_eq!((got.bits, got.rows, got.cols), (want.bits, want.rows, want.cols));
                    assert!(got.codes == &want.codes[..] && got.scales == &want.scales[..] && got.codebook == &want.codebook[..], "{width}: expert {e}");
                }
            }
            assert!(host.expert(7).is_err());
            // a layer of one width has one group
            let one = HbgHost::of(&random_layer(&mut rng, width, rot, inter, gated, &[2, 2], 1)).unwrap();
            assert!(one.groups[0].is_some() && one.groups[1].is_none());
        }
    }

    /// What is not a layer the kernels take is refused: a matrix of another shape or width than the layer's, an expert's two matrices of two widths, signs that are not +1 and -1 or not W of them, a block that does not divide the width.
    #[test]
    fn layers_that_are_not_the_kernels_are_refused() {
        let mut rng = Rng(3);
        let good = random_layer(&mut rng, 256, 128, 32, true, &[2, 3], 1);
        assert!(HbgHost::of(&good).is_ok());
        let why = |l: &LayerRows| HbgHost::of(l).err().expect("refused").to_string();
        let mut l = good.clone();
        l.experts[1].gate_up = random_matrix(&mut rng, 3, 32, 256);
        assert!(why(&l).contains("expert 1"), "a gated layer's first matrix has 2 I rows");
        let mut l = good.clone();
        l.experts[0].down_t = random_matrix(&mut rng, 3, 32, 256);
        assert!(why(&l).contains("expert 0"), "down_t at other bits than the expert's");
        let mut l = good.clone();
        l.experts[0].down_t = random_matrix(&mut rng, 2, 32, 128);
        assert!(why(&l).contains("expert 0"), "down_t of another width");
        let mut l = good.clone();
        l.signs[3] = 0;
        assert!(why(&l).contains("signs"));
        let mut l = good.clone();
        l.signs.pop();
        assert!(why(&l).contains("signs"));
        let mut l = good.clone();
        l.rot = 96;
        assert!(why(&l).contains("rotation blocks"));
        let mut l = good.clone();
        l.rot = 512;
        assert!(why(&l).contains("rotation blocks"));
        let mut l = good.clone();
        l.experts.clear();
        assert!(why(&l).contains("0 experts"));
        let mut l = good.clone();
        l.inter = 36;
        assert!(HbgHost::of(&l).is_err());
        let mut l = good.clone();
        l.experts[0].bits = 4;
        assert!(why(&l).contains("4 bits"));
    }

    /// A layer's config is held to the experts' shape: the number of experts, the width they work in (the latent where there is one), the intermediate size, a gate where the activation has one.
    #[test]
    fn the_layers_config_is_held_to_the_experts_shape() {
        let cfg = |experts, hidden, inter, act| MoeCfg { experts, router_hidden: 512, hidden, inter, router: Router::softmax(2), act };
        let shape = HbgShape { experts: 7, width: 256, rot: 256, inter: 64, gated: true };
        let swiglu = Act::SwiGlu { limit: None };
        shape.check(&cfg(7, 256, 64, swiglu)).unwrap();
        shape.check(&cfg(7, 256, 64, Act::Situ { beta: 4.0, linear: 25.0 })).unwrap();
        for (c, why) in [
            (cfg(8, 256, 64, swiglu), "7 of width 256"),
            (cfg(7, 512, 64, swiglu), "width 256"),
            (cfg(7, 256, 128, swiglu), "64 neurons"),
            (cfg(7, 256, 64, Act::Relu2), "gated"),
        ] {
            let e = shape.check(&c).unwrap_err().to_string();
            assert!(e.contains(why), "{e} does not say {why:?}");
        }
        let flat = HbgShape { gated: false, ..shape };
        flat.check(&cfg(7, 256, 64, Act::Relu2)).unwrap();
        assert!(flat.check(&cfg(7, 256, 64, swiglu)).is_err());
        assert!(HbgShape { inter: 60, ..shape }.check(&cfg(7, 256, 60, swiglu)).is_err(), "neurons in runs of 8");
        assert!(HbgShape { rot: 96, ..shape }.check(&cfg(7, 256, 64, swiglu)).is_err());
        assert!(HbgShape { experts: 2000, ..shape }.check(&cfg(2000, 256, 64, swiglu)).is_err());
        assert_eq!((shape.up_rows(), flat.up_rows()), (128, 64));
    }

    /// The types of the parameters of a kernel of kernels/moe_hbg.cu in the source's order, one letter each: `p` a pointer, `i` an int, `l` a long long, `f` a float.
    fn params(name: &str) -> String {
        let head = format!("extern \"C\" __global__ void __launch_bounds__(256) {name}(");
        let at = SRC.find(&head).unwrap_or_else(|| panic!("no kernel {name} in kernels/moe_hbg.cu")) + head.len();
        let list = &SRC[at..at + SRC[at..].find(')').expect("the parameters' end")];
        list.split(',')
            .map(|p| match p {
                p if p.contains('*') => 'p',
                p if p.contains("long long") => 'l',
                p if p.contains("int ") => 'i',
                p if p.contains("float ") => 'f',
                other => panic!("{name}: a parameter {other:?}"),
            })
            .collect()
    }

    /// What the launches in this file pass is what the kernels declare: each kernel is in the source, its parameters in the types and the number the launcher gives (device addresses, ints, the strides as long long, the rotation's scale as a
    /// float), and the sizes the launchers share with the kernels are the source's.
    #[test]
    fn kernels_and_signatures_are_in_the_source() {
        // (the launches: rot_in x xs signs xr W b scale; up and down the pairs' arrays (plan, ids, the tables) and the matrix's three arrays and their strides; inverse plan inv pairs E; sum_rot part inv wts ids signs y extra E k W b scale;
        // unrot part plan signs rows E W b scale)
        for (name, want) in [
            ("hbg_rot_in", "pipp".to_string() + "iif"),
            ("hbg_up2", "pppppppp".to_string() + "lllpiiii"),
            ("hbg_up3", "pppppppp".to_string() + "lllpiiii"),
            ("hbg_down2", "pppppppp".to_string() + "lllpiii"),
            ("hbg_down3", "pppppppp".to_string() + "lllpiii"),
            ("hbg_inv", "ppii".to_string()),
            ("hbg_sum_rot", "ppppppp".to_string() + "iiiif"),
            ("hbg_unrot", "pppp".to_string() + "iiif"),
        ] {
            assert_eq!(params(name), want, "{name}");
        }
        let define = |n: &str| -> Option<usize> {
            SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!(
            (define("HBG_UP_ROWS"), define("HBG_DOWN_OUT"), define("HBG_XPAD"), define("HBG_MAX_RUNS")),
            (Some(UP_ROWS), Some(DOWN_OUT), Some(XPAD), Some(MAX_WIDTH / RUN))
        );
        assert_eq!(define("HBG_RL"), Some(MAX_WIDTH / RUN / 32), "the runs a lane of the first product has at most");
        assert!(SRC.contains("tile * HBG_UP_ROWS + warp * 8") && SRC.contains("chunk * 4 + r"), "the tiles of the products");
        // (the kernels use moe_hb.cu's helpers and the engine includes this file after it; build.rs watches it)
        assert!(SRC.contains("hb_code<BITS>") && SRC.contains("hb_bf("));
        let engine = include_str!("../kernels/engine.cu");
        assert!(engine.find("#include \"moe_hb.cu\"").unwrap() < engine.find("#include \"moe_hbg.cu\"").expect("engine.cu includes moe_hbg.cu"));
        assert!(include_str!("../build.rs").contains("rerun-if-changed=kernels/moe_hbg.cu"));
    }

    /// A layer's scratch sizes: the rotated inputs, the parts and the pairs' rows.
    #[test]
    fn the_scratch_sizes() {
        assert_eq!((xr_bytes(2048, 3584), part_bytes(2048 * 8, 3584), rows_bytes(2048 * 8, 3584)), (2048 * 3584 * 4, 2048 * 8 * 3584 * 4, 2048 * 8 * 3584 * 2));
    }
}
