//! Hummingbird (Linear GLY-98, GLY-134): Nemotron 3 Super's routed experts at 3 and 2 bits a weight on the GPU, `NhMoeHb`, the `MoeBlock` the layers of a glyd-v7 directory at the Hummingbird level run (`hb` in glyd.json). An expert is
//! two matrices of 2,688 neuron rows of 1,024 values (`up`, and `downT`, down transposed), each row its codes (3 or 2 bits, a bitstream), a bf16 scale for each run of 32 and a bf16 codebook of its own (glyd-gpu's codebook.rs
//! has the layout; 464 and 328 bytes a row, 2,494,464 and 1,763,328 bytes an expert), in the basis of one rotation R = H diag(signs) of the 1,024 axis for the layer (H the natural-order Hadamard matrix over sqrt 1,024, the signs
//! a tensor of the directory). A layer's call is: the router on the layer's input; `fc1_latent_proj` of it, z; z' = R z once for the token (kernels/moe_hb.cu `hb_rot_in`); for each expert a pair chose, the first product over its up
//! rows with relu squared (`hb_up2`, `hb_up3`: u_n = <w_n, z'>, rounded to bf16, relu squared rounded to bf16 once), the second product over the rows of the neurons that are not zero (`hb_down2`, `hb_down3`: the rows of downT of
//! the active neurons added, each times its activation, into the 1,024 outputs; the sparse product of src/moe_rows.rs); the weighted sum of a token's pairs and R^T once (`hb_sum_rot`: y = signs o (H y')), one rounding to bf16;
//! `fc2_latent_proj`; the shared expert. No atomics: every launch gives the same bits every time, and an expert gives the same bits in whatever place of the GPU's memory it is, so that a layer through the expert cache is the layer
//! with every expert resident bit for bit, and a decode step of a layer with every expert there is captured in a CUDA graph.
//!
//! An expert has its own bits (the recipe hb40: the 40% most used of a layer 3, the rest 2), so a layer's experts are held in two groups, one a bit width (`HbStack`: the codes, and the scales and codebooks, of its experts end to end,
//! an expert's two matrices at a stride, the same for all of a group's): a group is a layer of its own to the expert cache (src/offload.rs: layer 2 i + 1 for the three-bit group of the i-th mixture-of-experts layer, 2 i for the other),
//! with places in its width's size, and the pairs of a call are sent to their group's kernels by their experts' numbers in it (`hb_group_ids`). A place of the cache holds an expert's two matrices as the stack does, the copy moves
//! them (`PackIn`: the codes, then the scales and codebooks), and the host tier holds each group at its real bytes. A layer's budget of places is split between the groups by their share of the experts.
//!
//! The SSD tier (src/offload.rs `Tier`): where the computer's memory cannot hold all of a model's experts (32 GB against 39.2 GiB), a layer's experts stay in the packed directory and `HbDisk` reads one into a slot of the host tier
//! when the router asks for it: the six tensors of its two matrices are one run of the shard (the tensors of a shard are in the order of their names), read in one direct read (src/disk.rs: no page cache) and placed as `HbPlace` says.
//! `index_layer` finds each expert's width, so its group and number there, from glyd.json alone, and a layer is loaded without reading its experts.
//!
//! The host's half (Linear GLY-130; src/offload.rs `run_groups`): both matrices of an expert are neuron rows in the host tier, so the CPU computes the pairs of the experts that are not in places from the host tier where they are
//! (`HbPlace::expert`, src/cpu_hb.rs): no second copy of any matrix, no memory besides the experts'. The host's inputs are the rotated z' (fp32, 4 KB a token, read back with the router's choices and weights: the layer's one wait),
//! its results the pairs' rows times their routing weights in the rotated basis, added by token (fp32) and sent up (`GroupsCall::part`); `hb_sum_rot` adds them to the GPU's weighted sum before R^T and the one rounding to bf16,
//! the pairs the host computed being no experts in the sum's ids. The planner (`Mode`: copy, cpu, split, auto) sees both groups' misses at once, and a copy costs by its group's bytes.

use crate::cpu_hb::{CbView, HbExpert};
use crate::dev::{Buf, Dev, Kernel};
use crate::disk::{Aligned, DirectFile};
use crate::ffi::{err, Error, Lib, Result, Stream};
use crate::kern35::Mat;
use crate::kern_nh::{MoeBlock, NhHost};
use crate::launch;
use crate::moe::{add_shared_with, route_with, shared_add_with, shared_mlp_with, MoeCfg, MoeScratch, Shape, Shared};
use crate::moe_nh::{bias_f32, moe_cfg};
use crate::moe_rows::{split_for, ROWS_MAX_INTER};
use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
use crate::offload::{Call, DiskIn, ExpertStore, Groups, GroupsCall, HostHalf, PackIn, Reader, Stage, CPU_TOKENS};
use crate::weights::{ordered, read_pack, upload_bytes, upload_packed, v7_manifest, Layout};
use glyd_gpu::codebook::{cb_rows_bits, cb_strides, CbRows};
use glyd_gpu::json::Value;
use glyd_gpu::safetensors::Checkpoint;
use std::cell::{OnceCell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// The share (percent) of a layer's experts at three bits in the recipe hb40, which a plan assumes where it has not read the directory.
pub const HB40_THREE_BIT_PERCENT: usize = 40;
/// Neurons a block of `hb_up2` and `hb_up3` takes (kernels/moe_hb.cu defines the same: a test holds them equal).
const UP_ROWS: usize = 64;
/// Outputs a block of `hb_down2` and `hb_down3` takes (the sparse product's: src/moe_rows.rs).
const DOWN_OUT: usize = 1024;
/// The widths of rows the rotation's kernels take: a power of two from 128 to 1,024.
const ROT_MIN: usize = 128;
const ROT_MAX: usize = 1024;

/// A bit width's index: 0 for two bits, 1 for three (the group a layer's experts of that width are in, and the kernel).
fn group_index(bits: u32) -> usize {
    bits as usize - 2
}

/// The Hummingbird layer's kernels (kernels/moe_hb.cu) with typed launches.
pub struct HbKern {
    rot: Kernel,
    group: Kernel,
    up: [Kernel; 2],
    down: [Kernel; 2],
    sum: Kernel,
    sms: usize,
}

impl HbKern {
    pub fn load(dev: &Dev) -> Result<HbKern> {
        Ok(HbKern {
            rot: dev.kernel(kname!("hb_rot_in"))?,
            group: dev.kernel(kname!("hb_group_ids"))?,
            up: [dev.kernel(kname!("hb_up2"))?, dev.kernel(kname!("hb_up3"))?],
            down: [dev.kernel(kname!("hb_down2"))?, dev.kernel(kname!("hb_down3"))?],
            sum: dev.kernel(kname!("hb_sum_rot"))?,
            sms: dev.sms.max(1) as usize,
        })
    }

    /// The blocks a pair's neurons are shared among in the second product, for `pairs` pairs (the sparse product's: `moe_rows::split_for`).
    pub fn split(&self, pairs: usize) -> usize {
        split_for(self.sms, pairs)
    }

    /// zr [tokens, n] (fp32) = R z for z [tokens, n] bf16 (rows `zs` elements apart): the signs (int8, +1 and -1) applied, then the Hadamard transform over sqrt n. n a power of two, 128 to 1,024.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernel reads and writes (kernels/moe_hb.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rot_in(&self, s: Stream, z: u64, zs: usize, signs: u64, zr: u64, tokens: usize, n: usize) -> Result<()> {
        if !(ROT_MIN..=ROT_MAX).contains(&n) || !n.is_power_of_two() || zs < n {
            return err(format!("a transform over {n} values in rows {zs} apart: a power of two from {ROT_MIN} to {ROT_MAX}, the rows at least that apart"));
        }
        if tokens == 0 {
            return Ok(());
        }
        launch!(self.rot, [tokens as u32, 1, 1], [(n / 4) as u32, 1, 1], s; z, zs as i32, signs, zr, n as i32, (1.0 / (n as f64).sqrt()) as f32)
    }

    /// out [pairs] (int64) = for each router choice `ids` [pairs] the expert's number in group `group` (`group_of` [E] u8 says each expert's group, `local_of` [E] u16 its number there), else -1.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn group_ids(&self, s: Stream, ids: u64, group_of: u64, local_of: u64, group: usize, out: u64, pairs: usize, experts: usize) -> Result<()> {
        if pairs == 0 {
            return Ok(());
        }
        launch!(self.group, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; ids, group_of, local_of, group as i32, out, pairs as i32, experts as i32)
    }

    /// The first product of a group's experts (`bits` 2 or 3) with relu squared: a [pairs in the plan, inter] (bf16) from zr, for the pairs of `plan` (`Lib::moe_way`'s over `experts` ids, made of `via`: the expert's
    /// number in the stack, or its place in the cache's), the matrix the three arrays and their strides between places describe.
    ///
    /// # Safety
    /// As `rot_in`; the ids in `via` are below the places the stack has.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn up(
        &self,
        s: Stream,
        bits: u32,
        st: &HbStack,
        zr: u64,
        plan: u64,
        via: u64,
        a: u64,
        experts: usize,
        inter: usize,
        cols: usize,
        k: usize,
        pairs: usize,
    ) -> Result<()> {
        let m = st.up_arrays();
        launch!(self.up[group_index(bits)], [pairs as u32, inter.div_ceil(UP_ROWS) as u32, 1], [256, 1, 1], s;
            zr, plan, via, m.codes, m.scales, m.book, m.codes_stride, m.scales_stride, m.book_stride, a, experts as i32, inter as i32, cols as i32, k as i32)
    }

    /// The second product of a group's experts (`bits` 2 or 3) for the neurons that are not zero: part [pairs, split, hidden] (fp32) from `a` (the first product's rows in the plan's order), over the rows of downT.
    ///
    /// # Safety
    /// As `up`; `part` has room for `pairs split hidden` floats.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn down(
        &self,
        s: Stream,
        bits: u32,
        st: &HbStack,
        a: u64,
        plan: u64,
        via: u64,
        part: u64,
        experts: usize,
        inter: usize,
        hidden: usize,
        split: usize,
        pairs: usize,
    ) -> Result<()> {
        let m = st.down_arrays();
        launch!(self.down[group_index(bits)], [pairs as u32, split as u32, hidden.div_ceil(DOWN_OUT) as u32], [256, 1, 1], s;
            a, plan, via, m.codes, m.scales, m.book, m.codes_stride, m.scales_stride, m.book_stride, part, experts as i32, inter as i32, hidden as i32, split as i32)
    }

    /// y [tokens, hidden] (bf16) = R^T of the weighted sum of each token's pairs' parts (`wts` [pairs] fp32; `ids` the router's choices, int64 [pairs], an expert below `experts` counting) and of `extra` (fp32 [tokens, hidden] in the
    /// rotated basis, the host's weighted sums of the pairs it computed: 0, none): signs o (H y') over sqrt hidden, rounded to bf16 once.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sum_rot(
        &self,
        s: Stream,
        part: u64,
        wts: u64,
        ids: u64,
        signs: u64,
        y: u64,
        extra: u64,
        tokens: usize,
        experts: usize,
        k: usize,
        hidden: usize,
        split: usize,
    ) -> Result<()> {
        if !(ROT_MIN..=ROT_MAX).contains(&hidden) || !hidden.is_power_of_two() {
            return err(format!("a transform over {hidden} values: a power of two from {ROT_MIN} to {ROT_MAX}"));
        }
        if tokens == 0 {
            return Ok(());
        }
        launch!(self.sum, [tokens as u32, 1, 1], [(hidden / 4) as u32, 1, 1], s; part, wts, ids, signs, y, extra, experts as i32, k as i32, hidden as i32, split as i32, (1.0 / (hidden as f64).sqrt()) as f32)
    }
}

// ---- a group's experts: the bytes of a place and of a stack -----------------------------------------------------------------------------------------------------------------

fn pad16(n: usize) -> usize {
    n.next_multiple_of(16)
}

/// Where the parts of an expert of `bits` bits are in a place (an expert's bytes in the stack, or a place of the cache): `data` holds the up matrix's codes and then downT's, `meta` the up matrix's scales and codebook and then downT's,
/// each part to 16 bytes. Both matrices are `rows` neuron rows (the intermediate size) of `cols` values (the latent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HbPlace {
    pub bits: u32,
    pub rows: usize,
    pub cols: usize,
    /// bytes of `data` and of `meta` an expert takes
    pub data: usize,
    pub meta: usize,
    /// where downT's codes start in `data`; where the up matrix's codebook, downT's scales and downT's codebook start in `meta`
    pub down_codes: usize,
    pub up_book: usize,
    pub down_scales: usize,
    pub down_book: usize,
}

impl HbPlace {
    pub fn new(bits: u32, rows: usize, cols: usize) -> Result<HbPlace> {
        if !(2..=3).contains(&bits) || rows == 0 || cols == 0 || !cols.is_multiple_of(32) {
            return err(format!("an expert of {rows} neuron rows of {cols} values at {bits} bits: 2 or 3 bits and rows in runs of 32 are wanted"));
        }
        let (c, s, b) = cb_strides(bits, cols);
        let (codes, scales, book) = (pad16(rows * c), pad16(rows * s), pad16(rows * b));
        Ok(HbPlace {
            bits,
            rows,
            cols,
            data: 2 * codes,
            meta: 2 * (scales + book),
            down_codes: codes,
            up_book: scales,
            down_scales: scales + book,
            down_book: 2 * scales + book,
        })
    }

    /// The bytes an expert takes in the stack, as the plans count them (codes, scales and codebooks of its two matrices).
    pub fn bytes(&self) -> u64 {
        (self.data + self.meta) as u64
    }

    /// An expert's two matrices as the CPU half reads them (src/cpu_hb.rs), over the bytes of its place as they are: `data` and `meta` are its place's `data` and `meta` bytes, in the host tier or in a stack's host copy (nothing is copied).
    pub fn expert<'a>(&self, data: &'a [u8], meta: &'a [u8]) -> Result<HbExpert<'a>> {
        if data.len() != self.data || meta.len() != self.meta {
            return err(format!(
                "{} + {} bytes, where an expert of {} x {} at {} bits takes {} + {}",
                data.len(),
                meta.len(),
                self.rows,
                self.cols,
                self.bits,
                self.data,
                self.meta
            ));
        }
        let (c, s, b) = cb_strides(self.bits, self.cols);
        let view = |codes: usize, scales: usize, book: usize| CbView {
            bits: self.bits,
            rows: self.rows,
            cols: self.cols,
            codes: &data[codes..codes + self.rows * c],
            scales: &meta[scales..scales + self.rows * s],
            codebook: &meta[book..book + self.rows * b],
        };
        HbExpert::new(view(0, 0, self.up_book), view(self.down_codes, self.down_scales, self.down_book))
    }

    /// Expert's two matrices written into its place: `data` and `meta` are the place's bytes.
    fn put(&self, up: &CbRows, down: &CbRows, data: &mut [u8], meta: &mut [u8]) -> Result<()> {
        for m in [up, down] {
            if (m.bits, m.rows, m.cols) != (self.bits, self.rows, self.cols) || !m.is_valid() {
                return err(format!(
                    "a matrix of {} x {} at {} bits, where the group's experts are {} x {} at {}",
                    m.rows, m.cols, m.bits, self.rows, self.cols, self.bits
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

/// A group's experts end to end on the host (expert k's place at k times the place's `data` and `meta`).
pub struct HbHost {
    pub place: HbPlace,
    pub experts: usize,
    pub data: Vec<u8>,
    pub meta: Vec<u8>,
}

impl HbHost {
    /// The experts' pairs of matrices (up, downT), joined: all at the place's bits and shape.
    pub fn of(place: HbPlace, pairs: &[(CbRows, CbRows)]) -> Result<HbHost> {
        let (mut data, mut meta) = (vec![0u8; pairs.len() * place.data], vec![0u8; pairs.len() * place.meta]);
        for (k, (up, down)) in pairs.iter().enumerate() {
            place.put(up, down, &mut data[k * place.data..(k + 1) * place.data], &mut meta[k * place.meta..(k + 1) * place.meta])?;
        }
        Ok(HbHost { place, experts: pairs.len(), data, meta })
    }
}

/// The three arrays of one matrix of a stack, as the kernels take them: the addresses of the first place's, and the bytes between places.
#[derive(Clone, Copy, Debug)]
pub struct HbArrays {
    pub codes: u64,
    pub scales: u64,
    pub book: u64,
    pub codes_stride: i64,
    pub scales_stride: i64,
    pub book_stride: i64,
}

/// A group's experts (one bit width) in device memory: `slots` places of `experts`, the codes in `data` and the scales and codebooks in `meta`.
pub struct HbStack {
    pub place: HbPlace,
    pub experts: usize,
    /// The experts the buffers have room for: `experts`, or fewer in a stack made by `pooled` (the expert cache's places).
    pub slots: usize,
    pub bytes: usize,
    data: Buf,
    meta: Buf,
}

impl HbStack {
    /// The experts of `h` copied to the GPU.
    pub fn upload(h: &HbHost) -> Result<HbStack> {
        let (data, meta) = (upload_bytes(&h.data)?, upload_bytes(&h.meta)?);
        Ok(HbStack { place: h.place, experts: h.experts, slots: h.experts, bytes: data.bytes + meta.bytes, data, meta })
    }

    /// Room for `slots` experts standing for a group's `experts` (zeroed): the expert cache's places.
    pub fn pooled(place: HbPlace, experts: usize, slots: usize) -> Result<HbStack> {
        if slots == 0 || slots > experts {
            return err(format!("room for {slots} experts standing for {experts}"));
        }
        let (data, meta) = (Buf::zeroed(slots * place.data)?, Buf::zeroed(slots * place.meta)?);
        Ok(HbStack { place, experts, slots, bytes: data.bytes + meta.bytes, data, meta })
    }

    /// The addresses of the codes and of the scales and codebooks (of the first place's), and the bytes of an expert's of each: what the expert cache copies an expert into.
    pub fn places(&self) -> ([u64; 2], [u64; 2]) {
        (self.buffers(), [self.place.data as u64, self.place.meta as u64])
    }

    fn buffers(&self) -> [u64; 2] {
        [self.data.ptr, self.meta.ptr]
    }

    pub fn up_arrays(&self) -> HbArrays {
        let p = &self.place;
        HbArrays {
            codes: self.data.ptr,
            scales: self.meta.ptr,
            book: self.meta.ptr + p.up_book as u64,
            codes_stride: p.data as i64,
            scales_stride: p.meta as i64,
            book_stride: p.meta as i64,
        }
    }

    pub fn down_arrays(&self) -> HbArrays {
        let p = &self.place;
        HbArrays {
            codes: self.data.ptr + p.down_codes as u64,
            scales: self.meta.ptr + p.down_scales as u64,
            book: self.meta.ptr + p.down_book as u64,
            codes_stride: p.data as i64,
            scales_stride: p.meta as i64,
            book_stride: p.meta as i64,
        }
    }
}

// ---- reading a layer from a directory ---------------------------------------------------------------------------------------------------------------------------------------

/// A layer's experts read from a Hummingbird directory: the two groups (index 0 the two-bit experts, 1 the three-bit ones; None where a layer has none of that width), each expert's group and its number there, and the signs of the rotation.
pub struct HbLayerHost {
    pub groups: [Option<HbHost>; 2],
    pub group_of: Vec<u8>,
    pub local_of: Vec<u16>,
    pub signs: Vec<i8>,
}

/// The bits of the pack `key` of a Hummingbird directory, its entry in glyd.json held to the layout (c3r or c2r), the tag, the shape [rows, cols] and whether the tensor is its transpose (`down`).
fn matrix_bits(manifest: &Value, key: &str, rows: usize, cols: usize, down: bool) -> Result<u32> {
    let e = manifest.get("packs").and_then(|p| p.get(key)).ok_or_else(|| Error(format!("glyd.json: no pack {key}")))?;
    let lay = e.get("layout").and_then(Value::as_str).unwrap_or("");
    let bits = match lay {
        "c3r" => 3,
        "c2r" => 2,
        other => return err(format!("glyd.json: {key}: layout {other:?}, where a Hummingbird directory's experts are `c3r` or `c2r`")),
    };
    let tiers =
        e.get("tiers").and_then(Value::as_u64s).filter(|t| t.len() == 3).ok_or_else(|| Error(format!("glyd.json: {key}: the pack's words are missing")))?;
    if cb_rows_bits((tiers[0] >> 24) as u32) != Some(bits) || tiers[0] & 0xFF_FFFF != 0 || tiers[1] != 0 || tiers[2] != 0 {
        return err(format!(
            "glyd.json: {key}: words {tiers:x?}, where the {lay} layout's are [{:#x}, 0, 0]",
            (glyd_gpu::codebook::cb_rows_tag(bits).unwrap_or(0) as u64) << 24
        ));
    }
    if e.get("shape").and_then(Value::as_u64s) != Some(vec![rows as u64, cols as u64]) || e.get("transposed").and_then(Value::as_bool) != Some(down) {
        return err(format!(
            "glyd.json: {key}: shape {:?} (transposed {:?}), where the config has {rows} neuron rows of {cols} values ({})",
            e.get("shape").and_then(Value::as_u64s),
            e.get("transposed").and_then(Value::as_bool),
            if down { "the transpose of the weight" } else { "the weight itself" }
        ));
    }
    Ok(bits)
}

/// The pack `key` of a Hummingbird directory as a matrix of neuron rows: its entry in glyd.json held to the layout (`matrix_bits`), its three buffers read.
fn read_matrix(ck: &Checkpoint, manifest: &Value, key: &str, rows: usize, cols: usize, down: bool) -> Result<CbRows> {
    let bits = matrix_bits(manifest, key, rows, cols, down)?;
    let read = |b: &str| ck.read(&format!("{key}.glyd_{b}")).map_err(|e| Error(e.to_string()));
    CbRows::new(bits, rows, cols, read("data")?, read("blocks")?, read("codebook")?).ok_or_else(|| {
        Error(format!("{key}: buffers that are not {rows} neuron rows of {cols} values at {bits} bits, or a scale or an entry that is not finite"))
    })
}

/// The signs of layer `l`'s rotation: `glyd_rot_signs`, int8 [cols] of +1 and -1.
fn read_signs(ck: &Checkpoint, c: &ConfigNemotronH, l: usize) -> Result<Vec<i8>> {
    let name = format!("{}layers.{l}.mixer.glyd_rot_signs", c.prefix);
    let (_, t) = ck.get(&name).ok_or_else(|| Error(format!("{}: no tensor {name}: the directory is not a Hummingbird one", ck.dir.display())))?;
    if t.dtype != "I8" || t.shape != [c.moe.latent as u64] {
        return err(format!("{name} is {} {:?}, where I8 [{}] is wanted", t.dtype, t.shape, c.moe.latent));
    }
    let signs: Vec<i8> = ck.read(&name).map_err(|e| Error(e.to_string()))?.into_iter().map(|b| b as i8).collect();
    if signs.iter().any(|&s| s != 1 && s != -1) {
        return err(format!("{name}: a sign that is neither +1 nor -1"));
    }
    Ok(signs)
}

/// Layer `l`'s experts and signs from `ck` (a directory at the Hummingbird level), each expert's two packs read on as many threads as the machine has, in order, and joined by bit width. ponytail: the host holds the layer's packs
/// twice at the join (the real model's 1.05 GB a layer, 2.1 GB at the peak); group by group into the device buffers if a host with less memory runs it.
pub fn read_layer(ck: &Checkpoint, manifest: &Value, c: &ConfigNemotronH, l: usize) -> Result<HbLayerHost> {
    let (m, p) = (&c.moe, format!("{}layers.{l}.mixer.experts", c.prefix));
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let (mut pairs, mut group_of): ([Vec<(CbRows, CbRows)>; 2], Vec<u8>) = ([Vec::new(), Vec::new()], Vec::new());
    let mut local_of = Vec::new();
    ordered(
        m.experts,
        threads,
        2 * threads,
        &|e| {
            Ok((
                read_matrix(ck, manifest, &format!("{p}.{e}.up_proj"), m.inter, m.latent, false)?,
                read_matrix(ck, manifest, &format!("{p}.{e}.down_proj"), m.inter, m.latent, true)?,
            ))
        },
        |e, (up, down)| {
            if up.bits != down.bits {
                return err(format!("{p}.{e}: its up matrix is {} bits and its down matrix {}", up.bits, down.bits));
            }
            let g = group_index(up.bits);
            group_of.push(g as u8);
            local_of.push(pairs[g].len() as u16);
            pairs[g].push((up, down));
            Ok(())
        },
    )?;
    let mut groups = [None, None];
    for (g, ps) in pairs.iter().enumerate() {
        if !ps.is_empty() {
            groups[g] = Some(HbHost::of(HbPlace::new(g as u32 + 2, m.inter, m.latent)?, ps)?);
        }
    }
    Ok(HbLayerHost { groups, group_of, local_of, signs: read_signs(ck, c, l)? })
}

// ---- the experts that stay on the disk ------------------------------------------------------------------------------------------------------------------------------------------

/// A layer's experts found without reading any of their bytes (glyd.json says each one's width): each expert's group and its number there, as `read_layer` numbers them (by the order of the experts), and the experts of each group
/// in its order (`members[g][local]` the expert's number in the layer).
pub struct HbIndex {
    pub group_of: Vec<u8>,
    pub local_of: Vec<u16>,
    pub members: [Vec<u16>; 2],
}

impl HbIndex {
    pub fn counts(&self) -> [usize; 2] {
        [self.members[0].len(), self.members[1].len()]
    }
}

/// Layer `l`'s experts' groups from glyd.json alone: the widths of each expert's two matrices (which must be one), held to the config's shape.
pub fn index_layer(manifest: &Value, c: &ConfigNemotronH, l: usize) -> Result<HbIndex> {
    let (m, p) = (&c.moe, format!("{}layers.{l}.mixer.experts", c.prefix));
    let mut out = HbIndex { group_of: Vec::new(), local_of: Vec::new(), members: [Vec::new(), Vec::new()] };
    for e in 0..m.experts {
        let (up, down) = (
            matrix_bits(manifest, &format!("{p}.{e}.up_proj"), m.inter, m.latent, false)?,
            matrix_bits(manifest, &format!("{p}.{e}.down_proj"), m.inter, m.latent, true)?,
        );
        if up != down {
            return err(format!("{p}.{e}: its up matrix is {up} bits and its down matrix {down}"));
        }
        let g = group_index(up);
        out.group_of.push(g as u8);
        out.local_of.push(out.members[g].len() as u16);
        out.members[g].push(e as u16);
    }
    Ok(out)
}

/// The bytes of all the routed experts of a directory (every mixture-of-experts layer's, by the widths glyd.json gives them): what a budget for the computer's memory is shared against.
pub fn experts_bytes(manifest: &Value, c: &ConfigNemotronH) -> Result<u64> {
    let mut total = 0u64;
    for l in (0..c.layers).filter(|&l| c.kinds[l] == NhKind::Moe) {
        for (g, n) in index_layer(manifest, c, l)?.counts().into_iter().enumerate() {
            total += n as u64 * expert_bytes_at(c, g as u32 + 2)?;
        }
    }
    Ok(total)
}

/// A layer as `read_layer` gives it but with none of its experts' bytes: the groups' places and the numbers of their experts (buffers empty), the groups of the experts and the signs of the rotation.
fn index_host(ck: &Checkpoint, c: &ConfigNemotronH, l: usize, ix: &HbIndex) -> Result<HbLayerHost> {
    let mut groups = [None, None];
    for (g, n) in ix.counts().into_iter().enumerate().filter(|&(_, n)| n > 0) {
        groups[g] = Some(HbHost { place: HbPlace::new(g as u32 + 2, c.moe.inter, c.moe.latent)?, experts: n, data: Vec::new(), meta: Vec::new() });
    }
    Ok(HbLayerHost { groups, group_of: ix.group_of.clone(), local_of: ix.local_of.clone(), signs: read_signs(ck, c, l)? })
}

/// Experts' parts that are within this many bytes of each other in a shard are read in one read (the tensors of an expert are one run of the shard when the shard was written in the names' order; a shard boundary or another
/// order splits them).
const GAP: u64 = 64 << 10;

/// Where one part of an expert is: the shard (an index of `HbDisk::files`), the byte of the file, the length, and where it goes in the expert's place: `to` in `meta` or in `data`.
#[derive(Clone, Copy, Default)]
struct Seg {
    file: u16,
    meta: bool,
    to: u32,
    len: u32,
    at: u64,
}

/// An expert's group and the six parts of its two matrices (the codes, the scales and the codebook of each) in the files.
struct DiskExpert {
    group: u8,
    segs: [Seg; 6],
}

thread_local! {
    /// The buffer a thread's direct reads land in before their parts go to the slot (the biggest read the thread has made).
    static BOUNCE: RefCell<Aligned> = const { RefCell::new(Aligned::new()) };
}

/// A layer's experts in the directory's shards, for reading them one at a time into the bytes of a place (`HbPlace`: the codes of the up matrix and then of downT; the scales and the codebook of the up matrix, then of downT):
/// the shards the experts are in, opened for reads that keep nothing in the page cache, and for each expert where its six tensors are. The reader of a group (`reader`) is what the expert cache's disk tier calls.
pub struct HbDisk {
    files: Vec<DirectFile>,
    experts: Vec<DiskExpert>,
    places: [HbPlace; 2],
}

impl HbDisk {
    /// The experts of layer `l` of the directory `ck` (`index`: their groups): each tensor found and held to the size its place wants.
    pub fn open(ck: &Checkpoint, c: &ConfigNemotronH, l: usize, index: &HbIndex) -> Result<HbDisk> {
        let (m, p) = (&c.moe, format!("{}layers.{l}.mixer.experts", c.prefix));
        let places = [HbPlace::new(2, m.inter, m.latent)?, HbPlace::new(3, m.inter, m.latent)?];
        let (mut paths, mut experts): (Vec<PathBuf>, Vec<DiskExpert>) = (Vec::new(), Vec::with_capacity(m.experts));
        for e in 0..m.experts {
            let g = index.group_of[e] as usize;
            let place = &places[g];
            let (cs, ss, bs) = cb_strides(place.bits, place.cols);
            let mut segs = [Seg::default(); 6];
            let mut i = 0;
            for (matrix, codes, scales, book) in [("up_proj", 0, 0, place.up_book), ("down_proj", place.down_codes, place.down_scales, place.down_book)] {
                for (part, bytes, meta, to) in
                    [("data", place.rows * cs, false, codes), ("blocks", place.rows * ss, true, scales), ("codebook", place.rows * bs, true, book)]
                {
                    let key = format!("{p}.{e}.{matrix}.glyd_{part}");
                    let (file, t) = ck.get(&key).ok_or_else(|| Error(format!("{}: no tensor {key}", ck.dir.display())))?;
                    if t.bytes != bytes as u64 || t.dtype != "U8" {
                        return err(format!("{key} is {} bytes of {}, where an expert's {part} at {} bits are {bytes} of U8", t.bytes, t.dtype, place.bits));
                    }
                    let f = paths.iter().position(|q| *q == file.path).unwrap_or_else(|| {
                        paths.push(file.path.clone());
                        paths.len() - 1
                    });
                    segs[i] = Seg { file: f as u16, meta, to: to as u32, len: bytes as u32, at: t.offset };
                    i += 1;
                }
            }
            experts.push(DiskExpert { group: g as u8, segs });
        }
        let files = paths.iter().map(|q| DirectFile::open(q).map_err(|e| Error(e.to_string()))).collect::<Result<Vec<_>>>()?;
        Ok(HbDisk { files, experts, places })
    }

    /// The bytes of one expert's place (`data`, `meta`: the sizes of its group's place), read from the shards and placed; what lies between the parts is zero, as `HbHost::of` leaves it. The parts that are near each other in a shard
    /// are read at once, the blocks that cover them in one direct read.
    pub fn read(&self, e: usize, data: &mut [u8], meta: &mut [u8]) -> Result<()> {
        let ex = self.experts.get(e).ok_or_else(|| Error(format!("expert {e} of {}", self.experts.len())))?;
        let place = &self.places[ex.group as usize];
        if data.len() != place.data || meta.len() != place.meta {
            return err(format!("a place of {} + {} bytes for an expert of {} + {}", data.len(), meta.len(), place.data, place.meta));
        }
        let mut order = [0usize, 1, 2, 3, 4, 5];
        order.sort_by_key(|&i| (ex.segs[i].file, ex.segs[i].at));
        BOUNCE.with(|b| -> Result<()> {
            let mut bounce = b.borrow_mut();
            let mut i = 0;
            while i < order.len() {
                let first = ex.segs[order[i]];
                let (mut end, mut j) = (first.at + first.len as u64, i + 1);
                while j < order.len() && ex.segs[order[j]].file == first.file && ex.segs[order[j]].at <= end + GAP {
                    end = end.max(ex.segs[order[j]].at + ex.segs[order[j]].len as u64);
                    j += 1;
                }
                let span = self.files[first.file as usize].read_span(first.at, (end - first.at) as usize, &mut bounce).map_err(|e| Error(e.to_string()))?;
                for &k in &order[i..j] {
                    let sg = ex.segs[k];
                    let from = (sg.at - first.at) as usize;
                    let to = if sg.meta { &mut *meta } else { &mut *data };
                    to[sg.to as usize..][..sg.len as usize].copy_from_slice(&span[from..from + sg.len as usize]);
                }
                i = j;
            }
            Ok(())
        })?;
        let (c, s, b) = cb_strides(place.bits, place.cols);
        let (codes, scales, book) = (place.rows * c, place.rows * s, place.rows * b);
        data[codes..place.down_codes].fill(0);
        data[place.down_codes + codes..].fill(0);
        meta[scales..place.up_book].fill(0);
        meta[place.up_book + book..place.down_scales].fill(0);
        meta[place.down_scales + scales..place.down_book].fill(0);
        meta[place.down_book + book..].fill(0);
        Ok(())
    }

    /// Expert `e` read into `data` and `meta`, made the sizes of its group's place (a caller with no slot to read into: a measurement of the disk).
    pub fn read_sized(&self, e: usize, data: &mut Vec<u8>, meta: &mut Vec<u8>) -> Result<()> {
        let place = self.places[self.experts.get(e).ok_or_else(|| Error(format!("expert {e} of {}", self.experts.len())))?.group as usize];
        data.resize(place.data, 0);
        meta.resize(place.meta, 0);
        self.read(e, data, meta)
    }

    /// How the shards are read, in a word (`DirectFile::mode`; of the first, which every read of an expert touches).
    pub fn mode(&self) -> &'static str {
        self.files.first().map_or("none", DirectFile::mode)
    }

    /// The bytes of expert `e`'s place (the codes, the scales and the codebooks of its two matrices).
    pub fn bytes_of(&self, e: usize) -> u64 {
        self.places[self.experts[e].group as usize].bytes()
    }

    /// The reader of a group for the expert cache's disk tier: its expert `local` is the layer's expert `members[local]`.
    pub fn reader(self: &Arc<HbDisk>, members: Vec<u16>) -> Reader {
        let disk = self.clone();
        Arc::new(move |local, data, meta| {
            let e = *members.get(local).ok_or_else(|| Error(format!("expert {local} of a group of {}", members.len())))?;
            disk.read(e as usize, data, meta)
        })
    }
}

// ---- the layer on the GPU ---------------------------------------------------------------------------------------------------------------------------------------------------

/// What the experts' products of every layer share besides the mixture-of-experts scratch: the rotated input (fp32), the router's choices in a group's numbers, and the places the expert cache sends.
struct HbScratch {
    kern: HbKern,
    zr: Buf,
    gids: Buf,
    // the pairs' places in each group's cache and the sum's ids, for the calls the host computes some of (at most `CPU_TOKENS` tokens)
    ways: [Buf; 2],
    sids: Buf,
}

/// The scratch of all a model's Hummingbird layers, made by the first that runs (as the engine's `moe_scratch`: one for all of them).
#[derive(Default)]
pub struct HbShared {
    scratch: OnceCell<HbScratch>,
}

/// One bit width's experts of a layer on the GPU, and the expert cache's layer number for them (None: all there).
struct HbGroup {
    stack: HbStack,
    store_layer: Option<usize>,
}

/// A Hummingbird mixture-of-experts layer on the GPU: the router, the shared expert and the latent projections as `NhMoe` has them, the experts in two groups.
pub struct NhMoeHb {
    cfg: MoeCfg,
    router: Buf,
    bias: Buf,
    shared: Shared,
    fc1: Mat,
    fc2: Mat,
    groups: [Option<HbGroup>; 2],
    group_of: Buf,
    local_of: Buf,
    signs: Buf,
    // each expert's group and its number there, on the host (the expert cache places a call's pairs by them)
    host_group_of: Vec<u8>,
    host_local_of: Vec<u16>,
    store: Option<Rc<ExpertStore>>,
    common: Rc<HbShared>,
}

/// Places for a layer of `n` experts of each width, out of `slots` (the number of experts a layer's budget buys at the mean of the widths): their shares of the experts, each group at least one place and at most all of its experts, the rest
/// going to the other group where one is full.
pub fn split_slots(slots: usize, n: [usize; 2]) -> [usize; 2] {
    let total = n[0] + n[1];
    if total == 0 {
        return [0, 0];
    }
    let slots = slots.min(total);
    let mut s = [0usize; 2];
    for g in 0..2 {
        s[g] = if n[g] == 0 { 0 } else { ((slots * n[g] + total / 2) / total).clamp(1, n[g]) };
    }
    // (the rounding and the minimums may leave the sum off: take from the group that has more than its share, or give to the one that can take)
    while s[0] + s[1] > slots && s.iter().zip(n).any(|(&a, b)| a > 1 && b > 0) {
        let g = if s[0] * n[1] > s[1] * n[0] { 0 } else { 1 };
        let g = if s[g] > 1 { g } else { 1 - g };
        s[g] -= 1;
    }
    while s[0] + s[1] < slots {
        let g = if s[0] * n[1] <= s[1] * n[0] && s[0] < n[0] || s[1] >= n[1] { 0 } else { 1 };
        if s[g] >= n[g] {
            break;
        }
        s[g] += 1;
    }
    s
}

/// The bytes of an expert at `bits` for the model `c` (its two matrices' codes, scales and codebooks).
pub fn expert_bytes_at(c: &ConfigNemotronH, bits: u32) -> Result<u64> {
    Ok(HbPlace::new(bits, c.moe.inter, c.moe.latent)?.bytes())
}

/// The mean bytes of an expert of the recipe hb40 for the model `c`: 40% of a layer's experts at three bits, the rest at two. What a plan of the cache counts with where it has not read the directory.
pub fn expert_bytes_mean(c: &ConfigNemotronH) -> Result<u64> {
    let e = c.moe.experts;
    let threes = (e * HB40_THREE_BIT_PERCENT).div_ceil(100);
    Ok((threes as u64 * expert_bytes_at(c, 3)? + (e - threes) as u64 * expert_bytes_at(c, 2)?) / e as u64)
}

/// The shapes a layer's matrices must have for the kernels: the latent a power of two from 128 to 1,024 (the rotation), the neurons at most the sparse product's list, in runs of 8.
fn check_shape(c: &ConfigNemotronH) -> Result<()> {
    let m = &c.moe;
    if !(ROT_MIN..=ROT_MAX).contains(&m.latent) || !m.latent.is_power_of_two() || m.inter > ROWS_MAX_INTER || !m.inter.is_multiple_of(8) || m.experts > 1024 {
        return err(format!("experts of {} neurons over a latent of {}: the Hummingbird kernels take a latent that is a power of two from {ROT_MIN} to {ROT_MAX}, at most {ROWS_MAX_INTER} neurons in runs of 8, and 1,024 experts", m.inter, m.latent));
    }
    Ok(())
}

impl NhMoeHb {
    /// Layer `l` of the model `c` from the Hummingbird directory `ck`, its experts all on the GPU, or (`cache`: the places a layer has in the mean of the experts' widths, and the store) in the computer's memory with places for that
    /// many on the GPU, split between the two widths (`split_slots`).
    pub fn load(l: usize, c: &ConfigNemotronH, ck: &Checkpoint, cache: Option<(usize, &Rc<ExpertStore>)>, common: &Rc<HbShared>) -> Result<NhMoeHb> {
        if c.moe.latent == 0 {
            return err("the hummingbird level's experts work on the latent: this model has none (kestrel, swift or wren take it)");
        }
        if c.kinds.get(l) != Some(&NhKind::Moe) {
            return err(format!("layer {l} is not a mixture-of-experts layer"));
        }
        check_shape(c)?;
        let v7 = v7_manifest(&ck.dir)?.ok_or_else(|| Error(format!("{}: not a glyd-v7 directory", ck.dir.display())))?;
        let (layout, manifest) = (v7.0, &v7.1);
        if layout != Layout::Hummingbird {
            return err(format!("{}: packed in the {} layout, where the Hummingbird level is asked", ck.dir.display(), layout.name()));
        }
        let p = format!("{}layers.{l}.mixer", c.prefix);
        let tensor = |name: &str| -> Result<(String, Vec<u8>)> {
            let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
            Ok((t.dtype.clone(), ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?))
        };
        let (dtype, router) = tensor(&format!("{p}.gate.weight"))?;
        if dtype != "BF16" {
            return err(format!("{p}.gate.weight is {dtype}, where the router is read as BF16"));
        }
        let bias = tensor(&format!("{p}.gate.e_score_correction_bias")).and_then(|(d, b)| bias_f32(&d, &b))?;
        let dense = c.layer_packs(l).iter().map(|pk| read_pack(ck, Some(&v7), pk, pack_layout(layout, &pk.key))).collect::<Result<Vec<_>>>()?;
        let Ok([fc1, fc2, s_up, s_down]) = <[_; 4]>::try_from(dense) else {
            return err(format!("layer {l}: a mixture-of-experts layer has four matrices besides its experts"));
        };
        let index = c.kinds[..l].iter().filter(|&&k| k == NhKind::Moe).count();
        // where the computer's memory holds only some of the model's experts (the store has a budget for them), the layer's experts stay in the packed directory and are read into the host tier's slots as the router asks for them
        // (src/offload.rs `Tier`): none of them is read here but the ones a group the GPU holds whole is uploaded from, and the slots' first experts
        let ix = cache.filter(|(_, store)| store.host_memory().is_some()).map(|_| index_layer(manifest, c, l)).transpose()?;
        let (host, disk) = match &ix {
            Some(ix) => {
                let counts = ix.counts();
                let whole = cache.is_some_and(|(n, _)| split_slots(n, counts).iter().zip(counts).any(|(&s, n)| n > 0 && s >= n));
                let host = if whole { read_layer(ck, manifest, c, l)? } else { index_host(ck, c, l, ix)? };
                (host, Some((Arc::new(HbDisk::open(ck, c, l, ix)?), ix)))
            }
            None => (read_layer(ck, manifest, c, l)?, None),
        };
        let counts = [0, 1].map(|g| host.groups[g].as_ref().map_or(0, |h| h.experts));
        let slots = cache.map(|(n, _)| split_slots(n, counts));
        let mut groups: [Option<HbGroup>; 2] = [None, None];
        for g in 0..2 {
            let Some(h) = &host.groups[g] else { continue };
            groups[g] = Some(match (cache, slots) {
                (Some((_, store)), Some(s)) if s[g] < h.experts => {
                    let stack = HbStack::pooled(h.place, h.experts, s[g])?;
                    let (dev, stride) = stack.places();
                    match &disk {
                        // (the host reads the group's places in the host tier where they are: no second copy of any matrix)
                        None => store.add_layer(
                            2 * index + g,
                            h.experts,
                            s[g],
                            &[PackIn { bytes: [&h.data, &h.meta], stride, dev }],
                            Some(HostHalf::Hb(h.place)),
                        )?,
                        Some((d, ix)) => store.add_layer_on_disk(
                            2 * index + g,
                            h.experts,
                            s[g],
                            DiskIn { place: h.place, dev, ram: store.host_slots(h.experts, h.place.bytes())?, read: d.reader(ix.members[g].clone()) },
                        )?,
                    }
                    HbGroup { stack, store_layer: Some(2 * index + g) }
                }
                _ => HbGroup { stack: HbStack::upload(h)?, store_layer: None },
            });
        }
        let cfg = moe_cfg(c);
        cfg.check()?;
        let shared = Shared { inter: c.moe.shared_inter, up: upload_packed(s_up)?, down: upload_packed(s_down)?, gate: None };
        let cached = groups.iter().flatten().any(|g| g.store_layer.is_some());
        Ok(NhMoeHb {
            cfg,
            router: upload_bytes(&router)?,
            bias: upload_bytes(&bias)?,
            shared,
            fc1: Mat::Packed(upload_packed(fc1)?),
            fc2: Mat::Packed(upload_packed(fc2)?),
            groups,
            group_of: upload_bytes(&host.group_of)?,
            local_of: upload_bytes(&host.local_of.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?,
            signs: upload_bytes(&host.signs.iter().map(|&s| s as u8).collect::<Vec<u8>>())?,
            host_group_of: host.group_of.clone(),
            host_local_of: host.local_of.clone(),
            store: cache.filter(|_| cached).map(|(_, s)| s.clone()),
            common: common.clone(),
        })
    }

    /// Whether any of the layer's experts are held in the computer's memory (a call reads the router's choices on the host to place them).
    pub fn cached(&self) -> bool {
        self.store.is_some()
    }

    /// The mixture-of-experts scratch every layer shares (`MoeScratch::for_shapes` over this layer's shape), and the Hummingbird scratch beside it, made by the first call.
    fn scratch<'a>(&'a self, e: &'a dyn NhHost) -> Result<(&'a MoeScratch, &'a HbScratch)> {
        if e.moe_scratch().get().is_none() {
            let c = &self.cfg;
            let shape = Shape {
                cfg: *c,
                twelve: false,
                shared: Some((self.shared.inter, self.shared.up.pack.cols as usize, self.shared.down.pack.rows as usize)),
                expert_rows: c.inter,
                sparse: true,
                slice: 1,
                hb: 0,
            };
            let _ = e.moe_scratch().set(MoeScratch::for_shapes(e.dev(), e.lib(), &[shape], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        if self.common.scratch.get().is_none() {
            let (n, k) = (e.chunk(), self.cfg.router.top_k);
            let cpu_pairs = n.min(CPU_TOKENS) * k * 8;
            let _ = self.common.scratch.set(HbScratch {
                kern: HbKern::load(e.dev())?,
                zr: Buf::new(n * self.cfg.hidden * 4)?,
                gids: Buf::new(n * k * 8)?,
                ways: [Buf::new(cpu_pairs)?, Buf::new(cpu_pairs)?],
                sids: Buf::new(cpu_pairs)?,
            });
        }
        Ok((sc, self.common.scratch.get().ok_or_else(|| Error("the Hummingbird scratch is not made".into()))?))
    }

    /// A group's pairs' two products: the plan over `via` (the pairs' numbers in the group where it is all on the GPU, their places in the cache's where not), then the first product and the second.
    ///
    /// # Safety
    /// As `experts`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn products(&self, lib: &Lib, sc: &MoeScratch, hs: &HbScratch, g: usize, via: u64, pairs: usize, split: usize, s: Stream) -> Result<()> {
        let (c, group) = (&self.cfg, self.groups[g].as_ref().ok_or_else(|| Error(format!("no group {g} of experts in the layer")))?);
        lib.moe_way(via, pairs as i64, group.stack.experts as i64, sc.plan.ptr, s)?;
        hs.kern.up(s, g as u32 + 2, &group.stack, hs.zr.ptr, sc.plan.ptr, via, sc.a.ptr, group.stack.experts, c.inter, c.hidden, c.router.top_k, pairs)?;
        hs.kern.down(s, g as u32 + 2, &group.stack, sc.a.ptr, sc.plan.ptr, via, sc.part.ptr, group.stack.experts, c.inter, c.hidden, split, pairs)
    }

    /// The experts' weighted sum of `t` tokens into y [t, hidden]: z' = R z, then each group's pairs (the first product, the second), then the sum and R^T. `route` has made `ids` and `wts`; `z` [t, hidden] is the experts'
    /// input. Returns whether the shared expert's MLP from `h` was queued with them (where the layer is in the cache, beside the host's work: the caller adds its result), else it is not.
    ///
    /// The groups that are all on the GPU run on their pairs' numbers in them (`hb_group_ids`). With a cache a call of at most `CPU_TOKENS` tokens in a mode that has the host compute pairs is `ExpertStore::run_groups`: one
    /// wait for the layer, both groups' misses planned together, the pairs' places sent for each group, the host's pairs computed from the host tier (z' read back, fp32) while the GPU runs the products, their weighted
    /// sums added in `hb_sum_rot` before the rotation back. Any other call of the cache is `run`, a group at a time: the misses copied, in waves where a group has more experts in the call than places.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write; the stream is the one the layer's other calls are on.
    #[allow(clippy::too_many_arguments)]
    unsafe fn experts(&self, dev: &Dev, lib: &Lib, sc: &MoeScratch, hs: &HbScratch, h: u64, z: u64, t: usize, y: u64) -> Result<bool> {
        let (c, s) = (&self.cfg, dev.stream);
        let (k, pairs) = (c.router.top_k, t * c.router.top_k);
        let split = hs.kern.split(pairs).min(sc.part.bytes / (pairs.max(1) * c.hidden * 4));
        if split == 0 {
            return err(format!(
                "{pairs} pairs of {} outputs need {} bytes of partial sums, where {} are given",
                c.hidden,
                pairs * c.hidden * 4,
                sc.part.bytes
            ));
        }
        let sum = |ids: u64, extra: u64| {
            hs.kern.sum_rot(s, sc.part.ptr, sc.wts.ptr, if ids != 0 { ids } else { sc.ids.ptr }, self.signs.ptr, y, extra, t, c.experts, k, c.hidden, split)
        };
        hs.kern.rot_in(s, z, c.hidden, self.signs.ptr, hs.zr.ptr, t, c.hidden)?;
        let layers: [Option<usize>; 2] = [0, 1].map(|g| self.store.as_ref().and_then(|_| self.groups[g].as_ref()?.store_layer));
        for g in (0..2).filter(|&g| self.groups[g].is_some() && layers[g].is_none()) {
            hs.kern.group_ids(s, sc.ids.ptr, self.group_of.ptr, self.local_of.ptr, g, hs.gids.ptr, pairs, c.experts)?;
            self.products(lib, sc, hs, g, hs.gids.ptr, pairs, split, s)?;
        }
        let Some(store) = self.store.as_ref().filter(|_| layers.iter().any(Option::is_some)) else {
            sum(0, 0)?;
            return Ok(false);
        };
        let groups = Groups { layers, group_of: &self.host_group_of, local_of: &self.host_local_of };
        if store.serves_groups(&groups, t, pairs) {
            let call = GroupsCall {
                ids: sc.ids.ptr,
                wts: sc.wts.ptr,
                z: hs.zr.ptr,
                ways: [hs.ways[0].ptr, hs.ways[1].ptr],
                sum_ids: hs.sids.ptr,
                part: sc.cpu.ptr,
                tokens: t,
                pairs,
            };
            let mut queued = false;
            store.run_groups(&groups, dev, &call, &mut |stage, g, hy| match stage {
                Stage::Products => {
                    // (after the wait, so that the shared expert runs on the GPU while the host computes its pairs)
                    if !std::mem::replace(&mut queued, true) {
                        shared_mlp_with(c, Some(&self.shared), lib, sc, s, h, t)?;
                    }
                    if hy.gpu {
                        self.products(lib, sc, hs, g, hs.ways[g].ptr, pairs, split, s)?;
                    }
                    Ok(())
                }
                Stage::Sum => sum(hy.ids, hy.extra),
            })?;
            return Ok(queued);
        }
        shared_mlp_with(c, Some(&self.shared), lib, sc, s, h, t)?;
        for (g, layer) in layers.iter().enumerate().filter_map(|(g, l)| l.map(|l| (g, l))) {
            hs.kern.group_ids(s, sc.ids.ptr, self.group_of.ptr, self.local_of.ptr, g, hs.gids.ptr, pairs, c.experts)?;
            let call = Call { ids: hs.gids.ptr, wts: sc.wts.ptr, z, way: sc.way.ptr, part: sc.cpu.ptr, tokens: t, pairs };
            store.run(layer, dev, &call, &mut |stage, _, _, _| {
                if stage == Stage::Products {
                    self.products(lib, sc, hs, g, sc.way.ptr, pairs, split, s)
                } else {
                    Ok(())
                }
            })?;
        }
        sum(0, 0)?;
        Ok(true)
    }
}

impl MoeBlock for NhMoeHb {
    /// `NhMoe::forward`'s launches with the experts' products of this module: `fc1_latent_proj` of `h` into the engine's `mix` buffer, the router, the experts' weighted sum of it into `xc` (rotated and rotated back), the shared
    /// expert's MLP (queued with the experts' products where the layer is in the cache), `fc2_latent_proj` of the sum into `out`, the shared expert added. With every expert on the GPU all launches are on the engine's stream and none reads a result, so a decode
    /// step captures into the graph; with a cache the choices are read on the host to place the experts (`ExpertStore::run_groups`, or `run` for each width a layer has) and the step is not captured.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let (c, s) = (&self.cfg, e.dev().stream);
        let need = n * c.hidden * 2;
        if e.mix().bytes < need || e.xc().bytes < need {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.hidden));
        }
        let (sc, hs) = self.scratch(e)?;
        e.lin(&self.fc1, h, n, e.mix().ptr)?;
        route_with(c, &self.router, Some(&self.bias), None, sc, s, h, c.router_hidden, n)?;
        let queued = self.experts(e.dev(), e.lib(), sc, hs, h, e.mix().ptr, n, e.xc().ptr)?;
        e.lin(&self.fc2, e.xc().ptr, n, out)?;
        if queued {
            shared_add_with(Some(&self.shared), sc, s, n, out)
        } else {
            add_shared_with(c, Some(&self.shared), e.lib(), sc, s, h, n, out)
        }
    }

    fn graphable(&self) -> bool {
        self.store.is_none()
    }

    fn bytes(&self) -> usize {
        self.router.bytes
            + self.bias.bytes
            + self.shared.up.bytes
            + self.shared.down.bytes
            + self.fc1.bytes()
            + self.fc2.bytes()
            + self.groups.iter().flatten().map(|g| g.stack.bytes).sum::<usize>()
            + self.group_of.bytes
            + self.local_of.bytes
            + self.signs.bytes
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        [&self.fc1, &self.fc2]
            .into_iter()
            .filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_gpu::codebook::{decode_codebook_rows, pack_codebook_rows};

    const SRC: &str = include_str!("../kernels/moe_hb.cu");

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn bf16(x: f32) -> u16 {
        ((x.to_bits() + 0x7FFF + ((x.to_bits() >> 16) & 1)) >> 16) as u16
    }

    fn matrix(rng: &mut Rng, bits: u32, rows: usize, cols: usize) -> CbRows {
        let codes: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
        let scales: Vec<u16> = (0..rows * cols / 32).map(|_| bf16(((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.3) * 0.05)).collect();
        let book: Vec<u16> = (0..rows << bits).map(|_| bf16((rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)).collect();
        pack_codebook_rows(&codes, &scales, &book, bits, rows, cols).unwrap()
    }

    /// The layer with some of its pairs on the host is the layer all there, to a bound: the pairs of the experts that have places are summed as `hb_sum_rot` sums the GPU's (fp32, in the rotated basis), the pairs of the
    /// others are the host's, computed from the experts' bytes of the host tier in place and added as `extra` before R^T and the one rounding to bf16; against every pair summed alone. The bits of a pair are the same either
    /// way (the kernels of cpu_hb.rs make one set of bits), so what differs is the order of an fp32 sum seen through the rounding to bf16: all 1,280 outputs of the 5 tokens are the same to the bit here (the sum without the host's pairs differs in most of them), against a bound of 2e-4 of a token's norm.
    #[test]
    fn a_layer_with_pairs_on_the_host_is_the_layer_all_there_to_a_bound() {
        use crate::cpu_experts::{Kernel, Z};
        use crate::cpu_hb::{experts_with, HbPair};
        let (e, rows, cols, k, tokens) = (12usize, 256usize, 256usize, 4usize, 5usize);
        let mut rng = Rng(0xFEED_5EED);
        let rand = |rng: &mut Rng| (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
        let places = [HbPlace::new(2, rows, cols).unwrap(), HbPlace::new(3, rows, cols).unwrap()];
        // (every third expert at three bits)
        let group: Vec<usize> = (0..e).map(|x| (x % 3 == 0) as usize).collect();
        let hosts: Vec<HbHost> = (0..2)
            .map(|g| {
                let n = group.iter().filter(|&&x| x == g).count();
                HbHost::of(
                    places[g],
                    &(0..n).map(|_| (matrix(&mut rng, g as u32 + 2, rows, cols), matrix(&mut rng, g as u32 + 2, rows, cols))).collect::<Vec<_>>(),
                )
                .unwrap()
            })
            .collect();
        let local: Vec<usize> = (0..e).map(|x| group[..x].iter().filter(|&&y| y == group[x]).count()).collect();
        let experts: Vec<HbExpert> = (0..e)
            .map(|x| {
                places[group[x]]
                    .expert(
                        &hosts[group[x]].data[local[x] * places[group[x]].data..][..places[group[x]].data],
                        &hosts[group[x]].meta[local[x] * places[group[x]].meta..][..places[group[x]].meta],
                    )
                    .unwrap()
            })
            .collect();
        let signs: Vec<i8> = (0..cols).map(|_| if rng.next() & 1 == 0 { 1 } else { -1 }).collect();
        let z: Vec<f32> = (0..tokens * cols).map(|_| f32::from_bits((bf16(rand(&mut rng) * 2.0 - 1.0) as u32) << 16)).collect();
        let wts: Vec<f32> = (0..tokens * k).map(|_| 0.3 + rand(&mut rng)).collect();
        let ids: Vec<usize> = (0..tokens).flat_map(|t| (0..k).map(move |c| (t * 5 + c * 3) % e)).collect();
        let pairs: Vec<HbPair> = ids.iter().enumerate().map(|(j, &x)| HbPair { expert: &experts[x], token: j / k, weight: wts[j] }).collect();
        let mut rows_out = vec![0f32; pairs.len() * cols];
        experts_with(Kernel::Scalar, Z::F32(&z), &pairs, &mut rows_out, 1).unwrap();
        // R^T as hb_sum_rot does it: signs o (scale H v), rounded to bf16 once
        let unrotate = |v: &[f32]| -> Vec<f32> {
            let mut w = v.to_vec();
            let mut h = 1;
            while h < w.len() {
                for i in (0..w.len()).step_by(2 * h) {
                    for j in i..i + h {
                        let (a, b) = (w[j], w[j + h]);
                        w[j] = a + b;
                        w[j + h] = a - b;
                    }
                }
                h *= 2;
            }
            let scale = (1.0 / (w.len() as f64).sqrt()) as f32;
            w.iter().zip(&signs).map(|(&x, &s)| f32::from_bits((bf16(s as f32 * (scale * x)) as u32) << 16)).collect()
        };
        let sum_of = |on: &dyn Fn(usize) -> bool, t: usize| -> Vec<f32> {
            let mut v = vec![0f32; cols];
            for c in 0..k {
                let j = t * k + c;
                if on(j) {
                    v.iter_mut().zip(&rows_out[j * cols..(j + 1) * cols]).for_each(|(a, b)| *a += *b);
                }
            }
            v
        };
        let (mut worst, mut differ, mut norm, mut missing) = (0f64, 0usize, 0f64, 0usize);
        for t in 0..tokens {
            let all = unrotate(&sum_of(&|_| true, t));
            // the host keeps the pairs of the experts that have no place: here, every pair whose expert is odd or 4
            let kept = |j: usize| ids[j] % 2 == 1 || ids[j] == 4;
            assert!((0..k).any(|c| kept(t * k + c)) || t > 0);
            // (the host's pairs by the best kernel, on the experts in place)
            let mine: Vec<usize> = (0..k).map(|c| t * k + c).filter(|&j| kept(j)).collect();
            let host_pairs: Vec<HbPair> = mine.iter().map(|&j| pairs[j]).collect();
            let mut host_rows = vec![0f32; host_pairs.len() * cols];
            experts_with(Kernel::best(), Z::F32(&z), &host_pairs, &mut host_rows, 2).unwrap();
            let mut extra = vec![0f32; cols];
            host_rows.chunks_exact(cols).for_each(|r| extra.iter_mut().zip(r).for_each(|(a, b)| *a += *b));
            let mut v = sum_of(&|j| !kept(j), t);
            v.iter_mut().zip(&extra).for_each(|(a, b)| *a += *b);
            let got = unrotate(&v);
            // (without the host's sums the layer is another: the pairs matter)
            let without = unrotate(&sum_of(&|j| !kept(j), t));
            missing += without.iter().zip(&all).filter(|(a, b)| a != b).count();
            for (a, b) in got.iter().zip(&all) {
                differ += (a != b) as usize;
                worst = worst.max((*a as f64 - *b as f64).abs());
                norm += (*b as f64) * (*b as f64);
            }
        }
        let norm = norm.sqrt() / (tokens as f64).sqrt();
        assert!(worst <= 2e-4 * norm, "the layer with pairs on the host differs by {worst:.3e} at most, against a bound of {:.3e} (2e-4 of a token's norm {norm:.3e}); {differ} of {} outputs differ", 2e-4 * norm, tokens * cols);
        assert!(differ < tokens * cols / 4, "{differ} of {} outputs differ", tokens * cols);
        assert!(missing > tokens * cols / 2, "the host's pairs changed {missing} of {} outputs", tokens * cols);
    }

    /// The bytes of a place at the real shape: 2,494,464 at three bits, 1,763,328 at two (the brief's), the parts in order and to 16 bytes; the mean of hb40 and what a layer of the recipe takes.
    #[test]
    fn a_place_and_its_bytes() {
        let (p3, p2) = (HbPlace::new(3, 2688, 1024).unwrap(), HbPlace::new(2, 2688, 1024).unwrap());
        assert_eq!((p3.bytes(), p2.bytes()), (2_494_464, 1_763_328));
        assert_eq!((p3.data, p3.meta, p2.data, p2.meta), (2 * 2688 * 384, 2 * 2688 * 80, 2 * 2688 * 256, 2 * 2688 * 72));
        assert_eq!((p3.down_codes, p3.up_book, p3.down_scales, p3.down_book), (2688 * 384, 2688 * 64, 2688 * 64 + 2688 * 16, 2 * 2688 * 64 + 2688 * 16));
        let c = crate::nemotron::tests::parse();
        assert_eq!((expert_bytes_at(&c, 3).unwrap(), expert_bytes_at(&c, 2).unwrap()), (2_494_464, 1_763_328));
        // 205 of 512 at three bits: a layer 1,052,706,816 bytes, 42.1 GB over 40 layers
        let layer = 205 * 2_494_464 + 307 * 1_763_328;
        assert_eq!(layer, 1_052_706_816);
        assert_eq!(expert_bytes_mean(&c).unwrap(), layer / 512);
        // the expert cache's plan at the real shape counts the mean expert: a 24 GB card's 13 GiB buys 169 places a layer of 40 layers of 512, the host tier holds the 42.1 GB
        let plan = crate::moe_nh::cache_plan(&c, Layout::Hummingbird, 13 << 30, 22).unwrap().unwrap();
        let mean = expert_bytes_mean(&c).unwrap();
        assert_eq!((plan.layers, plan.experts, plan.expert_bytes, plan.slots), (40, 512, mean, ((13u64 << 30) / (40 * mean)) as usize));
        assert_eq!((plan.slots, plan.host_bytes()), (169, 40 * 512 * mean));
        assert!((42.0e9..42.2e9).contains(&(plan.host_bytes() as f64)));
        assert!(HbPlace::new(4, 8, 32).is_err() && HbPlace::new(3, 0, 32).is_err() && HbPlace::new(3, 8, 48).is_err());
        // a shape whose parts are not whole 16 bytes is padded: 5 rows of 128 values at 2 bits, 160 bytes of codes, 40 of scales (48) and 40 of codebooks (48)
        let q = HbPlace::new(2, 5, 128).unwrap();
        assert_eq!((q.down_codes, q.up_book, q.down_scales, q.down_book, q.data, q.meta), (160, 48, 96, 144, 320, 192));
    }

    /// An expert's matrices are placed where the place says, the stack's experts end to end; read back from the parts, every weight is the matrix's.
    #[test]
    fn experts_are_placed_end_to_end() {
        let mut rng = Rng(0x5EED);
        for (bits, rows, cols) in [(3u32, 70usize, 128usize), (2, 5, 128), (2, 33, 256), (3, 64, 1024)] {
            let place = HbPlace::new(bits, rows, cols).unwrap();
            let pairs: Vec<(CbRows, CbRows)> = (0..3).map(|_| (matrix(&mut rng, bits, rows, cols), matrix(&mut rng, bits, rows, cols))).collect();
            let h = HbHost::of(place, &pairs).unwrap();
            assert_eq!((h.data.len(), h.meta.len()), (3 * place.data, 3 * place.meta));
            for (k, (up, down)) in pairs.iter().enumerate() {
                let (d, m) = (&h.data[k * place.data..(k + 1) * place.data], &h.meta[k * place.meta..(k + 1) * place.meta]);
                let (c, s, b) = cb_strides(bits, cols);
                for (mat, codes, scales, book) in [(up, 0, 0, place.up_book), (down, place.down_codes, place.down_scales, place.down_book)] {
                    let back = CbRows::new(
                        bits,
                        rows,
                        cols,
                        d[codes..codes + rows * c].to_vec(),
                        m[scales..scales + rows * s].to_vec(),
                        m[book..book + rows * b].to_vec(),
                    )
                    .unwrap();
                    assert_eq!(&back, mat);
                    let (mut w, mut v) = (vec![0f32; rows * cols], vec![0f32; rows * cols]);
                    assert!(decode_codebook_rows(&back, &mut w) && decode_codebook_rows(mat, &mut v) && w == v);
                }
                // the parts are 16-byte aligned and the padding between them zero
                assert!([place.down_codes, place.up_book, place.down_scales, place.down_book, place.data, place.meta].iter().all(|x| x % 16 == 0));
                assert!(d[rows * c..place.down_codes].iter().all(|&x| x == 0));
            }
            // a matrix of another width or shape is refused
            let other = matrix(&mut rng, 5 - bits, rows, cols);
            assert!(HbHost::of(place, &[(other.clone(), other)]).is_err());
            assert!(HbHost::of(place, &[(matrix(&mut rng, bits, rows + 1, cols), matrix(&mut rng, bits, rows + 1, cols))]).is_err());
        }
    }

    /// The places a layer's budget buys are shared by the widths in the proportion of their experts, each at least one and at most all of its own, the sum the budget where it can be.
    #[test]
    fn places_are_split_by_the_share_of_the_experts() {
        assert_eq!(split_slots(100, [307, 205]), [60, 40]);
        assert_eq!(split_slots(122, [307, 205]), [73, 49]);
        assert_eq!(split_slots(512, [307, 205]), [307, 205]);
        assert_eq!(split_slots(1000, [307, 205]), [307, 205]);
        assert_eq!(split_slots(5, [10, 6]), [3, 2]);
        assert_eq!(split_slots(3, [10, 6]), [2, 1]);
        assert_eq!(split_slots(1, [10, 6]).iter().sum::<usize>(), 2, "at least a place to each width");
        assert_eq!(split_slots(8, [16, 0]), [8, 0]);
        assert_eq!(split_slots(20, [0, 16]), [0, 16]);
        assert_eq!(split_slots(5, [0, 0]), [0, 0]);
        for slots in 2..=48 {
            let s = split_slots(slots, [30, 18]);
            assert!(s[0] >= 1 && s[1] >= 1 && s[0] <= 30 && s[1] <= 18 && s[0] + s[1] == slots, "{slots}: {s:?}");
        }
    }

    /// The types of the parameters of a kernel of kernels/moe_hb.cu in the source's order, one letter each: `p` a pointer, `i` an int, `l` a long long, `f` a float.
    fn params(name: &str) -> String {
        let head = format!("extern \"C\" __global__ void __launch_bounds__(256) {name}(");
        let at = SRC.find(&head).unwrap_or_else(|| panic!("no kernel {name} in kernels/moe_hb.cu")) + head.len();
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

    /// What the launches in this file pass is what the kernels declare: each kernel is in the source, its parameters in the types and the number the launcher gives (device addresses, ints, the strides as long long, the rotation's
    /// scale as a float), and the sizes the launchers share with the kernels are the source's.
    #[test]
    fn kernels_and_signatures_are_in_the_source() {
        // (the launches: rot_in z zs signs zr n scale; group_ids ids group_of local_of group out pairs E; up and down the three arrays and their strides; sum_rot part wts ids signs y extra E k hidden split scale)
        for (name, want) in [
            ("hb_rot_in", "pipp".to_string() + "if"),
            ("hb_group_ids", "pppipii".to_string()),
            ("hb_up2", "pppppplllpiiii".to_string()),
            ("hb_up3", "pppppplllpiiii".to_string()),
            ("hb_down2", "pppppplllpiiii".to_string()),
            ("hb_down3", "pppppplllpiiii".to_string()),
            ("hb_sum_rot", "pppppp".to_string() + "iiiif"),
        ] {
            assert_eq!(params(name), want, "{name}");
        }
        assert!(SRC.contains("tile * 64 + warp * 8"), "the first product's tile of {UP_ROWS} neurons");
        assert_eq!((UP_ROWS, DOWN_OUT, ROT_MIN, ROT_MAX), (64, 1024, 128, 1024));
    }

    /// The CPU half reads a group's bytes where they are: glyd-gpu's pack is cpu_hb's packer's bytes, and the pairs of experts computed from the views over a host copy's places (`HbPlace::expert`) are, bit for bit, those computed from
    /// the matrices apart; the computation is in the rotated basis (its input z' fp32), so the caller's sum and R^T are the GPU's `hb_sum_rot`.
    #[test]
    fn the_cpu_half_reads_the_hosts_bytes_in_place() {
        use crate::cpu_experts::Z;
        use crate::cpu_hb::{experts_with, CbRows as CpuRows, HbPair};
        let mut rng = Rng(0xC0DE);
        for bits in [2u32, 3] {
            let (rows, cols) = (192usize, 256usize);
            let place = HbPlace::new(bits, rows, cols).unwrap();
            let mut pairs = Vec::new();
            for _ in 0..3 {
                let one = |rng: &mut Rng| {
                    let codes: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
                    let scales: Vec<u16> = (0..rows * cols / 32).map(|_| bf16(((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.4) * 0.06)).collect();
                    let book: Vec<u16> = (0..rows << bits).map(|_| bf16((rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)).collect();
                    let theirs = pack_codebook_rows(&codes, &scales, &book, bits, rows, cols).unwrap();
                    let mine = CpuRows::from_codes(bits, rows, cols, &codes, &scales, &book).unwrap();
                    assert!(
                        theirs.codes == mine.codes && theirs.scales == mine.scales && theirs.codebook == mine.codebook,
                        "{bits} bits: the CPU half's rows are not glyd-gpu's pack"
                    );
                    theirs
                };
                pairs.push((one(&mut rng), one(&mut rng)));
            }
            let host = HbHost::of(place, &pairs).unwrap();
            let z: Vec<f32> = (0..2 * cols).map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0).collect();
            let apart: Vec<(CpuRows, CpuRows)> = pairs
                .iter()
                .map(|(u, d)| {
                    (
                        CpuRows::new(bits, rows, cols, u.codes.clone(), u.scales.clone(), u.codebook.clone()).unwrap(),
                        CpuRows::new(bits, rows, cols, d.codes.clone(), d.scales.clone(), d.codebook.clone()).unwrap(),
                    )
                })
                .collect();
            let experts_apart: Vec<HbExpert> = apart.iter().map(|(u, d)| HbExpert::new(u.view(), d.view()).unwrap()).collect();
            let experts_in_place: Vec<HbExpert> = (0..3)
                .map(|k| place.expert(&host.data[k * place.data..(k + 1) * place.data], &host.meta[k * place.meta..(k + 1) * place.meta]).unwrap())
                .collect();
            let tokens = [(0usize, 0usize, 1.5f32), (1, 1, 0.5), (0, 2, 2.0), (1, 0, 1.0)];
            let run = |e: &[HbExpert]| {
                let ps: Vec<HbPair> = tokens.iter().map(|&(t, x, w)| HbPair { expert: &e[x], token: t, weight: w }).collect();
                let mut out = vec![0f32; ps.len() * cols];
                let n = experts_with(crate::cpu_experts::Kernel::Scalar, Z::F32(&z), &ps, &mut out, 1).unwrap();
                (out, n)
            };
            let (a, b) = (run(&experts_apart), run(&experts_in_place));
            assert!(
                a.0.iter().zip(&b.0).all(|(x, y)| x.to_bits() == y.to_bits()) && a.1 == b.1 && a.1.iter().any(|&n| n > 0),
                "{bits} bits: {:?} against {:?}",
                a.1,
                b.1
            );
            assert!(place.expert(&host.data[..place.data - 1], &host.meta[..place.meta]).is_err());
        }
    }

    /// The experts read one at a time from the shards are the layer read whole, byte for byte, in the tiny model's directory: every expert of its three layers, both widths, with the groups and numbers `index_layer` says (from
    /// glyd.json alone), in a directory of one shard and in one of shards of 24 KB, where an expert's six tensors are in two files now and then (two reads); the padding between the parts is zero; a place of another size,
    /// an expert that is not in the layer, and a directory that lacks a tensor are refused; the reader of a group maps its numbers to the layer's.
    #[cfg(feature = "dev")]
    #[test]
    fn experts_read_one_at_a_time_are_the_layer_read_whole() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-hbdisk-{}", std::process::id()));
        for (shards, shard_bytes) in [("one shard", 1u64 << 30), ("shards of 24 KB", 24 << 10)] {
            let (out, c) = crate::hbpack::tests::packed_toy(&dir, shard_bytes, 40);
            let ck = Checkpoint::open(&out).unwrap();
            let (_, manifest) = v7_manifest(&out).unwrap().unwrap();
            assert_eq!(ck.files.len() > 1, shard_bytes < 1 << 20, "{shards}: {} files", ck.files.len());
            let mut two_reads = 0;
            for l in [1usize, 3, 5] {
                let host = read_layer(&ck, &manifest, &c, l).unwrap();
                let ix = index_layer(&manifest, &c, l).unwrap();
                assert_eq!((&ix.group_of, &ix.local_of), (&host.group_of, &host.local_of), "{shards}, layer {l}");
                assert_eq!(ix.counts(), [0, 1].map(|g| host.groups[g].as_ref().map_or(0, |h| h.experts)));
                assert!(ix.counts()[0] > 0 && ix.counts()[1] > 0);
                let disk = Arc::new(HbDisk::open(&ck, &c, l, &ix).unwrap());
                two_reads += disk.experts.iter().filter(|e| e.segs.iter().any(|s| s.file != e.segs[0].file)).count();
                // (in one shard an expert's six tensors are one run of the file, as the writer sorts them by name: one read of exactly the bytes of its place; if the writer's order changed, every read would be three)
                if shard_bytes > 1 << 20 {
                    for (e, ex) in disk.experts.iter().enumerate() {
                        let mut segs = ex.segs;
                        segs.sort_by_key(|s| s.at);
                        assert!(
                            segs.windows(2).all(|w| w[0].at + w[0].len as u64 == w[1].at),
                            "{shards}, layer {l}: the tensors of expert {e} are not one run of the file"
                        );
                        assert_eq!(segs.iter().map(|s| s.len as u64).sum::<u64>(), disk.places[ex.group as usize].bytes());
                    }
                }
                for e in 0..c.moe.experts {
                    let (g, k) = (ix.group_of[e] as usize, ix.local_of[e] as usize);
                    let h = host.groups[g].as_ref().unwrap();
                    // (a place that was not zeroed: whatever the read does not write is the read's to write)
                    let (mut d, mut m) = (vec![0xEEu8; h.place.data], vec![0xEEu8; h.place.meta]);
                    disk.read(e, &mut d, &mut m).unwrap();
                    assert!(
                        d == h.data[k * h.place.data..(k + 1) * h.place.data] && m == h.meta[k * h.place.meta..(k + 1) * h.place.meta],
                        "{shards}, layer {l}, expert {e}"
                    );
                }
                // the reader of a group: its expert k is the layer's expert members[k]
                for g in 0..2 {
                    let h = host.groups[g].as_ref().unwrap();
                    let read = disk.reader(ix.members[g].clone());
                    let k = h.experts / 2;
                    let (mut d, mut m) = (vec![0u8; h.place.data], vec![0u8; h.place.meta]);
                    read(k, &mut d, &mut m).unwrap();
                    assert!(
                        d == h.data[k * h.place.data..(k + 1) * h.place.data] && m == h.meta[k * h.place.meta..(k + 1) * h.place.meta],
                        "{shards}, layer {l}, group {g}'s expert {k}"
                    );
                    assert!(read(h.experts, &mut d, &mut m).is_err());
                }
                // refused: a place of another size, an expert past the layer's
                let h = host.groups[1].as_ref().unwrap();
                assert!(disk.read(0, &mut vec![0u8; h.place.data + 16], &mut vec![0u8; h.place.meta]).is_err());
                assert!(disk.read(c.moe.experts, &mut vec![0u8; h.place.data], &mut vec![0u8; h.place.meta]).is_err());
            }
            // (the shards of 24 KB put some experts' tensors in two files; the one shard none)
            assert_eq!(two_reads > 0, shard_bytes < 1 << 20, "{shards}: {two_reads} experts in two files");
            // a layer of the directory that is not a mixture-of-experts layer has no experts to find
            assert!(HbDisk::open(&ck, &c, 0, &index_layer(&manifest, &c, 1).unwrap()).is_err());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
