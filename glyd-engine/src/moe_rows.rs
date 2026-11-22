//! Sparse experts (Linear GLY-130): a relu squared expert's second product for the neurons that are not zero. After relu squared 92% of an expert's 2,688 activations are exactly 0 for a token (the same in decode:
//! benchmarks/weights/offload-probe-2026-10-10), and a zero activation needs no column of the down matrix, so an expert's output is its first product in full, then the sum over the active neurons n of a_n times down's
//! column n: the same result as the dense product up to the order of a float sum. The block layouts cannot skip them (a step of 64 rows by 16 columns holds an active neuron in nearly every one), so a down matrix is
//! held transposed in a row layout (glyd-gpu's `pack_rows`, glyd-v7 entries `i6r`, `i5r`, `i4r`): a neuron's 1,024 weights are a row of their own, the very codes and scales the block layout of that transposed matrix
//! makes, at 4 bits 512 bytes of codes and 64 of scales. The kernels (kernels/moe_rows.cu) list a pair's active neurons and add their rows, then sum a token's pairs (weighted).
//!
//! `Down` is the layer's second matrix, in either layout: `Block` (the library's grouped products, the dense path: a directory saved before the row layouts, or `GLYD_ENGINE_DENSE_DOWN` in a development build)
//! or `Rows`. A layer with `Rows` runs the sparse path for every count of tokens, a decode step's 1 and a prompt chunk's thousands: with no block copy of the matrix there is no dense product for a prompt (the
//! block layout of down, [1,024 outputs x 2,688 neurons], groups 32 neurons to a scale, the row layout's 32 outputs: one cannot be made of the other without rounding again). The roundings are the dense
//! path's: the weight is the bf16 of (code - bias) times the run's scale, the product with the activation (also bf16) is exact in fp32, the sums are fp32, the result is rounded to bf16 once.
//!
//! The expert cache (src/offload.rs) holds the rows as it holds the block layout: a place is an expert's first matrix and its rows (`RowStack::pooled`), the copy moves both (the same bytes as the block layout's: a row
//! layout of the transpose takes as many as the block layout of the matrix), the sparse kernels read the rows at the places of the pairs' experts (`RowsKern::run`'s `via`) and the sum is of the router's choices.
//!
//! The CPU kernel (src/cpu_experts.rs) computes the pairs of experts that are not in the cache's places from these bytes: the host tier's own second matrices, and for a pair a first product in full (the first matrices in the
//! row layout too, a copy the store holds), then the rows of the neurons that are not zero; its 1,024-wide fp32 sums of a token's pairs are `moe_rows_sum`'s `extra`, added before the one rounding to bf16.

use crate::dev::{Buf, Dev, Kernel};
use crate::ffi::{err, Result, Stream};
use crate::launch;
use crate::moe::{Act, ExpertStack, MoeCfg};
use glyd_gpu::pack::{rows_bits, rows_stride, Tiered};

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// Outputs a block of `moe_down_rows` takes, the neurons it lists at most, and the blocks it asks for at most beyond one a pair (kernels/moe_rows.cu defines the first two: a test holds them equal).
pub const ROWS_OUT: usize = 1024;
pub const ROWS_MAX_INTER: usize = 4096;
const SPLIT_MAX: usize = 8;
/// The blocks a launch aims at, a multiple of the GPU's multiprocessors, and the most it may add to the pairs' (a decode step's few pairs are shared among several blocks each).
const BLOCKS_PER_SM: usize = 2;
const BLOCKS_MOST: usize = 512;

const _: () = assert!(ROWS_MAX_INTER >= 2688, "Nemotron 3 Super's intermediate size");

/// The blocks a pair's neurons are shared among on a GPU of `sms` multiprocessors (`RowsKern::split`; src/moe_hb.rs's second product shares it).
pub fn split_for(sms: usize, pairs: usize) -> usize {
    if let Some(n) = crate::tune!("GLYD_ENGINE_SPARSE_SPLIT").and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0) {
        return n.min(SPLIT_MAX);
    }
    (BLOCKS_PER_SM * sms).min(BLOCKS_MOST).div_ceil(pairs.max(1)).clamp(1, SPLIT_MAX)
}

/// A row layout's bytes for an expert of `rows` neurons of `cols` outputs: its codes and its scales.
pub fn row_strides(tag: u32, rows: usize, cols: usize) -> Result<(u64, u64)> {
    let Some(bits) = rows_bits(tag) else {
        return err(format!("a layout of tag {tag}: sparse experts are held in the row layouts (six, five and four bits) alone"));
    };
    if rows == 0 || cols == 0 || !cols.is_multiple_of(32) {
        return err(format!("an expert of {rows} neurons of {cols} outputs: outputs in runs of 32 are wanted"));
    }
    Ok(((rows * rows_stride(bits, cols)) as u64, (rows * cols / 32 * 2) as u64))
}

/// E experts' packs in a row layout end to end on the host: the codes and the scales as the kernels read them.
pub struct RowsJoined {
    pub tag: u32,
    pub experts: usize,
    /// neurons (the pack's rows) and outputs (its columns)
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
}

impl RowsJoined {
    /// The packs of the experts, one each, joined: all in one row layout, of one shape.
    pub fn of(packs: &[&Tiered]) -> Result<RowsJoined> {
        let Some(p0) = packs.first() else { return err("no experts") };
        let tag = p0.tiers[0] >> 24;
        let (data_bytes, scale_bytes) = row_strides(tag, p0.rows, p0.cols)?;
        let (mut data, mut scales) = (Vec::with_capacity(packs.len() * data_bytes as usize), Vec::with_capacity(packs.len() * scale_bytes as usize));
        for (i, p) in packs.iter().enumerate() {
            if p.tiers != p0.tiers || (p.rows, p.cols) != (p0.rows, p0.cols) || p.data.len() as u64 != data_bytes || p.blocks.len() as u64 != scale_bytes {
                return err(format!(
                    "expert {i}: a pack of {} x {} (words {:x?}, {} + {} bytes), where expert 0's is {} x {} ({:x?}, {data_bytes} + {scale_bytes})",
                    p.rows,
                    p.cols,
                    p.tiers,
                    p.data.len(),
                    p.blocks.len(),
                    p0.rows,
                    p0.cols,
                    p0.tiers
                ));
            }
            data.extend_from_slice(&p.data);
            scales.extend_from_slice(&p.blocks);
        }
        Ok(RowsJoined { tag, experts: packs.len(), rows: p0.rows, cols: p0.cols, data, scales })
    }
}

/// E experts' matrices (the transpose of an expert's down: `rows` neurons of `cols` outputs) in one row layout, end to end in device memory: expert e's codes are at `e * data_stride`, its scales at `e * scale_stride`.
pub struct RowStack {
    pub experts: usize,
    /// The experts the buffers have room for: `experts`, or fewer in a stack made by `pooled`.
    pub slots: usize,
    pub rows: usize,
    pub cols: usize,
    pub tag: u32,
    pub bits: u32,
    pub data_stride: u64,
    pub scale_stride: u64,
    pub bytes: usize,
    data: Buf,
    scales: Buf,
}

impl RowStack {
    /// The experts of `j` copied to the GPU.
    pub fn upload(j: &RowsJoined) -> Result<RowStack> {
        let (data, scales) = (Buf::new(j.data.len())?, Buf::new(j.scales.len())?);
        data.write(0, &j.data)?;
        scales.write(0, &j.scales)?;
        RowStack::from_buffers(j.tag, j.experts, j.rows, j.cols, data, scales)
    }

    /// The stack of experts whose codes and scales are in device memory already.
    pub fn from_buffers(tag: u32, experts: usize, rows: usize, cols: usize, data: Buf, scales: Buf) -> Result<RowStack> {
        let (data_stride, scale_stride) = row_strides(tag, rows, cols)?;
        if data.bytes as u64 != data_stride * experts as u64 || scales.bytes as u64 != scale_stride * experts as u64 {
            return err(format!(
                "{experts} experts of {rows} neurons of {cols} outputs in the layout of tag {tag} take {} + {} bytes: {} + {} are given",
                data_stride * experts as u64,
                scale_stride * experts as u64,
                data.bytes,
                scales.bytes
            ));
        }
        let bits = rows_bits(tag).expect("row_strides took the tag");
        let bytes = data.bytes + scales.bytes;
        Ok(RowStack { experts, slots: experts, rows, cols, tag, bits, data_stride, scale_stride, bytes, data, scales })
    }

    /// Room for `slots` experts' matrices [rows, cols] in a row layout, standing for a layer's `experts` (zeroed): `ExpertStack::pooled`'s for the row layouts. The sparse product reads the place of each pair's
    /// expert (`RowsKern::run`'s `via`), which is below `slots`.
    pub fn pooled(tag: u32, experts: usize, slots: usize, rows: usize, cols: usize) -> Result<RowStack> {
        let (data_stride, scale_stride) = row_strides(tag, rows, cols)?;
        if slots == 0 || slots > experts {
            return err(format!("room for {slots} experts standing for {experts}, in the layout of tag {tag}"));
        }
        let (data, scales) = (Buf::zeroed(slots * data_stride as usize)?, Buf::zeroed(slots * scale_stride as usize)?);
        let bits = rows_bits(tag).expect("row_strides took the tag");
        let bytes = data.bytes + scales.bytes;
        Ok(RowStack { experts, slots, rows, cols, tag, bits, data_stride, scale_stride, bytes, data, scales })
    }

    /// The addresses of the codes and of the scales (of the first expert's, or of the first place).
    pub fn buffers(&self) -> [u64; 2] {
        [self.data.ptr, self.scales.ptr]
    }
}

/// A layer's second matrix, E experts' [hidden, inter] in either layout.
pub enum Down {
    /// The block layout (the library's grouped products: the dense path).
    Block(ExpertStack),
    /// The row layout of the transpose [inter, hidden] (the sparse path).
    Rows(RowStack),
}

impl Down {
    /// Device bytes.
    pub fn bytes(&self) -> usize {
        match self {
            Down::Block(s) => s.bytes,
            Down::Rows(s) => s.bytes,
        }
    }

    /// The addresses of the codes and of the scales (of the first expert's, or of the first place) and the bytes of an expert's of each: what the expert cache copies an expert into.
    pub fn places(&self) -> ([u64; 2], [u64; 2]) {
        match self {
            Down::Block(s) => (s.buffers(), [s.data_stride, s.scale_stride]),
            Down::Rows(s) => (s.buffers(), [s.data_stride, s.scale_stride]),
        }
    }

    /// (experts, outputs, neurons) of the second matrix the stack holds.
    pub fn shape(&self) -> (usize, usize, usize) {
        match self {
            Down::Block(s) => (s.experts, s.rows, s.cols),
            Down::Rows(s) => (s.experts, s.cols, s.rows),
        }
    }

    /// Whether a layer of `cfg` can hold it (`rows_fit`, for a row layout).
    pub fn check(&self, cfg: &MoeCfg) -> Result<()> {
        match self {
            Down::Rows(r) => rows_fit(r.experts, r.rows, r.cols, cfg),
            Down::Block(_) => Ok(()),
        }
    }
}

/// Whether `experts` experts' matrices of `rows` neurons of `cols` outputs in a row layout are the second matrix of a layer of `cfg`: relu squared experts (their activations are mostly zero), the layer's number of
/// experts, neurons and outputs, a neuron list that fits the kernels' (at most `ROWS_MAX_INTER`), outputs in runs of 32.
pub fn rows_fit(experts: usize, rows: usize, cols: usize, cfg: &MoeCfg) -> Result<()> {
    if cfg.act != Act::Relu2 {
        return err(format!("a row layout for the second matrix of {:?} experts: it is for relu squared, whose activations are mostly zero", cfg.act));
    }
    if (experts, rows, cols) != (cfg.experts, cfg.inter, cfg.hidden) || cfg.inter > ROWS_MAX_INTER || !cfg.hidden.is_multiple_of(32) {
        return err(format!("a row layout of {experts} experts of {rows} neurons of {cols} outputs, where the layer has {} of {} neurons of {} outputs and the kernels take at most {ROWS_MAX_INTER} neurons, outputs in runs of 32", cfg.experts, cfg.inter, cfg.hidden));
    }
    Ok(())
}

/// The sparse product's kernels (kernels/moe_rows.cu) with typed launches.
pub struct RowsKern {
    down: [Kernel; 3],
    sum: crate::kt::moe_rows_sum,
    sms: usize,
}

impl RowsKern {
    pub fn load(dev: &Dev) -> Result<RowsKern> {
        Ok(RowsKern {
            down: [dev.kernel(kname!("moe_down_rows4"))?, dev.kernel(kname!("moe_down_rows5"))?, dev.kernel(kname!("moe_down_rows6"))?],
            sum: dev.typed()?,
            sms: dev.sms.max(1) as usize,
        })
    }

    /// The blocks a pair's neurons are shared among, for `pairs` pairs: enough that a launch has about 2 blocks a multiprocessor (a decode step's 22 pairs would fill 22 of them), at most 8 and at most
    /// `BLOCKS_MOST` more than the pairs. A development build can force a count with `GLYD_ENGINE_SPARSE_SPLIT`.
    pub fn split(&self, pairs: usize) -> usize {
        split_for(self.sms, pairs)
    }

    /// The bytes of the partial sums `run` writes for `pairs` pairs of `hidden` outputs: the pairs' blocks (at most `BLOCKS_MOST` more than the pairs) a row of `hidden` floats each.
    pub fn part_bytes(pairs: usize, hidden: usize) -> usize {
        (pairs + BLOCKS_MOST) * hidden * 4
    }

    /// The blocks a pair's neurons are cut in for `t` tokens, as the room for partial sums allows (`down` and `sum` both ask).
    fn split_for(&self, cfg: &MoeCfg, part_room: usize, t: usize) -> Result<usize> {
        let (hidden, pairs) = (cfg.hidden, t * cfg.router.top_k);
        let room = part_room / (pairs * hidden * 4);
        if room == 0 {
            return err(format!("{pairs} pairs of {hidden} outputs need {} bytes of partial sums, where {part_room} are given", pairs * hidden * 4));
        }
        Ok(self.split(pairs).min(room))
    }

    /// The sparse product of each pair of the plan into `part` (fp32, `part_bytes` of room), from the first product's rows `a` [pairs, inter] (bf16, relu squared applied, the pairs in the plan's order) and the plan of
    /// `Lib::moe_way` (made of `via`). `via` (int64 [t, k]) is each pair's expert's place in `w`: the router's `ids` where `w` holds every expert, else the expert cache's `way` (a pair computed in another call of the
    /// layer is -1 in this one's; its parts, written by that call, are in `part` still, which is why `part` and the split are the layer's alone between its calls).
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (kernels/moe_rows.cu); the stream is the one the layer's other calls are on; the ids in `via` are below the places `w` has.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn down(&self, s: Stream, cfg: &MoeCfg, w: &RowStack, a: u64, plan: u64, via: u64, part: u64, part_room: usize, t: usize) -> Result<()> {
        let (hidden, inter, pairs) = (cfg.hidden, cfg.inter, t * cfg.router.top_k);
        if inter > ROWS_MAX_INTER || !inter.is_multiple_of(8) || !hidden.is_multiple_of(32) || (w.rows, w.cols, w.experts) != (inter, hidden, cfg.experts) {
            return err(format!("sparse experts of {inter} neurons and {hidden} outputs over a stack of {} x {} ({} experts)", w.rows, w.cols, w.experts));
        }
        if t == 0 {
            return Ok(());
        }
        if !a.is_multiple_of(16) {
            return err(format!("the activations: {a:#x} is not 16-byte aligned"));
        }
        let split = self.split_for(cfg, part_room, t)?;
        launch!(self.down[w.bits as usize - 4], [pairs as u32, split as u32, hidden.div_ceil(ROWS_OUT) as u32], [256, 1, 1], s;
            a, plan, via, w.data.ptr, w.scales.ptr, w.data_stride as i64, w.scale_stride as i64, part, cfg.experts as i32, inter as i32, hidden as i32, split as i32)
    }

    /// The tokens' weighted sums into y [t, hidden] (bf16) of the parts (`down`'s) of every pair whose id in `ids` (int64 [t, k]) is an expert, with the router's weights `wts` (fp32 [t, k]), and `extra` (fp32 [t, hidden];
    /// 0: none) added before the one rounding: the sums that the host made of the pairs whose experts it computed, which `ids` then names -1.
    ///
    /// # Safety
    /// As `down`; `extra` is a live device buffer of t x hidden floats where it is not 0.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sum(&self, s: Stream, cfg: &MoeCfg, ids: u64, wts: u64, part: u64, part_room: usize, y: u64, t: usize, extra: u64) -> Result<()> {
        let (k, hidden) = (cfg.router.top_k, cfg.hidden);
        if t == 0 {
            return Ok(());
        }
        let split = self.split_for(cfg, part_room, t)?;
        launch!(self.sum, [t as u32, hidden.div_ceil(64) as u32, 1], [256, 1, 1], s; part, wts, ids, y, cfg.experts as i32, k as i32, hidden as i32, split as i32, extra)
    }

    /// The routed sum of `t` tokens into y [t, hidden]: `down`, then `sum` of the router's choices `ids` (no host sums).
    ///
    /// # Safety
    /// As `down`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        s: Stream,
        cfg: &MoeCfg,
        w: &RowStack,
        a: u64,
        plan: u64,
        ids: u64,
        via: u64,
        wts: u64,
        part: u64,
        part_room: usize,
        y: u64,
        t: usize,
    ) -> Result<()> {
        self.down(s, cfg, w, a, plan, via, part, part_room, t)?;
        self.sum(s, cfg, ids, wts, part, part_room, y, t, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::Act;
    use crate::moe::Router;
    use glyd_gpu::pack::{pack_rows, transpose, unpack_rows, INT4_ROW_TAG, INT4_TAG, INT5_ROW_TAG, INT6_ROW_TAG};

    const SRC: &str = include_str!("../kernels/moe_rows.cu");

    fn bf16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    fn bf(w: u16) -> f32 {
        f32::from_bits((w as u32) << 16)
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
        }
    }

    /// What the Rust side names in the kernels' source is there: each kernel, and the sizes it shares with the kernels.
    #[test]
    fn kernels_and_constants_are_in_the_source() {
        for name in ["moe_down_rows4", "moe_down_rows5", "moe_down_rows6", "moe_rows_sum"] {
            assert!(SRC.contains(&format!("extern \"C\" __global__ void __launch_bounds__(256) {name}(")), "no kernel {name} in kernels/moe_rows.cu");
        }
        let define = |n: &str| -> Option<usize> {
            SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        assert_eq!((define("MOE_ROWS_OUT"), define("MOE_ROWS_MAX_INTER"), define("MOE_ROWS_WARPS")), (Some(ROWS_OUT), Some(ROWS_MAX_INTER), Some(8)));
    }

    /// The strides of an expert's rows in the three layouts at Nemotron 3 Super's shape (2,688 neurons of 1,024 outputs): 512, 640 and 768 bytes of codes a neuron and 64 of scales, the bytes the block layouts take; the
    /// experts joined are the packs end to end; a block layout's tag, packs of two shapes or layouts, and shapes the kernels do not take are refused.
    #[test]
    fn rows_strides_and_joins() {
        for (tag, row) in [(INT4_ROW_TAG, 512u64), (INT5_ROW_TAG, 640), (INT6_ROW_TAG, 768)] {
            assert_eq!(row_strides(tag, 2688, 1024).unwrap(), (2688 * row, 2688 * 64), "tag {tag}");
        }
        assert!(row_strides(INT4_TAG, 2688, 1024).is_err() && row_strides(2, 8, 32).is_err());
        assert!(row_strides(INT4_ROW_TAG, 0, 32).is_err() && row_strides(INT4_ROW_TAG, 8, 48).is_err() && row_strides(INT4_ROW_TAG, 8, 0).is_err());
        let w: Vec<u16> = (0..3 * 64 * 64).map(|i| bf16_bits(((i * 37 % 101) as f32 - 50.0) * 0.01)).collect();
        for bits in [4u32, 5, 6] {
            let packs: Vec<Tiered> = (0..3).map(|e| pack_rows(bits, &w[e * 4096..(e + 1) * 4096], 64, 64).unwrap()).collect();
            let j = RowsJoined::of(&packs.iter().collect::<Vec<_>>()).unwrap();
            assert_eq!((j.experts, j.rows, j.cols), (3, 64, 64));
            assert_eq!(j.data, packs.iter().flat_map(|p| p.data.clone()).collect::<Vec<u8>>());
            assert_eq!(j.scales, packs.iter().flat_map(|p| p.blocks.clone()).collect::<Vec<u8>>());
            assert_eq!((j.data.len() as u64, j.scales.len() as u64), (3 * row_strides(j.tag, 64, 64).unwrap().0, 3 * row_strides(j.tag, 64, 64).unwrap().1));
        }
        let (a, b) = (pack_rows(4, &w[..4096], 64, 64).unwrap(), pack_rows(5, &w[..4096], 64, 64).unwrap());
        assert!(RowsJoined::of(&[&a, &b]).is_err(), "two layouts");
        let c = pack_rows(4, &w[..2048], 32, 64).unwrap();
        assert!(RowsJoined::of(&[&a, &c]).is_err(), "two shapes");
        assert!(RowsJoined::of(&[]).is_err());
        let mut short = a.clone();
        short.data.pop();
        assert!(RowsJoined::of(&[&short]).is_err());
    }

    /// Room for fewer experts than a layer has (the expert cache's places) is asked for what the row layouts hold: places for 1 to all of the layer's experts, in a row layout, of a shape the layouts take (all refused
    /// before anything is allocated).
    #[test]
    fn row_places_are_refused_what_they_cannot_hold() {
        let why = |r: Result<RowStack>| r.err().expect("refused").to_string();
        assert!(why(RowStack::pooled(INT4_ROW_TAG, 8, 0, 64, 64)).contains("room for 0 experts standing for 8"));
        assert!(why(RowStack::pooled(INT4_ROW_TAG, 8, 9, 64, 64)).contains("room for 9 experts standing for 8"));
        assert!(why(RowStack::pooled(INT4_TAG, 8, 4, 64, 64)).contains("row layouts"));
        assert!(why(RowStack::pooled(INT5_ROW_TAG, 8, 4, 64, 48)).contains("runs of 32"));
        assert!(why(RowStack::pooled(INT6_ROW_TAG, 8, 4, 0, 64)).contains("runs of 32"));
    }

    /// The sparse product of one pair on the host, as the kernels do it from the bytes of the row layout (the planes and words read by the layout's doc, apart from the packer's decode): the pair's neurons whose
    /// bf16 is not zero, in order, each its row's weights (the bf16 of (code - bias) times the run's scale) times the activation, added in fp32 (the sums in fp32, a pair's order the kernel's but for how its
    /// blocks and warps cut the list).
    fn sparse_pair(bits: u32, data: &[u8], scales: &[u8], a: &[u16], hidden: usize) -> Vec<f32> {
        let (runs, hb) = (hidden / 32, 4 * (bits as usize - 4));
        let stride = rows_stride(bits, hidden);
        let word = |row: &[u8], at: usize| u32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]);
        let mut out = vec![0f32; hidden];
        for (n, &av) in a.iter().enumerate().filter(|(_, &v)| v & 0x7fff != 0) {
            let row = &data[n * stride..(n + 1) * stride];
            for k in 0..runs {
                let s = bf(u16::from_le_bytes([scales[(n * runs + k) * 2], scales[(n * runs + k) * 2 + 1]]));
                for i in 0..32 {
                    let (pr, h) = (i / 2, i % 2);
                    let mut u = (word(row, 16 * k + 4 * (pr / 4)) >> (4 * (pr % 4) + 16 * h)) & 15;
                    if bits == 5 {
                        u |= ((word(row, 16 * runs + hb * k) >> (pr + 16 * h)) & 1) << 4;
                    }
                    if bits == 6 {
                        u |= ((word(row, 16 * runs + hb * k + 4 * (pr / 8)) >> (2 * (pr % 8) + 16 * h)) & 3) << 4;
                    }
                    let w = bf(bf16_bits((u as i32 - (1 << (bits - 1))) as f32 * s));
                    out[32 * k + i] = bf(av).mul_add(w, out[32 * k + i]);
                }
            }
        }
        out
    }

    /// The CPU half's row layout (`cpu_experts::Rows`, made from the block pack of the same matrix: the runs of 32 along its columns) is the bytes `pack_rows` makes at the three widths, codes and scales: the CPU kernel
    /// reads the packs the GPU's sparse product reads, whichever packer made them.
    #[test]
    fn the_cpu_halfs_rows_are_the_row_packs() {
        use crate::cpu_experts::Rows;
        use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6};
        let (rows, cols) = (192usize, 96usize);
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let w: Vec<u16> = (0..rows * cols).map(|i| bf16_bits(r.f(-1.0, 1.0) * if i / 32 % 3 == 0 { 0.02 } else { 1.5 })).collect();
        type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
        for (bits, block) in [(4u32, pack_int4 as Pack), (5, pack_int5 as Pack), (6, pack_int6 as Pack)] {
            let (theirs, mine) = (Rows::from_block(&block(&w, rows, cols).unwrap()).unwrap(), pack_rows(bits, &w, rows, cols).unwrap());
            assert!(theirs.data == mine.data && theirs.scales == mine.blocks, "{bits} bits: the CPU half's rows are not pack_rows'");
        }
    }

    /// The sparse product is the dense product: random relu squared rows of about 90% zeros (a row of all zeros, a row of every neuron, negative zeros and subnormals among them), the down matrices packed in the
    /// row layout at the three widths and the dense product taken on the weights they decode to (every neuron, in f64); the pairs of a token weighted and summed as the layer does, rounded to bf16 once.
    #[test]
    fn the_sparse_product_is_the_dense_product() {
        let (inter, hidden, experts, k, tokens) = (320usize, 128usize, 6usize, 3usize, 4usize);
        let mut r = Rng(0x5eed_2026);
        for bits in [4u32, 5, 6] {
            // the down matrices [hidden, inter] of each expert, packed as their transposes
            let down: Vec<Vec<u16>> = (0..experts).map(|_| (0..hidden * inter).map(|_| bf16_bits(r.f(-0.08, 0.08))).collect()).collect();
            let packs: Vec<Tiered> = down.iter().map(|d| pack_rows(bits, &transpose(d, hidden, inter), inter, hidden).unwrap()).collect();
            let decoded: Vec<Vec<f32>> = packs
                .iter()
                .map(|p| {
                    let mut t = vec![0u16; inter * hidden];
                    assert!(unpack_rows(p, &mut t));
                    transpose(&t, inter, hidden).into_iter().map(bf).collect() // (back to [hidden, inter], the weights the dense product multiplies by)
                })
                .collect();
            let mut worst = 0f64;
            for t in 0..tokens {
                let (mut dense, mut sparse) = (vec![0f64; hidden], vec![0f64; hidden]);
                let mut mag = vec![0f64; hidden];
                for c in 0..k {
                    let e = (t * 2 + c * 3) % experts;
                    let wt = r.f(0.2, 2.5);
                    // a pair's activations: about 90% zero; the first pair of the first token none, the first of the second all, the last of each a mix with negative zeros and a subnormal
                    let a: Vec<u16> = (0..inter)
                        .map(|n| match (t, c) {
                            (0, 0) => 0,
                            (1, 0) => bf16_bits(r.f(0.01, 3.0)),
                            _ if n % 41 == 0 => 0x8000,
                            _ if n == 7 => 0x0001,
                            _ if r.next().is_multiple_of(10) => bf16_bits(r.f(0.01, 3.0)),
                            _ => 0,
                        })
                        .collect();
                    let pair = sparse_pair(bits, &packs[e].data, &packs[e].blocks, &a, hidden);
                    for o in 0..hidden {
                        let d: f64 = (0..inter).map(|n| bf(a[n]) as f64 * decoded[e][o * inter + n] as f64).sum();
                        dense[o] += d * wt as f64;
                        sparse[o] += pair[o] as f64 * wt as f64;
                        mag[o] += (0..inter).map(|n| (bf(a[n]) as f64 * decoded[e][o * inter + n] as f64).abs()).sum::<f64>() * wt as f64;
                    }
                }
                for o in 0..hidden {
                    let off = (dense[o] - sparse[o]).abs() / mag[o].max(1e-30);
                    worst = worst.max(off);
                    // (the bf16 each is rounded to: within a place unless the sums fall either side of a rounding, or the sum is lost in the terms it is made of)
                    let (x, y) = (bf16_bits(dense[o] as f32) as i32, bf16_bits(sparse[o] as f32) as i32);
                    assert!((x - y).abs() <= 1 || off < 4e-6, "{bits} bits, token {t}, output {o}: {} against {}", dense[o], sparse[o]);
                }
            }
            assert!(worst < 2e-6, "{bits} bits: the sparse sums differ from the dense by {worst:e} of the terms' magnitude");
        }
    }

    /// A row layout is the second matrix of relu squared experts of the layer's numbers, in sizes the kernels take; the room for the partial sums holds a decode step's pairs cut in eight and a prompt chunk's pairs
    /// whole, about the dense product's floats.
    #[test]
    fn a_row_layout_is_checked_against_the_layer() {
        let cfg = MoeCfg { experts: 512, router_hidden: 4096, hidden: 1024, inter: 2688, router: Router::sigmoid(22, 5.0), act: Act::Relu2 };
        rows_fit(512, 2688, 1024, &cfg).unwrap();
        let says = |r: Result<()>, what: &str| {
            let e = r.unwrap_err().msg;
            assert!(e.contains(what), "{e:?} does not say {what:?}");
        };
        says(rows_fit(512, 2688, 1024, &MoeCfg { act: Act::GeGlu, ..cfg }), "it is for relu squared");
        says(rows_fit(511, 2688, 1024, &cfg), "where the layer has 512 of 2688 neurons of 1024 outputs");
        says(rows_fit(512, 1024, 2688, &cfg), "where the layer has 512 of 2688");
        let big = MoeCfg { inter: 4128, ..cfg };
        says(rows_fit(512, 4128, 1024, &big), "at most 4096 neurons");
        let odd = MoeCfg { hidden: 1040, ..cfg };
        says(rows_fit(512, 2688, 1040, &odd), "runs of 32");
        assert!(RowsKern::part_bytes(22, 1024) >= 22 * 8 * 1024 * 4, "a decode step's 22 pairs can all be cut in 8");
        let chunk = RowsKern::part_bytes(2048 * 22, 1024);
        assert!(
            (2048 * 22 * 1024 * 4..2048 * 22 * 1024 * 4 * 102 / 100).contains(&chunk),
            "a prompt chunk of 2,048 tokens takes about the dense product's partial sums: {chunk} bytes"
        );
    }
}
