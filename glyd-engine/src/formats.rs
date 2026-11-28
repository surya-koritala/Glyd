//! A release's quantized matrices decoded exactly, by the type they are stored in and the scales stored beside them: FP8 (E4M3) with a power-of-two scale
//! (E8M0) a block of the matrix, the block read from the scales' shape; FP4 (E2M1, two weights a byte, the first in the low four bits) with an E8M0 scale a
//! run of 32 of a row, as a release names it (`X.weight` stored as I8, its scales `X.scale`) or as compressed-tensors packs it (MXFP4: `X.weight_packed`,
//! `X.weight_scale`). A weight times a power of two is exact in f32, and in bf16 where it is in range. And a table read by rows stored at 4 bits (NF4 levels
//! times a bf16 scale a row: `X.q4`), as v0.34.2 decoded it. Where a description names a matrix's scales (`Stored`), the scales' tensor is the one it names,
//! whatever its suffix, and the block is the one the scales' shape and the config's `weight_block_size` give: fp32 scales a block (`X.qscale_weight` as
//! `X.weight_scale_inv`), or a byte's power of two a run of 32 columns of a row; and MXFP4 codes under a release's own names.

use crate::ffi::{err, Error, Result};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::BTreeMap;

/// An E8M0 scale's value: 2^(b - 127); 255 is NaN.
pub fn e8m0(b: u8) -> f32 {
    match b {
        255 => f32::NAN,
        // (2^-127 is an f32 subnormal)
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((b as u32) << 23),
    }
}

/// E2M1's values, the low three bits the magnitude and the fourth the sign.
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// An FP4 matrix [rows, cols] (two weights a byte, the first in the low four bits; `bytes` rows x cols / 2) with its E8M0 scales (one per 32 weights of a row) as five-bit runs: each weight's code (twice its
/// E2M1 value, -12 to 12) and each run's bf16 scale (half its E8M0 scale), so that code x scale is the weight exactly. An Err where a scale is NaN or past bf16.
pub fn fp4_runs(bytes: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<(Vec<i8>, Vec<u16>)> {
    if cols % 32 != 0 || bytes.len() * 2 != rows * cols || scales.len() * 32 != rows * cols {
        return err(format!("{} bytes and {} scales for a {rows} x {cols} FP4 matrix", bytes.len(), scales.len()));
    }
    const CODE: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let code: Vec<i8> = bytes.iter().flat_map(|&b| [CODE[(b & 15) as usize], CODE[(b >> 4) as usize]]).collect();
    let scale = scales
        .iter()
        .enumerate()
        .map(|(i, &e)| match e {
            255 => err(format!("run {i}'s scale is NaN")),
            // 2^(e - 128): bf16's exponent e - 1, or below its normal range a subnormal (2^-133 times 2^(e + 5))
            2.. => Ok(((e - 1) as u16) << 7),
            _ => Ok(1u16 << (e + 5)),
        })
        .collect::<Result<_>>()?;
    Ok((code, scale))
}

/// An FP8 matrix [rows, cols] with its E8M0 scales, one a block of `block` rows and columns, as bf16 words, exactly (`glyd_gpu::fp8::dequant`).
pub fn fp8_blocks(q: &[u8], scales: &[u8], rows: usize, cols: usize, block: [usize; 2]) -> Result<Vec<u16>> {
    let s: Vec<f32> = scales.iter().map(|&b| e8m0(b)).collect();
    let mut out = vec![0u16; rows * cols];
    glyd_gpu::fp8::dequant(q, &s, rows, cols, block, &mut out).map_err(Error)?;
    Ok(out)
}

/// `fp8_blocks` in blocks of 128 x 128.
pub fn fp8_bf16(q: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<u16>> {
    fp8_blocks(q, scales, rows, cols, [128, 128])
}

/// An MXFP4 matrix [rows, cols] as compressed-tensors packs it: two weights a byte (the even column in the low four bits), an E8M0 scale (2^(s - 127)) per 32 of a row.
pub fn mxfp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if cols % 32 != 0 || packed.len() != rows * cols / 2 || scales.len() != rows * cols / 32 {
        return err(format!("an MXFP4 matrix of {rows} x {cols}: {} bytes and {} scales", packed.len(), scales.len()));
    }
    Ok((0..rows * cols)
        .map(|i| {
            let b = packed[i / 2];
            let code = if i % 2 == 0 { b & 15 } else { b >> 4 };
            E2M1[code as usize] * 2f32.powi(scales[i / 32] as i32 - 127)
        })
        .collect())
}

/// The block of rows and columns a quantized matrix of `rows` x `cols` has one scale of, from its scales' shape (a block of 128, of 32, or one scale a row
/// or a column), else None.
pub fn scale_block(rows: usize, cols: usize, shape: &[u64]) -> Option<[usize; 2]> {
    let (sr, sc) = (*shape.first()? as usize, shape.get(1).copied().unwrap_or(1) as usize);
    let of = |n: usize, k: usize| [128usize, 32, 1].into_iter().find(|&b| n.div_ceil(b) == k);
    Some([of(rows, sr)?, of(cols, sc)?])
}

/// The fp32 block scales stored beside the FP8 matrix `name` as transformers' fine-grained FP8 names them (`X.weight_scale_inv`, what each E4M3 value of
/// its block is multiplied by: [ceil(rows / 128), ceil(cols / 128)]), where the checkpoint holds them so; None otherwise.
pub fn scales_beside(ck: &Checkpoint, name: &str) -> Option<String> {
    let (_, t) = ck.get(name)?;
    let s = format!("{name}_scale_inv");
    let (_, st) = ck.get(&s)?;
    let (rows, cols) = match t.shape[..] {
        [r, c] => (r, c),
        _ => return None,
    };
    (t.dtype == "F8_E4M3" && st.dtype == "F32" && st.shape == [rows.div_ceil(128), cols.div_ceil(128)]).then_some(s)
}

/// An FP8 matrix [rows, cols] with its fp32 scales, one a block of 128 x 128 ([ceil(rows / 128), ceil(cols / 128)]): each value its E4M3 code's value
/// times its block's scale, one fp32 product (the value the GPU's products take: an E4M3 value is exact in bf16 and f32, the product rounded once).
pub fn fp8_scaled(q: &[u8], scales: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    fp8_scaled_blocks(q, scales, rows, cols, [128, 128])
}

/// `fp8_scaled` in blocks of `block` rows and columns (the scales row-major over [ceil(rows / block[0]), ceil(cols / block[1])], the last block of a row
/// or a column short where the matrix does not divide).
pub fn fp8_scaled_blocks(q: &[u8], scales: &[f32], rows: usize, cols: usize, block: [usize; 2]) -> Result<Vec<f32>> {
    let [br, bc] = block;
    if br == 0 || bc == 0 || cols == 0 || q.len() != rows * cols || scales.len() != rows.div_ceil(br) * cols.div_ceil(bc) {
        return err(format!("{} bytes and {} scales for an FP8 matrix of {rows} x {cols} in blocks of {br} x {bc}", q.len(), scales.len()));
    }
    let gc = cols.div_ceil(bc);
    Ok(q.iter().enumerate().map(|(i, &b)| glyd_gpu::fp8::e4m3(b) * scales[(i / cols) / br * gc + (i % cols) / bc]).collect())
}

/// The block a described FP8 matrix of `rows` x `cols` has one scale of, and the shards its scales start again at, from its scales' shape (a grid of
/// two dimensions): the first of `given` (the description's block, the config's `weight_block_size`) whose grid is that shape, else the first of 128, 32
/// and 1 rows by 128, 32 and 1 columns whose grid is (a run of 32 columns of a row is [1, 32]). A matrix the description cuts in `shards` equal parts
/// of its rows (1: not cut) may hold a grid a shard, one after another, [shards x ceil(rows / shards / block[0]), ceil(cols / block[1])]: that reading
/// is taken where the shape is its shape, before one grid over the whole matrix (the two are one where a shard is whole blocks). None where no block
/// makes the shape.
pub fn grid_block(rows: usize, cols: usize, shape: &[u64], given: &[[usize; 2]], shards: usize) -> Option<([usize; 2], usize)> {
    let [gr, gc] = shape[..] else { return None };
    let fits = |b: [usize; 2], k: usize| {
        b[0] > 0 && b[1] > 0 && k > 0 && rows.is_multiple_of(k) && [(k * (rows / k).div_ceil(b[0])) as u64, cols.div_ceil(b[1]) as u64] == [gr, gc]
    };
    const SIZES: [usize; 3] = [128, 32, 1];
    let blocks = given.iter().copied().chain(SIZES.iter().flat_map(|&a| SIZES.iter().map(move |&b| [a, b])));
    blocks.flat_map(|b| [(b, shards.max(1)), (b, 1)]).find(|&(b, k)| fits(b, k))
}

/// An FP8 matrix [rows, cols] cut in `shards` equal parts of its rows, each with its own grid of fp32 scales a block (`scales`: the shards' grids one
/// after another, each [ceil(rows / shards / block[0]), ceil(cols / block[1])]): each shard decoded with its grid (`fp8_scaled_blocks`), a shard's last
/// block of rows short where its rows do not divide, and the next shard's first block whole.
pub fn fp8_scaled_shards(q: &[u8], scales: &[f32], rows: usize, cols: usize, block: [usize; 2], shards: usize) -> Result<Vec<f32>> {
    if shards == 0 || !rows.is_multiple_of(shards) || q.len() != rows * cols || !scales.len().is_multiple_of(shards) {
        return err(format!("{} bytes and {} scales for an FP8 matrix of {rows} x {cols} in {shards} shards", q.len(), scales.len()));
    }
    let (each, grid) = (rows / shards, scales.len() / shards);
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..shards {
        out.extend(fp8_scaled_blocks(&q[i * each * cols..(i + 1) * each * cols], &scales[i * grid..(i + 1) * grid], each, cols, block)?);
    }
    Ok(out)
}

/// The stored formats of a checkpoint as its description says them (`describe::Described::stored`): which tensor holds a quantized matrix's scales, and
/// the FP8 matrices' block where the config gives one.
#[derive(Clone, Debug, Default)]
pub struct Stored {
    /// A quantized matrix's name and its scales' (`Described::scales`).
    pub scales: BTreeMap<String, String>,
    /// The FP8 matrices' square block (`Described::fp8_block`).
    pub block: Option<usize>,
    /// A matrix the description cuts in shards (a fused product whose roles are each a range of rows a shard, `describe::Source::RowsOf`): how many.
    /// Its block scales may start again at each (`grid_block`).
    pub shards: BTreeMap<String, usize>,
}

impl Stored {
    /// Tensor `name` as f32: a matrix the description names scales for, decoded with those scales (`described`); any other tensor, a matrix stored in
    /// another type than the description says (a random model's bf16), and a format `described` does not take, by what the checkpoint holds (`tensor`).
    pub fn tensor(&self, ck: &Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        self.described(ck, name).or_else(|| tensor(ck, name))
    }

    /// The matrix `name` decoded with the scales the description names for it, by the types the two are stored in:
    ///  - FP8 (E4M3) with fp32 scales a block (`grid_block`): each value its code's value times its block's scale, one fp32 product (`fp8_scaled_blocks`);
    ///  - FP8 with a byte's power-of-two scale (U8 or E8M0: 2^(s - 127)) a block: the product, exact, as the bf16 it is (`fp8_blocks`).
    ///
    ///  - 4-bit codes two a byte (U8 [rows, cols / 2]) with a byte's power-of-two scale a run of 32 of a row (U8 or E8M0 [rows, cols / 32]): the MXFP4
    ///    matrix [rows, cols] (`mxfp4`), under whatever names the release stores the two (MiMo V2's experts: `X.weight`, `X.weight_scale`).
    ///
    /// A matrix the description cuts in shards whose scales are a grid a shard is decoded a shard at a time, each with its grid.
    ///
    /// None where the description names no scales for it, or the two tensors are not one of these.
    fn described(&self, ck: &Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let sn = self.scales.get(name)?;
        let ((_, t), (_, st)) = (ck.get(name)?, ck.get(sn)?);
        let [rows, cols] = t.shape[..] else { return None };
        let (rows, cols) = (rows as usize, cols as usize);
        let bf16 = |w: u16| f32::from_bits((w as u32) << 16);
        match (t.dtype.as_str(), st.dtype.as_str()) {
            ("F8_E4M3", "F32" | "U8" | "F8_E8M0") => {
                let given: Vec<[usize; 2]> = self.block.map(|b| [b, b]).into_iter().chain(ck.fp8_block).collect();
                let (block, shards) = grid_block(rows, cols, &st.shape, &given, self.shards.get(name).copied().unwrap_or(1))?;
                let (q, s) = (ck.read(name).ok()?, ck.read(sn).ok()?);
                let v = if st.dtype == "F32" {
                    let s: Vec<f32> = s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    fp8_scaled_shards(&q, &s, rows, cols, block, shards).ok()?
                } else {
                    let (each, grid) = (rows / shards, s.len() / shards);
                    let mut v = Vec::with_capacity(rows * cols);
                    for i in 0..shards {
                        let w = fp8_blocks(&q[i * each * cols..(i + 1) * each * cols], &s[i * grid..(i + 1) * grid], each, cols, block).ok()?;
                        v.extend(w.into_iter().map(bf16));
                    }
                    v
                };
                Some((vec![rows, cols], v))
            }
            ("U8", "U8" | "F8_E8M0") if st.shape == [rows as u64, cols as u64 / 16] => {
                Some((vec![rows, cols * 2], mxfp4(&ck.read(name).ok()?, &ck.read(sn).ok()?, rows, cols * 2).ok()?))
            }
            _ => None,
        }
    }
}

/// A checkpoint's tensor as f32, by the type it is stored in: bf16 and f32 as stored, int64 (a hash table's ids) as their values; an FP8 matrix with its
/// scales (`X.scale`, a block each, or fp32 block scales `X.weight_scale_inv`: `scales_beside`); an FP4 matrix stored as I8 with its runs' scales
/// (`X.scale`); and where `X.weight` or its codes `X.weight_packed` are asked for and the checkpoint holds the matrix as compressed-tensors packs it, that
/// MXFP4 matrix, or the table stored at 4 bits (`X.q4`) as the bf16 values it decodes to (`nf4_row`). Another type, or a quantized matrix without its
/// scales: none. The scales are found by these names alone: a matrix whose scales its description names otherwise is read by `Stored::tensor`.
pub fn tensor(ck: &Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let Some((_, t)) = ck.get(name) else { return packed_fp4(ck, name).or_else(|| q4_table(ck, name)) };
    let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
    let bf16 = |w: u16| f32::from_bits((w as u32) << 16);
    let v = match t.dtype.as_str() {
        "BF16" => ck.read(name).ok()?.chunks_exact(2).map(|c| bf16(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "F32" => ck.read(name).ok()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "I64" => ck.read(name).ok()?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()) as f32).collect(),
        "U8" if name.ends_with(".weight_packed") => return packed_fp4(ck, name),
        "F8_E4M3" if shape.len() == 2 && scales_beside(ck, name).is_some() => {
            let s = ck.read(&scales_beside(ck, name)?).ok()?;
            let s: Vec<f32> = s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            fp8_scaled(&ck.read(name).ok()?, &s, shape[0], shape[1]).ok()?
        }
        "F8_E4M3" | "I8" if shape.len() == 2 => {
            let scales = format!("{}.scale", name.strip_suffix(".weight")?);
            let (_, st) = ck.get(&scales)?;
            let (q, s) = (ck.read(name).ok()?, ck.read(&scales).ok()?);
            let (rows, cols) = (shape[0], shape[1]);
            if t.dtype == "I8" {
                let (code, scale) = fp4_runs(&q, &s, rows, cols * 2).ok()?;
                return Some((vec![rows, cols * 2], code.iter().enumerate().map(|(i, &c)| c as f32 * bf16(scale[i / 32])).collect()));
            }
            let block = scale_block(rows, cols, &st.shape)?;
            fp8_blocks(&q, &s, rows, cols, block).ok()?.into_iter().map(bf16).collect()
        }
        _ => return None,
    };
    Some((shape, v))
}

/// NF4's levels (the quantiles of a normal distribution, scaled to [-1, 1]): the values of a table stored at 4 bits (`nf4_row`).
pub const NF4: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// The name a table read by rows (`X.weight`, bf16) has stored at 4 bits: `X.q4`, U8 [rows, 2 + cols / 2], a row its bf16 scale, then a code of 4 bits a
/// value, low nibble first (`nf4_row`). Glyd's slim checkpoints store the n-gram table so.
pub fn q4_name(name: &str) -> Option<String> {
    name.strip_suffix(".weight").map(|b| format!("{b}.q4"))
}

/// The name a stored tensor stands for: a table at 4 bits (`X.q4`) for `X.weight`, which a description reads; any other name for itself.
pub fn stands_for(stored: &str) -> std::borrow::Cow<'_, str> {
    match stored.strip_suffix(".q4") {
        Some(b) => format!("{b}.weight").into(),
        None => stored.into(),
    }
}

/// The 4-bit table stored for `name` (`q4_name`) where the checkpoint holds it and not `name` itself.
pub fn q4_beside(ck: &Checkpoint, name: &str) -> Option<String> {
    let q = q4_name(name).filter(|_| ck.get(name).is_none())?;
    ck.get(&q).map(|_| q)
}

/// The bytes a row of `cols` values takes stored at 4 bits, where its shape `[rows, bytes]` and type are that form's for `cols` (an even number).
pub fn q4_row_bytes(dtype: &str, shape: &[u64], cols: usize) -> Option<usize> {
    (dtype == "U8" && shape.len() == 2 && cols % 2 == 0 && shape[1] as usize == 2 + cols / 2).then_some(2 + cols / 2)
}

/// A row of a table stored at 4 bits (`q`: its bf16 scale, then the codes, low nibble first) as its `out.len() / 2` bf16 values (little-endian), each its
/// NF4 level times the scale: one f32 product, rounded to bf16 to nearest, ties to even. v0.34.2's decode of the n-gram table, bit for bit.
pub fn nf4_row(q: &[u8], out: &mut [u8]) {
    let scale = f32::from_bits(((q[0] as u32) | ((q[1] as u32) << 8)) << 16);
    for (j, o) in out.chunks_exact_mut(2).enumerate() {
        let code = (q[2 + j / 2] >> ((j & 1) * 4)) & 15;
        let b = (NF4[code as usize] * scale).to_bits();
        let v = (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16;
        o.copy_from_slice(&v.to_le_bytes());
    }
}

/// A bf16 row (little-endian) at 4 bits, in `nf4_row`'s form: its scale its largest magnitude rounded to bf16, each value the code whose decode is nearest
/// it (of a tie, the one of the value's own bits, else the lower: zeros stay +0). A random model's tables are written so (src/logits.rs `tables_at_4_bits`).
pub fn nf4_encode(row: &[u8]) -> Vec<u8> {
    let bf = |b: u32| (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16;
    let vals: Vec<f32> = row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let scale = bf(vals.iter().fold(0f32, |m, v| m.max(v.abs())).to_bits());
    let levels: Vec<f32> = NF4.iter().map(|l| f32::from_bits((bf((l * f32::from_bits((scale as u32) << 16)).to_bits()) as u32) << 16)).collect();
    let mut out = vec![0u8; 2 + vals.len().div_ceil(2)];
    out[..2].copy_from_slice(&scale.to_le_bytes());
    for (j, v) in vals.iter().enumerate() {
        let key = |c: usize| ((levels[c] - v).abs(), levels[c].to_bits() != v.to_bits());
        let code = (0..16).min_by(|&a, &b| key(a).0.total_cmp(&key(b).0).then(key(a).1.cmp(&key(b).1))).unwrap_or(7) as u8;
        out[2 + j / 2] |= code << ((j & 1) * 4);
    }
    out
}

/// A table stored at 4 bits (`X.q4`, `q4_beside`) of `cols` values a row as its bf16 words, every row decoded (`nf4_row`).
pub fn nf4_words(ck: &Checkpoint, q4: &str, cols: usize) -> Result<Vec<u16>> {
    let (_, t) = ck.get(q4).ok_or_else(|| Error(format!("{}: no tensor {q4}", ck.dir.display())))?;
    let rb = q4_row_bytes(&t.dtype, &t.shape, cols)
        .ok_or_else(|| Error(format!("{q4}: {} {:?}, where a table of {cols} values a row at 4 bits is U8 [rows, {}]", t.dtype, t.shape, 2 + cols / 2)))?;
    let q = ck.read(q4).map_err(|e| Error(format!("{q4}: {e}")))?;
    let mut out = vec![0u16; q.len() / rb * cols];
    // SAFETY: u16's bytes, as many as its elements take; written little-endian, as the host's.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) };
    for (row, o) in q.chunks_exact(rb).zip(bytes.chunks_exact_mut(cols * 2)) {
        nf4_row(row, o);
    }
    Ok(out)
}

/// The 4-bit table stored for `X.weight` (`X.q4`), decoded, where `X.weight` is asked for.
fn q4_table(ck: &Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let q = q4_beside(ck, name)?;
    let (_, t) = ck.get(&q)?;
    let cols = (*t.shape.get(1)? as usize).checked_sub(2)? * 2;
    let w = nf4_words(ck, &q, cols).ok()?;
    Some((vec![t.shape[0] as usize, cols], w.into_iter().map(|x| f32::from_bits((x as u32) << 16)).collect()))
}

/// The MXFP4 matrix compressed-tensors stores for `X.weight` (`X.weight_packed`, `X.weight_scale`), decoded, where `X.weight` or its codes are asked for.
fn packed_fp4(ck: &Checkpoint, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let base = name.strip_suffix(".weight_packed").or_else(|| name.strip_suffix(".weight"))?;
    let (_, t) = ck.get(&format!("{base}.weight_packed"))?;
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize * 2);
    let w = mxfp4(&ck.read(&format!("{base}.weight_packed")).ok()?, &ck.read(&format!("{base}.weight_scale")).ok()?, rows, cols).ok()?;
    Some((vec![rows, cols], w))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row at 4 bits is its bf16 scale, then a code a value, the low nibble first: codes 0 to 15 in turn under a scale of 2 decode to NF4's levels times 2
    /// (exact in f32), each rounded to bf16 once (-2, 0 and 2 exactly; the others within bf16's half step). A row written at 4 bits (`nf4_encode`) decodes to
    /// each value's nearest level of its scale, the scale its largest magnitude; a row of zeros to zeros. And the names: `X.weight` is `X.q4` at 4 bits.
    #[test]
    fn a_row_at_4_bits_is_a_scale_and_nf4_codes_low_nibble_first() {
        let words = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect() };
        let mut q = vec![0x00, 0x40];
        q.extend((0..8u8).map(|k| (2 * k) | ((2 * k + 1) << 4)));
        let mut out = vec![0u8; 32];
        nf4_row(&q, &mut out);
        let got = words(&out);
        for (c, g) in got.iter().enumerate() {
            let want = NF4[c] * 2.0;
            assert!((g - want).abs() <= want.abs() / 256.0, "code {c}: {g} for {want}");
        }
        assert_eq!((got[0], got[7], got[15]), (-2.0, 0.0, 2.0));
        let row: Vec<u8> = (0..40).flat_map(|i| (((((i * 37 % 23) as f32 - 11.0) / 7.0).to_bits() >> 16) as u16).to_le_bytes()).collect();
        let x = words(&row);
        let q = nf4_encode(&row);
        assert_eq!((q.len(), words(&q[..2])[0]), (22, x.iter().fold(0f32, |m, v| m.max(v.abs()))));
        let mut back = vec![0u8; row.len()];
        nf4_row(&q, &mut back);
        let levels: Vec<f32> = (0..16u8)
            .map(|c| {
                let mut o = [0u8; 2];
                nf4_row(&[q[0], q[1], c], &mut o);
                words(&o)[0]
            })
            .collect();
        for (v, b) in x.iter().zip(words(&back)) {
            assert!(levels.contains(&b) && levels.iter().all(|l| (l - v).abs() >= (b - v).abs()), "{v} decoded to {b}");
        }
        let zeros = nf4_encode(&[0u8; 8]);
        let mut z = [1u8; 8];
        nf4_row(&zeros, &mut z);
        assert_eq!(z, [0u8; 8]);
        assert_eq!(
            (q4_name("a.b.weight").as_deref(), q4_name("a.b"), stands_for("a.b.q4"), stands_for("a.b")),
            (Some("a.b.q4"), None, "a.b.weight".into(), "a.b".into())
        );
    }

    #[test]
    fn fp4_runs_are_the_weights_exactly() {
        for e in [0u8, 1, 2, 100, 120, 127, 128, 200, 254] {
            let bytes: Vec<u8> = (0..16u8).map(|i| i | (15 - i) << 4).collect();
            let (code, scale) = fp4_runs(&bytes, &[e], 1, 32).unwrap();
            let s = f32::from_bits((scale[0] as u32) << 16);
            for (k, &c) in code.iter().enumerate() {
                let nib = if k % 2 == 0 { bytes[k / 2] & 15 } else { bytes[k / 2] >> 4 };
                assert_eq!(c as f32 * s, E2M1[nib as usize] * e8m0(e), "scale {e}, weight {k}");
            }
        }
        assert!(fp4_runs(&[0; 16], &[255], 1, 32).is_err());
    }

    /// A checkpoint's quantized matrices are read by the type they are stored in and the scales beside them, whatever the model calls them: an FP8 matrix
    /// with its scales a block of 128 under one release's names and under transformers' (a second family's), and with its scales a block of 32; an FP4
    /// matrix stored as I8 with its runs' scales, and the same weights as compressed-tensors packs them (MXFP4): each the code's value times its scale,
    /// exactly. A quantized matrix without its scales is not read.
    #[test]
    fn a_quantized_matrix_is_read_by_its_format() {
        let dir = std::env::temp_dir().join(format!("glyd-formats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (rows, cols) = (256usize, 256usize);
        // (E4M3 codes but the two NaNs, and scales near one)
        let q: Vec<u8> = (0..rows * cols).map(|i| (i * 37 % 251) as u8).map(|b| if b & 0x7F == 0x7F { 0x7E } else { b }).collect();
        let (s128, s32): (Vec<u8>, Vec<u8>) = ((0..4).map(|i| 120 + i as u8).collect(), (0..64).map(|i| 118 + (i % 16) as u8).collect());
        let (fr, fc) = (4usize, 64usize);
        let nibbles: Vec<u8> = (0..fr * fc / 2).map(|i| (i * 11 % 256) as u8).collect();
        let runs: Vec<u8> = (0..fr * fc / 32).map(|i| 124 + i as u8).collect();
        let names = [
            ("layers.0.attn.wq_a.weight", "F8_E4M3", vec![rows as u64, cols as u64], &q[..]),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![2, 2], &s128[..]),
            ("model.layers.0.self_attn.q_proj.weight", "F8_E4M3", vec![rows as u64, cols as u64], &q[..]),
            ("model.layers.0.self_attn.q_proj.scale", "F8_E8M0", vec![2, 2], &s128[..]),
            ("layers.1.attn.wq_a.weight", "F8_E4M3", vec![rows as u64, cols as u64], &q[..]),
            ("layers.1.attn.wq_a.scale", "F8_E8M0", vec![8, 8], &s32[..]),
            ("layers.2.attn.wq_a.weight", "F8_E4M3", vec![rows as u64, cols as u64], &q[..]),
            ("layers.0.ffn.experts.0.w1.weight", "I8", vec![fr as u64, fc as u64 / 2], &nibbles[..]),
            ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", vec![fr as u64, fc as u64 / 32], &runs[..]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight_packed", "U8", vec![fr as u64, fc as u64 / 2], &nibbles[..]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight_scale", "U8", vec![fr as u64, fc as u64 / 32], &runs[..]),
        ];
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &names).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        let fp8 = |block: usize, s: &[u8]| -> Vec<f32> {
            (0..rows * cols).map(|i| glyd_gpu::fp8::e4m3(q[i]) * e8m0(s[(i / cols / block) * cols.div_ceil(block) + i % cols / block])).collect()
        };
        for (name, block, s) in
            [("layers.0.attn.wq_a.weight", 128, &s128), ("model.layers.0.self_attn.q_proj.weight", 128, &s128), ("layers.1.attn.wq_a.weight", 32, &s32)]
        {
            assert_eq!(tensor(&ck, name), Some((vec![rows, cols], fp8(block, s))), "{name}");
        }
        assert_eq!(tensor(&ck, "layers.2.attn.wq_a.weight"), None, "no scales beside it");
        // (fp32 block scales as transformers' fine-grained FP8 names them, of a shape not a multiple of the block: each value one fp32 product)
        let (r2, c2) = (192usize, 320usize);
        let q2: Vec<u8> = (0..r2 * c2).map(|i| (i * 53 % 251) as u8).map(|b| if b & 0x7F == 0x7F { 0x7E } else { b }).collect();
        let s2: Vec<f32> = (0..2 * 3).map(|i| 0.0137 * (i + 1) as f32).collect();
        let names2 = [
            ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "F8_E4M3", vec![r2 as u64, c2 as u64], q2.clone()),
            ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv", "F32", vec![2, 3], s2.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ("model.layers.0.mlp.down_proj.weight", "F8_E4M3", vec![r2 as u64, c2 as u64], q2.clone()),
            ("model.layers.0.mlp.down_proj.weight_scale_inv", "F32", vec![1, 3], vec![0u8; 12]),
        ];
        let dir2 = dir.join("scale_inv");
        std::fs::create_dir_all(&dir2).unwrap();
        let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = names2.iter().map(|(n, d, s, b)| (*n, *d, s.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&dir2.join("model.safetensors"), &[("format", "pt".to_string())], &refs).unwrap();
        let ck2 = Checkpoint::open(&dir2).unwrap();
        let want: Vec<f32> = (0..r2 * c2).map(|i| glyd_gpu::fp8::e4m3(q2[i]) * s2[(i / c2) / 128 * 3 + (i % c2) / 128]).collect();
        assert_eq!(tensor(&ck2, names2[0].0), Some((vec![r2, c2], want)));
        assert_eq!(tensor(&ck2, names2[2].0), None, "scales of another block's shape");
        let fp4: Vec<f32> = (0..fr * fc).map(|i| E2M1[((nibbles[i / 2] >> (4 * (i % 2))) & 15) as usize] * e8m0(runs[i / 32])).collect();
        assert_eq!(tensor(&ck, "layers.0.ffn.experts.0.w1.weight"), Some((vec![fr, fc], fp4.clone())));
        assert_eq!(tensor(&ck, "model.layers.0.mlp.experts.0.gate_proj.weight"), Some((vec![fr, fc], fp4.clone())));
        assert_eq!(tensor(&ck, "model.layers.0.mlp.experts.0.gate_proj.weight_packed"), Some((vec![fr, fc], fp4)), "the codes asked for by their own name");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// E4M3's definition in f64, a byte at a time: a sign, four bits of exponent with bias 7, three of fraction; exponent 0 subnormal.
    fn e4m3_f64(b: u8) -> f64 {
        let (e, m) = (((b >> 3) & 15) as i32, (b & 7) as f64);
        let v = if e == 0 { m / 8.0 * 2f64.powi(-6) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) };
        if b & 0x80 != 0 {
            -v
        } else {
            v
        }
    }

    /// Every E4M3 byte but the two NaNs, in turn, `n` of them from `from`.
    fn codes(from: usize, n: usize) -> Vec<u8> {
        (from..from + n).map(|i| (i % 256) as u8).map(|b| if b & 0x7F == 0x7F { b - 1 } else { b }).collect()
    }

    /// FP8 with fp32 scales a block of any size: a matrix whose rows and columns do not divide by the block (the last blocks short), every code, scales
    /// that are powers of two and that are not: each value the one product in fp32 of its code's value and its block's scale, the formula in f64 rounded
    /// once. The block is read from the scales' shape: the given block where its grid is the shape, else 128, 32 or 1 a side, a run of 32 columns of a
    /// row being [1, 32]; a shape no block makes is none.
    #[test]
    fn fp8_with_fp32_scales_a_block_of_any_size() {
        let (rows, cols, block) = (20usize, 28usize, [8usize, 8usize]);
        let q = codes(0, rows * cols);
        assert!((0..=255u8).filter(|b| b & 0x7F != 0x7F).all(|b| q.contains(&b)), "every code");
        let (gr, gc) = (rows.div_ceil(8), cols.div_ceil(8));
        let s: Vec<f32> = (0..gr * gc).map(|i| if i % 2 == 0 { 2f32.powi(i as i32 - 9) } else { 0.0137 * (i + 1) as f32 }).collect();
        let got = fp8_scaled_blocks(&q, &s, rows, cols, block).unwrap();
        for (i, g) in got.iter().enumerate() {
            let want = (e4m3_f64(q[i]) * s[(i / cols / 8) * gc + (i % cols) / 8] as f64) as f32;
            assert_eq!(g.to_bits(), want.to_bits(), "row {}, column {}", i / cols, i % cols);
        }
        assert_eq!(fp8_scaled(&q[..256 * 2], &[0.5, 0.3], 2, 256).unwrap(), fp8_scaled_blocks(&q[..256 * 2], &[0.5, 0.3], 2, 256, [128, 128]).unwrap());
        assert!(fp8_scaled_blocks(&q, &s[1..], rows, cols, block).is_err() && fp8_scaled_blocks(&q, &s, rows, cols, [0, 8]).is_err());
        assert_eq!(grid_block(rows, cols, &[3, 4], &[block], 1), Some((block, 1)));
        assert_eq!(grid_block(192, 320, &[2, 3], &[], 1), Some(([128, 128], 1)));
        assert_eq!(grid_block(192, 320, &[2, 3], &[[8, 8]], 1), Some(([128, 128], 1)), "a given block of another grid is not the scales'");
        assert_eq!(grid_block(8192, 6144, &[8192, 192], &[], 1), Some(([1, 32], 1)));
        assert_eq!(grid_block(8192, 6144, &[8192, 192], &[[1, 32]], 1), Some(([1, 32], 1)));
        assert_eq!(grid_block(192, 320, &[6, 10], &[], 1), Some(([32, 32], 1)));
        assert_eq!((grid_block(192, 320, &[1, 3], &[], 1), grid_block(192, 320, &[2], &[], 1)), (None, None));
        assert_eq!(grid_block(192, 320, &[2, 3], &[[0, 0]], 0), Some(([128, 128], 1)));
    }

    /// FP8 with a byte's power-of-two scale a run of 32 columns of a row (block [1, 32]): W = e4m3(code) x 2^(scale - 127), exactly, whatever the scale
    /// tensor is named; a row whose columns do not divide by 32 has a short last run. Every code under scales from 2^-27 to 2^13 is the formula in f64 to
    /// the bit; a scale of 255 (NaN) and a product past bf16 are refused.
    #[test]
    fn fp8_with_a_power_of_two_scale_a_run_of_32() {
        let (rows, cols) = (41usize, 80usize);
        let q = codes(3, rows * cols);
        let s: Vec<u8> = (0..rows * 3).map(|i| 100 + (i % 41) as u8).collect();
        let got = fp8_blocks(&q, &s, rows, cols, [1, 32]).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for (i, g) in got.iter().enumerate() {
            let e = s[i / cols * 3 + (i % cols) / 32];
            seen.insert((q[i], e));
            let want = e4m3_f64(q[i]) * 2f64.powi(e as i32 - 127);
            assert_eq!(
                f32::from_bits((*g as u32) << 16).to_bits(),
                (want as f32).to_bits(),
                "row {}, column {}: code {:#04x}, scale {e}",
                i / cols,
                i % cols,
                q[i]
            );
            assert_eq!(f32::from_bits((*g as u32) << 16) as f64, want, "exact");
        }
        assert!((0..=255u8).filter(|b| b & 0x7F != 0x7F).all(|b| seen.iter().any(|x| x.0 == b)) && (100..=140u8).all(|e| seen.iter().any(|x| x.1 == e)));
        assert!(fp8_blocks(&q[..32], &[255], 1, 32, [1, 32]).is_err(), "a NaN scale");
        assert!(fp8_blocks(&[0x7E; 32], &[254], 1, 32, [1, 32]).is_err(), "448 x 2^127 is past bf16");
        assert!(fp8_blocks(&q, &s[1..], rows, cols, [1, 32]).is_err());
    }

    /// A described matrix is read with the scales its description names, whatever their suffix: fp32 scales a block of 128 under `X.qscale_weight`, fp32
    /// scales of the description's own block, and a byte's power of two a run of 32 under `X.weight_scale_inv`: none of them read without the
    /// description. A matrix the description names scales for and the checkpoint holds as bf16 is the bf16; a format the description's reader does not
    /// take is read as before.
    #[test]
    fn a_described_matrix_is_read_with_the_scales_its_description_names() {
        let dir = std::env::temp_dir().join(format!("glyd-formats-described-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (r, c) = (192usize, 320usize);
        let (qa, sa): (Vec<u8>, Vec<f32>) = (codes(0, r * c), (0..6).map(|i| 2f32.powi(i - 7)).collect());
        let (rb, cb) = (20usize, 28usize);
        let (qb, sb): (Vec<u8>, Vec<f32>) = (codes(7, rb * cb), (0..12).map(|i| 0.011 * (i + 1) as f32).collect());
        let (rc, cc) = (6usize, 96usize);
        let (qc, sc): (Vec<u8>, Vec<u8>) = (codes(11, rc * cc), (0..rc * 3).map(|i| 118 + i as u8).collect());
        let bf: Vec<u8> = (0..8u16).flat_map(|i| (0x3F00 + i * 16).to_le_bytes()).collect();
        let (sa_b, sb_b) = (f32s(&sa), f32s(&sb));
        let names = [
            ("layers.0.attention.wq.weight", "F8_E4M3", vec![r as u64, c as u64], &qa[..]),
            ("layers.0.attention.wq.qscale_weight", "F32", vec![2, 3], &sa_b[..]),
            ("layers.0.attention.wo.weight", "F8_E4M3", vec![rb as u64, cb as u64], &qb[..]),
            ("layers.0.attention.wo.qscale_weight", "F32", vec![3, 4], &sb_b[..]),
            ("model.layers.0.self_attn.q_proj.weight", "F8_E4M3", vec![rc as u64, cc as u64], &qc[..]),
            ("model.layers.0.self_attn.q_proj.weight_scale_inv", "U8", vec![rc as u64, 3], &sc[..]),
            ("model.layers.0.self_attn.k_proj.weight", "BF16", vec![2, 4], &bf[..]),
            ("layers.1.attn.wq_a.weight", "F8_E4M3", vec![rc as u64, cc as u64], &qc[..]),
            ("layers.1.attn.wq_a.scale", "F8_E8M0", vec![rc as u64, 3], &sc[..]),
        ];
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &names).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        let st = Stored {
            scales: [(0, 1), (2, 3), (4, 5), (7, 8)]
                .into_iter()
                .map(|(w, s)| (names[w].0.to_string(), names[s].0.to_string()))
                .chain([("model.layers.0.self_attn.k_proj.weight".to_string(), "model.layers.0.self_attn.k_proj.weight_scale_inv".to_string())])
                .collect(),
            block: Some(8),
            shards: BTreeMap::new(),
        };
        let want_a: Vec<f32> = (0..r * c).map(|i| (e4m3_f64(qa[i]) * sa[i / c / 128 * 3 + i % c / 128] as f64) as f32).collect();
        let want_b: Vec<f32> = (0..rb * cb).map(|i| (e4m3_f64(qb[i]) * sb[i / cb / 8 * 4 + i % cb / 8] as f64) as f32).collect();
        let want_c: Vec<f32> = (0..rc * cc).map(|i| (e4m3_f64(qc[i]) * 2f64.powi(sc[i / cc * 3 + i % cc / 32] as i32 - 127)) as f32).collect();
        assert_eq!(st.tensor(&ck, names[0].0), Some((vec![r, c], want_a)), "fp32 scales a block of 128, the block read from their shape");
        assert_eq!(st.tensor(&ck, names[2].0), Some((vec![rb, cb], want_b)), "fp32 scales of the description's block");
        assert_eq!(st.tensor(&ck, names[4].0), Some((vec![rc, cc], want_c.clone())), "a byte's power of two a run of 32");
        for n in [names[0].0, names[2].0, names[4].0] {
            assert_eq!(tensor(&ck, n), None, "{n}: not read without its description");
            assert_eq!(Stored::default().tensor(&ck, n), None, "{n}: nor by a description that names no scales for it");
        }
        assert_eq!(st.tensor(&ck, names[6].0), tensor(&ck, names[6].0), "a described matrix held as bf16");
        assert!(st.tensor(&ck, names[6].0).is_some_and(|(s, v)| s == [2, 4] && v[1] == f32::from_bits(0x3F10_0000)));
        assert_eq!(
            (st.tensor(&ck, names[7].0), tensor(&ck, names[7].0)),
            (Some((vec![rc, cc], want_c.clone())), Some((vec![rc, cc], want_c))),
            "E8M0 scales as before"
        );
        // (a description whose scales are of no block's shape: not read)
        let odd = Stored { scales: [(names[0].0.to_string(), names[3].0.to_string())].into_iter().collect(), ..Stored::default() };
        assert_eq!(odd.tensor(&ck, names[0].0), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// FP8 block scales that start again at each shard of a fused matrix: a matrix of 4 shards of 21 rows in blocks of 8 holds 4 grids of 3 rows of
    /// blocks (12, where one grid over its 84 rows would be 11), a shard's last block 5 rows and the next shard's first block whole. Each value is its
    /// code's value times the scale at (shard, row in the shard / 8, column / 8), the formula in f64 rounded once, and a shard decoded alone is the
    /// same; the released shape of the family's matrix (4 shards of 3,392 rows in blocks of 128: 4 x 27 rows of blocks, not 106) is read as shards,
    /// its layers whose shards are whole blocks (4 x 29 = 116) either way, and a shape neither reading makes is none.
    #[test]
    fn fp8_block_scales_start_again_at_each_shard() {
        let (shards, each, cols, block) = (4usize, 21usize, 28usize, [8usize, 8usize]);
        let rows = shards * each;
        let (gr, gc) = (each.div_ceil(8), cols.div_ceil(8));
        let q = codes(5, rows * cols);
        let s: Vec<f32> = (0..shards * gr * gc).map(|i| if i % 3 == 0 { 0.0173 * (i + 1) as f32 } else { 2f32.powi(i as i32 % 11 - 8) }).collect();
        let got = fp8_scaled_shards(&q, &s, rows, cols, block, shards).unwrap();
        for (i, g) in got.iter().enumerate() {
            let (r, c) = (i / cols, i % cols);
            let want = (e4m3_f64(q[i]) * s[(r / each) * gr * gc + (r % each) / 8 * gc + c / 8] as f64) as f32;
            assert_eq!(g.to_bits(), want.to_bits(), "row {r}, column {c}");
        }
        for k in 0..shards {
            let alone = fp8_scaled_blocks(&q[k * each * cols..(k + 1) * each * cols], &s[k * gr * gc..(k + 1) * gr * gc], each, cols, block).unwrap();
            assert_eq!(alone, got[k * each * cols..(k + 1) * each * cols], "shard {k}");
        }
        // (one grid over the whole matrix is another reading: row 21, the second shard's first, is in the whole matrix's third block of rows)
        let whole = fp8_scaled_blocks(&q, &s[..rows.div_ceil(8) * gc], rows, cols, block).unwrap();
        assert_ne!(whole[each * cols..(each + 3) * cols], got[each * cols..(each + 3) * cols]);
        assert!(fp8_scaled_shards(&q, &s, rows, cols, block, 5).is_err() && fp8_scaled_shards(&q, &s[..s.len() - 1], rows, cols, block, shards).is_err());
        assert_eq!(fp8_scaled_shards(&q, &s[..rows.div_ceil(8) * gc], rows, cols, block, 1).unwrap(), whole);
        assert_eq!(grid_block(rows, cols, &[12, 4], &[block], shards), Some((block, shards)));
        assert_eq!(grid_block(rows, cols, &[11, 4], &[block], shards), Some((block, 1)), "one grid over the whole matrix");
        assert_eq!(grid_block(rows, cols, &[12, 4], &[block], 1), None, "a grid a shard, of a matrix the description does not cut");
        assert_eq!(grid_block(13568, 4096, &[108, 32], &[[128, 128]], 4), Some(([128, 128], 4)));
        assert_eq!(grid_block(13568, 4096, &[108, 32], &[], 4), Some(([128, 128], 4)), "the block from the shape alone");
        assert_eq!(grid_block(14848, 4096, &[116, 32], &[[128, 128]], 4), Some(([128, 128], 4)), "shards of whole blocks: either reading");
        assert_eq!(grid_block(13568, 4096, &[107, 32], &[[128, 128]], 4), None);
        // (two shards of 100 rows in blocks of 128: two rows of blocks either way, and the description's shards decide)
        assert_eq!((grid_block(200, 128, &[2, 1], &[], 2), grid_block(200, 128, &[2, 1], &[], 1)), (Some(([128, 128], 2)), Some(([128, 128], 1))));
    }

    /// A described matrix cut in shards is read a shard at a time where its scales are a grid a shard, under fp32 scales and under a byte's power of
    /// two alike: the values of `fp8_scaled_shards`, and each shard's of `fp8_blocks`. Without the description's shards it is not read.
    #[test]
    fn a_described_matrix_in_shards_is_read_a_shard_at_a_time() {
        let dir = std::env::temp_dir().join(format!("glyd-formats-shards-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"quantization_config": {"weight_block_size": [8, 8]}}"#).unwrap();
        let (shards, each, cols) = (2usize, 28usize, 32usize);
        let rows = shards * each;
        let q = codes(9, rows * cols);
        let s: Vec<f32> = (0..shards * 4 * 4).map(|i| 2f32.powi(i as i32 % 7 - 6)).collect();
        let e: Vec<u8> = (0..shards * 4 * 4).map(|i| 120 + i as u8 % 9).collect();
        let s_b: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (w, we) = ("model.layers.0.self_attn.qkv_proj.weight", "model.layers.1.self_attn.qkv_proj.weight");
        let (sn, sne) = (format!("{w}_scale_inv"), format!("{we}_scale_inv"));
        let names = [
            (w, "F8_E4M3", vec![rows as u64, cols as u64], &q[..]),
            (&sn[..], "F32", vec![8, 4], &s_b[..]),
            (we, "F8_E4M3", vec![rows as u64, cols as u64], &q[..]),
            (&sne[..], "U8", vec![8, 4], &e[..]),
        ];
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &names).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        assert_eq!(ck.fp8_block, Some([8, 8]));
        let scales: BTreeMap<String, String> = [(w.to_string(), sn.clone()), (we.to_string(), sne.clone())].into_iter().collect();
        let st = Stored { scales: scales.clone(), block: None, shards: [(w.to_string(), 2), (we.to_string(), 2)].into_iter().collect() };
        assert_eq!(st.tensor(&ck, w), Some((vec![rows, cols], fp8_scaled_shards(&q, &s, rows, cols, [8, 8], 2).unwrap())));
        let by_shard: Vec<f32> = (0..2)
            .flat_map(|k| fp8_blocks(&q[k * each * cols..(k + 1) * each * cols], &e[k * 16..(k + 1) * 16], each, cols, [8, 8]).unwrap())
            .map(|x| f32::from_bits((x as u32) << 16))
            .collect();
        let want: Vec<f32> = (0..rows * cols)
            .map(|i| (e4m3_f64(q[i]) * 2f64.powi(e[(i / cols / each) * 16 + (i / cols % each) / 8 * 4 + i % cols / 8] as i32 - 127)) as f32)
            .collect();
        assert_eq!(by_shard, want);
        assert_eq!(st.tensor(&ck, we), Some((vec![rows, cols], want)));
        let uncut = Stored { scales, ..Stored::default() };
        assert_eq!((uncut.tensor(&ck, w), uncut.tensor(&ck, we), tensor(&ck, w)), (None, None, None), "8 rows of blocks, where the whole matrix has 7");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// E2M1's definition in f64: a sign, two bits of exponent with bias 1, one of fraction; exponent 0 subnormal (0 or a half).
    fn e2m1_f64(c: u8) -> f64 {
        let (e, m) = (((c >> 1) & 3) as i32, (c & 1) as f64);
        let v = if e == 0 { m / 2.0 } else { (1.0 + m / 2.0) * 2f64.powi(e - 1) };
        if c & 8 != 0 {
            -v
        } else {
            v
        }
    }

    /// MXFP4 under a release's own names: 4-bit codes two a byte along a row, the even column in the low four bits, and a byte's power-of-two scale a
    /// run of 32 (W = e2m1(code) x 2^(scale - 127)). Every pair of codes in a byte under scales from 2^-127 to 2^125 (6 x 2^125 is within fp32) is the formula in f64, to the bit;
    /// a byte of codes 1 and 2 is 0.5 then 1, not 1 then 0.5. Stored as `X.weight` (U8) with `X.weight_scale`, the matrix is read by a description that
    /// names the scales so and not without one; under compressed-tensors' names it is read as before, by either.
    #[test]
    fn mxfp4_codes_under_a_releases_own_names() {
        let (rows, cols) = (8usize, 64usize);
        let packed: Vec<u8> = (0..rows * cols / 2).map(|i| (i % 256) as u8).collect();
        let runs: Vec<u8> = vec![0, 1, 2, 60, 100, 119, 120, 121, 124, 125, 126, 127, 128, 130, 200, 252];
        let got = mxfp4(&packed, &runs, rows, cols).unwrap();
        for (i, g) in got.iter().enumerate() {
            let code = (packed[i / 2] >> (4 * (i % 2))) & 15;
            let want = e2m1_f64(code) * 2f64.powi(runs[i / 32] as i32 - 127);
            assert_eq!((g.to_bits(), *g as f64), ((want as f32).to_bits(), want), "weight {i}: code {code}, scale {}", runs[i / 32]);
        }
        assert_eq!(mxfp4(&[0x21; 16], &[127], 1, 32).unwrap()[..2], [0.5, 1.0], "the even column in the low four bits");
        assert!((0..16u8).all(|c| E2M1[c as usize] as f64 == e2m1_f64(c) && (E2M1[c as usize] == 0.0) == (c & 7 == 0)));
        assert!(
            mxfp4(&packed, &runs[1..], rows, cols).is_err()
                && mxfp4(&packed[..48], &[127; 3], 1, 96).is_ok()
                && mxfp4(&packed[..24], &[127; 2], 1, 48).is_err()
        );
        let dir = std::env::temp_dir().join(format!("glyd-formats-mxfp4-names-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (own, ct) = ("model.layers.1.mlp.experts.0.gate_proj", "model.layers.1.mlp.experts.1.gate_proj");
        let names = [
            (format!("{own}.weight"), "U8", vec![rows as u64, cols as u64 / 2], &packed[..]),
            (format!("{own}.weight_scale"), "U8", vec![rows as u64, cols as u64 / 32], &runs[..]),
            (format!("{ct}.weight_packed"), "U8", vec![rows as u64, cols as u64 / 2], &packed[..]),
            (format!("{ct}.weight_scale"), "U8", vec![rows as u64, cols as u64 / 32], &runs[..]),
            ("model.layers.1.mlp.gate.weight".to_string(), "U8", vec![rows as u64, cols as u64 / 2], &packed[..]),
            ("model.layers.1.mlp.gate.weight_scale".to_string(), "U8", vec![rows as u64, 1], &runs[..rows]),
        ];
        let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = names.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), *b)).collect();
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &refs).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        let st = Stored { scales: [0, 2, 4].into_iter().map(|i| (names[i].0.clone(), names[i + 1].0.clone())).collect(), ..Stored::default() };
        assert_eq!(st.tensor(&ck, &names[0].0), Some((vec![rows, cols], got.clone())));
        assert_eq!((tensor(&ck, &names[0].0), Stored::default().tensor(&ck, &names[0].0)), (None, None), "bytes, until a description names their scales");
        assert_eq!((st.tensor(&ck, &names[2].0), tensor(&ck, &names[2].0)), (Some((vec![rows, cols], got.clone())), Some((vec![rows, cols], got))));
        assert_eq!(st.tensor(&ck, &names[4].0), None, "scales of another run's shape");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
