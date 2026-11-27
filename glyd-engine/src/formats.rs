//! A release's quantized matrices decoded exactly, by the type they are stored in and the scales stored beside them: FP8 (E4M3) with a power-of-two scale
//! (E8M0) a block of the matrix, the block read from the scales' shape; FP4 (E2M1, two weights a byte, the first in the low four bits) with an E8M0 scale a
//! run of 32 of a row, as a release names it (`X.weight` stored as I8, its scales `X.scale`) or as compressed-tensors packs it (MXFP4: `X.weight_packed`,
//! `X.weight_scale`). A weight times a power of two is exact in f32, and in bf16 where it is in range. And a table read by rows stored at 4 bits (NF4 levels
//! times a bf16 scale a row: `X.q4`), as v0.34.2 decoded it.

use crate::ffi::{err, Error, Result};
use glyd_gpu::safetensors::Checkpoint;

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
    let gc = cols.div_ceil(128);
    if q.len() != rows * cols || scales.len() != rows.div_ceil(128) * gc {
        return err(format!("{} bytes and {} scales for an FP8 matrix of {rows} x {cols} in blocks of 128", q.len(), scales.len()));
    }
    Ok(q.iter().enumerate().map(|(i, &b)| glyd_gpu::fp8::e4m3(b) * scales[(i / cols) / 128 * gc + (i % cols) / 128]).collect())
}

/// A checkpoint's tensor as f32, by the type it is stored in: bf16 and f32 as stored, int64 (a hash table's ids) as their values; an FP8 matrix with its
/// scales (`X.scale`, a block each, or fp32 block scales `X.weight_scale_inv`: `scales_beside`); an FP4 matrix stored as I8 with its runs' scales
/// (`X.scale`); and where `X.weight` or its codes `X.weight_packed` are asked for and the checkpoint holds the matrix as compressed-tensors packs it, that
/// MXFP4 matrix, or the table stored at 4 bits (`X.q4`) as the bf16 values it decodes to (`nf4_row`). Another type, or a quantized matrix without its
/// scales: none.
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
}
