//! FP8 (E4M3) matrices with block scales, as zai-org/GLM-5.3-Flash holds its experts, dense and shared MLPs and latent-attention projections: `NAME.weight`, F8_E4M3 [rows, cols], and
//! `NAME.weight_scale_inv`, F32 [ceil(rows / bm), ceil(cols / bn)] for blocks of bm rows by bn columns (config.json's `quantization_config.weight_block_size`; the last blocks of a row or a column are
//! short where the matrix does not divide). A weight is its byte's value times its block's scale, and a matrix is read as bf16: the f32 product rounded once to nearest, ties to even, as transformers
//! 5.19.0's `Fp8Dequantize` makes it (`(q * s).to(bfloat16)` with both in f32).

/// The value of an E4M3 byte, the OCP FP8 format (NVIDIA's e4m3fn): a sign, four bits of exponent with bias 7 and three of fraction; no infinity; exponent 0 is subnormal (2^-6 times the fraction in eighths);
/// S.1111.111 is NaN (0x7F, 0xFF).
pub const fn e4m3(b: u8) -> f32 {
    let (sign, e, m) = ((b >> 7) as u32, ((b >> 3) & 15) as u32, (b & 7) as u32);
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let magnitude = if e == 0 {
        // (m / 8) * 2^-6: m times 2^-9, exact
        m as f32 * f32::from_bits(0x3B00_0000)
    } else {
        // (1 + m / 8) * 2^(e - 7): the f32 with exponent e - 7 + 127 and the fraction's three bits at the top of its 23
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    f32::from_bits(magnitude.to_bits() | sign << 31)
}

/// Every byte's value (`e4m3`).
pub static E4M3: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = e4m3(i as u8);
        i += 1;
    }
    t
};

/// A finite f32 rounded to bf16, to nearest and ties to even (torch's `.to(torch.bfloat16)`).
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// The scales' grid for a `rows` x `cols` matrix in blocks of `block` = [bm, bn]: [ceil(rows / bm), ceil(cols / bn)].
pub fn grid(rows: usize, cols: usize, block: [usize; 2]) -> [usize; 2] {
    [rows.div_ceil(block[0].max(1)), cols.div_ceil(block[1].max(1))]
}

/// A [rows, cols] matrix of E4M3 bytes `q` with its block scales (`scales`, row-major over `grid`) as bf16 words in `out`: each weight the f32 product of its byte's value and its block's scale, rounded to bf16.
/// An Err, naming the place, where a weight would not be finite: a scale that is not (NaN or infinite), a NaN byte, or a product past bf16's range.
pub fn dequant(q: &[u8], scales: &[f32], rows: usize, cols: usize, block: [usize; 2], out: &mut [u16]) -> Result<(), String> {
    let [gr, gc] = grid(rows, cols, block);
    if block[0] == 0 || block[1] == 0 || q.len() != rows * cols || out.len() != rows * cols || scales.len() != gr * gc {
        return Err(format!(
            "{} bytes, {} scales and {} words for a {rows} x {cols} matrix in blocks of {} x {}",
            q.len(),
            scales.len(),
            out.len(),
            block[0],
            block[1]
        ));
    }
    if let Some(i) = scales.iter().position(|s| !s.is_finite()) {
        return Err(format!("the scale of block ({}, {}) is {}", i / gc, i % gc, scales[i]));
    }
    let mut bad = false;
    for r in 0..rows {
        let (qr, or) = (&q[r * cols..(r + 1) * cols], &mut out[r * cols..(r + 1) * cols]);
        for (j, &s) in scales[r / block[0] * gc..][..gc].iter().enumerate() {
            let (a, b) = (j * block[1], ((j + 1) * block[1]).min(cols));
            for (w, &byte) in or[a..b].iter_mut().zip(&qr[a..b]) {
                *w = bf16(E4M3[byte as usize] * s);
                bad |= *w & 0x7F80 == 0x7F80;
            }
        }
    }
    if bad {
        // (the first such weight, for the message)
        let at = out.iter().position(|w| w & 0x7F80 == 0x7F80).unwrap_or(0);
        let (r, c) = (at / cols, at % cols);
        return Err(format!("the weight at row {r}, column {c} is not finite (byte {:#04x}, scale {})", q[at], scales[r / block[0] * gc + c / block[1]]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The format's definition in f64, a byte at a time (None for the two NaNs).
    fn reference(b: u8) -> Option<f64> {
        let (s, e, m) = ((b >> 7) as i32, ((b >> 3) & 15) as i32, (b & 7) as i32);
        let sign = if s == 1 { -1.0 } else { 1.0 };
        match (e, m) {
            (15, 7) => None,
            (0, _) => Some(sign * m as f64 / 8.0 * 2f64.powi(-6)),
            _ => Some(sign * (1.0 + m as f64 / 8.0) * 2f64.powi(e - 7)),
        }
    }

    /// All 256 bytes against the definition (0x7F and 0xFF are the NaNs), and the values the format is known by: 448 the largest, 2^-9 the smallest, 1.0 at 0x38, minus zero at 0x80.
    #[test]
    fn the_whole_table_is_the_formats() {
        let mut nan = Vec::new();
        for b in 0..=255u8 {
            match reference(b) {
                None => {
                    assert!(E4M3[b as usize].is_nan());
                    nan.push(b);
                }
                Some(v) => assert_eq!(E4M3[b as usize].to_bits(), (v as f32).to_bits(), "byte {b:#04x}: {} where {v} is wanted", E4M3[b as usize]),
            }
        }
        assert_eq!(nan, [0x7F, 0xFF]);
        assert_eq!(
            (E4M3[0x7E], E4M3[0xFE], E4M3[0x01], E4M3[0x38], E4M3[0x08], E4M3[0x07]),
            (448.0, -448.0, 2f32.powi(-9), 1.0, 2f32.powi(-6), 7.0 * 2f32.powi(-9))
        );
        assert_eq!((E4M3[0x00].to_bits(), E4M3[0x80].to_bits()), (0, 0x8000_0000));
        // 254 finite bytes: 126 values of each sign and the two zeros; the codes run up with the values
        assert_eq!(E4M3.iter().filter(|v| v.is_finite()).count(), 254);
        assert!(E4M3[..0x7F].windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // 1 + 2^-8 is half a bf16 step above 1: the tie goes to the even word 0x3F80, just over it up, and 1 + 3 * 2^-8 (a tie between 0x3F81 and 0x3F82) to the even one above
        assert_eq!(bf16(1.0 + 2f32.powi(-8)), 0x3F80);
        assert_eq!(bf16(1.0 + 2f32.powi(-8) + 2f32.powi(-20)), 0x3F81);
        assert_eq!(bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        assert_eq!(bf16(-448.0), 0xC3E0);
    }

    /// Bytes of a spread of values, none of them NaN.
    fn bytes(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| match ((i.wrapping_mul(2654435761) >> 9) & 0xFF) as u8 {
                0x7F => 0x7E,
                0xFF => 0xFE,
                b => b,
            })
            .collect()
    }

    /// One weight from the grid by division: the product the matrix is read as.
    fn one(q: &[u8], scales: &[f32], cols: usize, block: [usize; 2], r: usize, c: usize) -> u16 {
        bf16(E4M3[q[r * cols + c] as usize] * scales[(r / block[0]) * cols.div_ceil(block[1]) + c / block[1]])
    }

    /// A matrix that does not divide into its blocks, 200 x 300 in blocks of 128 x 128 (a grid of 2 x 3: the last row of blocks has 72 rows, the last column of blocks 44 columns), a scale of its own to each block.
    #[test]
    fn ragged_blocks_each_have_their_scale() {
        let (rows, cols, block) = (200, 300, [128, 128]);
        assert_eq!(grid(rows, cols, block), [2, 3]);
        let q = bytes(rows * cols);
        let scales = [0.5, 0.003, 7.0, 1.0 / 3.0, 2f32.powi(-9), 100.0];
        let mut out = vec![0u16; rows * cols];
        dequant(&q, &scales, rows, cols, block, &mut out).unwrap();
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(out[r * cols + c], one(&q, &scales, cols, block, r, c), "({r}, {c})");
            }
        }
        // a block's scale is used by its weights alone: with block (1, 1)'s changed, only its nonzero weights (rows 128 to 199, columns 128 to 255) change
        let mut other = scales;
        other[4] = 2.0;
        let mut out2 = vec![0u16; rows * cols];
        dequant(&q, &other, rows, cols, block, &mut out2).unwrap();
        let (mut changed, mut same_zero) = (0, 0);
        for r in 0..rows {
            for c in 0..cols {
                let (inside, zero) = (r >= 128 && (128..256).contains(&c), E4M3[q[r * cols + c] as usize] == 0.0);
                assert_eq!(out[r * cols + c] == out2[r * cols + c], !inside || zero, "({r}, {c})");
                changed += (inside && !zero) as usize;
                same_zero += (inside && zero) as usize;
            }
        }
        assert!(changed > 72 * 128 / 2 && same_zero > 0, "{changed} weights changed, {same_zero} zeros kept");
    }

    /// Blocks of other sizes (the config's `weight_block_size` is read, not assumed to be 128), whole or short: 4 x 6 in blocks of 2 x 3 (a grid of 2 x 2), in blocks of 3 x 4 (2 x 2, both short), in one block.
    #[test]
    fn the_block_is_the_configs() {
        let (rows, cols) = (4, 6);
        let q = bytes(rows * cols);
        for (block, scales) in [
            ([2, 3], vec![1.0, 2.0, 4.0, 8.0]),
            ([3, 4], vec![0.25, 0.5, 3.0, 5.0]),
            ([4, 6], vec![0.125]),
            ([1, 1], (1..=24).map(|i| i as f32 / 8.0).collect()),
        ] {
            assert_eq!(scales.len(), grid(rows, cols, block).iter().product::<usize>());
            let mut out = vec![0u16; rows * cols];
            dequant(&q, &scales, rows, cols, block, &mut out).unwrap();
            assert!((0..rows).all(|r| (0..cols).all(|c| out[r * cols + c] == one(&q, &scales, cols, block, r, c))), "{block:?}");
        }
        // scales for another grid than the matrix's blocks make are refused
        let mut out = vec![0u16; rows * cols];
        assert!(dequant(&q, &[1.0, 2.0, 4.0, 8.0], rows, cols, [4, 6], &mut out).unwrap_err().contains("4 scales"));
    }

    #[test]
    fn what_is_refused() {
        let q = vec![0x38u8; 16];
        let run = |scales: &[f32], q: &[u8]| dequant(q, scales, 4, 4, [2, 4], &mut [0u16; 16]);
        run(&[1.0, 1.0], &q).unwrap();
        assert!(run(&[1.0], &q).unwrap_err().contains("1 scales"));
        assert!(run(&[1.0, f32::NAN], &q).unwrap_err().contains("scale of block (1, 0) is NaN"));
        assert!(run(&[f32::INFINITY, 1.0], &q).unwrap_err().contains("scale of block (0, 0) is inf"));
        let mut nan = q.clone();
        nan[6] = 0x7F;
        assert!(run(&[1.0, 1.0], &nan).unwrap_err().contains("row 1, column 2 is not finite (byte 0x7f, scale 1)"));
        // a product past the range of bf16 (448 times the largest f32 overflows)
        let mut big = q.clone();
        big[0] = 0x7E;
        assert!(run(&[f32::MAX, 1.0], &big).unwrap_err().contains("row 0, column 0 is not finite"));
        assert!(dequant(&q, &[1.0, 1.0], 4, 4, [0, 4], &mut [0u16; 16]).is_err());
        assert!(dequant(&q[..15], &[1.0, 1.0], 4, 4, [2, 4], &mut [0u16; 16]).is_err());
    }
}
