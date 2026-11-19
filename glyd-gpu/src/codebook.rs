//! The Hummingbird codebook row layouts (glyd-v7 entries `c3r` and `c2r`; Linear GLY-98, GLY-134): a matrix of neuron rows, each row `cols` values (1,024 for Nemotron 3 Super's experts) held as indices into a codebook of the
//! row's own. A row is: its codes, `bits` (3 or 2) bits each in a little-endian bitstream (code j at stream bits [bits j, bits j + bits), stream bit i being bit i mod 8 of byte i / 8: at 2 bits byte b holds codes 4b .. 4b + 3,
//! at 3 bits codes 8g .. 8g + 7 are the 24-bit little-endian integer of bytes 3g .. 3g + 2); a bf16 scale for each run of 32 values; and 2^bits bf16 codebook entries, in units of the run's scale. A weight is
//! `scale[n][j / 32] * codebook[n][code[n][j]]`, the product of two bf16 taken exactly in f32 (the engine's kernels multiply by that; [`unpack_codebook_rows`] rounds it to bf16 once, which is what a pack's glyd.json
//! hashes). The weights are those of the matrix in the rotated basis of its layer (one rotation H diag(signs) on the row axis, shared by the layer's experts; the signs are a tensor of the directory, `glyd_rot_signs`).
//!
//! The three arrays are the pack's three buffers: `data` the codes (`cols bits / 8` bytes a row), `blocks` the scales (`cols / 16` bytes a row), `codebook` the entries (`2^bits 2` bytes a row); 328 bytes a row at 2 bits
//! and 464 at 3 (cols 1,024: 2.5625 and 3.625 bits a weight), an expert's two matrices of 2,688 rows 1,763,328 and 2,494,464 bytes. Tags: [`CB3_ROW_TAG`], [`CB2_ROW_TAG`] (tiers[0]'s top byte, tiers [tag << 24, 0, 0]).
//! The experts' down matrix is held transposed, as its neurons' rows (its tensor in glyd.json is the weight [out, in], the pack its transpose); an up matrix is held as it is.

use crate::pack::{CB2_ROW_TAG, CB3_ROW_TAG};

/// The bits of a code in a codebook row layout, by its tag.
pub fn cb_rows_bits(tag: u32) -> Option<u32> {
    match tag {
        CB3_ROW_TAG => Some(3),
        CB2_ROW_TAG => Some(2),
        _ => None,
    }
}

/// The tag of the codebook row layout of `bits` (2 or 3).
pub fn cb_rows_tag(bits: u32) -> Option<u32> {
    match bits {
        3 => Some(CB3_ROW_TAG),
        2 => Some(CB2_ROW_TAG),
        _ => None,
    }
}

/// The bytes of a row of `cols` values at `bits`: its codes, its scales, its codebook.
pub fn cb_strides(bits: u32, cols: usize) -> (usize, usize, usize) {
    (cols * bits as usize / 8, cols / 16, 2usize << bits)
}

/// Whether a bf16 is a finite number.
fn finite(x: u16) -> bool {
    x & 0x7F80 != 0x7F80
}

fn bf16_f32(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

/// A finite f32 rounded to bf16, to nearest and ties to even (an infinity stays one).
fn f32_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// A matrix of `rows` neuron rows of `cols` values (a multiple of 32) in the codebook row layout of `bits` (2 or 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CbRows {
    pub rows: usize,
    pub cols: usize,
    pub bits: u32,
    /// rows x `cols bits / 8` bytes: the codes' bitstreams.
    pub codes: Vec<u8>,
    /// rows x `cols / 32` little-endian bf16: the runs' scales.
    pub scales: Vec<u8>,
    /// rows x `2^bits` little-endian bf16: the rows' codebooks.
    pub codebook: Vec<u8>,
}

impl CbRows {
    /// The matrix whose three arrays are given as the layout holds them (the codes already packed): None where the shape or a length is not a layout's, or a scale or an entry is not finite.
    pub fn new(bits: u32, rows: usize, cols: usize, codes: Vec<u8>, scales: Vec<u8>, codebook: Vec<u8>) -> Option<CbRows> {
        let p = CbRows { rows, cols, bits, codes, scales, codebook };
        (p.is_valid() && p.scales.chunks_exact(2).chain(p.codebook.chunks_exact(2)).all(|b| finite(u16::from_le_bytes([b[0], b[1]])))).then_some(p)
    }

    /// Whether the fields are a layout's: 2 or 3 bits, rows, columns in runs of 32, and the arrays' lengths.
    pub fn is_valid(&self) -> bool {
        let (c, s, b) = cb_strides(self.bits, self.cols);
        (2..=3).contains(&self.bits) && self.rows > 0 && self.cols > 0 && self.cols.is_multiple_of(32) && self.codes.len() == self.rows * c && self.scales.len() == self.rows * s && self.codebook.len() == self.rows * b
    }

    /// The tag in tiers[0]'s top byte.
    pub fn tag(&self) -> u32 {
        cb_rows_tag(self.bits).expect("2 or 3 bits")
    }

    /// The words of the pack: [tag << 24, 0, 0].
    pub fn tiers(&self) -> [u32; 3] {
        [self.tag() << 24, 0, 0]
    }

    /// Bytes of the three arrays.
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() + self.codebook.len()
    }

    /// The code of weight (n, j), read by the layout's formula.
    pub fn code(&self, n: usize, j: usize) -> u8 {
        let (stride, bits) = (cb_strides(self.bits, self.cols).0, self.bits as usize);
        let row = &self.codes[n * stride..(n + 1) * stride];
        let at = bits * j;
        let two = u16::from_le_bytes([row[at / 8], row.get(at / 8 + 1).copied().unwrap_or(0)]);
        ((two >> (at % 8)) & ((1 << bits) - 1)) as u8
    }

    /// The scale of run k of row n (bf16 bits).
    pub fn scale(&self, n: usize, k: usize) -> u16 {
        let at = 2 * (n * (self.cols / 32) + k);
        u16::from_le_bytes([self.scales[at], self.scales[at + 1]])
    }

    /// Entry c of row n's codebook (bf16 bits).
    pub fn entry(&self, n: usize, c: usize) -> u16 {
        let at = 2 * (n * (1 << self.bits) + c);
        u16::from_le_bytes([self.codebook[at], self.codebook[at + 1]])
    }

    /// The weight (n, j) the kernels multiply by: scale times codebook entry, exact in f32.
    pub fn value(&self, n: usize, j: usize) -> f32 {
        bf16_f32(self.scale(n, j / 32)) * bf16_f32(self.entry(n, self.code(n, j) as usize))
    }
}

/// A row's codes (one index a byte) as the layout's bitstream.
fn pack_row(bits: u32, codes: &[u8], out: &mut [u8]) {
    if bits == 2 {
        for (o, c) in out.iter_mut().zip(codes.chunks_exact(4)) {
            *o = c[0] | c[1] << 2 | c[2] << 4 | c[3] << 6;
        }
    } else {
        for (o, c) in out.chunks_exact_mut(3).zip(codes.chunks_exact(8)) {
            let w = c.iter().enumerate().fold(0u32, |w, (m, &x)| w | (x as u32) << (3 * m));
            o.copy_from_slice(&w.to_le_bytes()[..3]);
        }
    }
}

/// The matrix whose weight (n, j) has the code `codes[n cols + j]` (an index below 2^bits), the scale `scales[n cols / 32 + j / 32]` and the codebook `codebook[n 2^bits ..][..2^bits]` (bf16 bits), in the layout of `bits` (2
/// or 3): the exported tensors of a calibration packed as they are. None where the shape is not one (rows and columns in runs of 32), a length is not the shape's, a code is past the codebook, or a scale or an entry is
/// not finite.
pub fn pack_codebook_rows(codes: &[u8], scales: &[u16], codebook: &[u16], bits: u32, rows: usize, cols: usize) -> Option<CbRows> {
    let (c, s, b) = cb_strides(bits, cols);
    if !(2..=3).contains(&bits) || rows == 0 || cols == 0 || !cols.is_multiple_of(32) || codes.len() != rows * cols || scales.len() != rows * cols / 32 || codebook.len() != rows << bits {
        return None;
    }
    if codes.iter().any(|&u| (u as u32) >> bits != 0) {
        return None;
    }
    let mut data = vec![0u8; rows * c];
    for (row, out) in codes.chunks_exact(cols).zip(data.chunks_exact_mut(c)) {
        pack_row(bits, row, out);
    }
    let le = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let p = CbRows::new(bits, rows, cols, data, le(scales), le(codebook))?;
    debug_assert_eq!((p.codes.len(), p.scales.len(), p.codebook.len()), (rows * c, rows * s, rows * b));
    Some(p)
}

/// The codes of a matrix, one index a byte, row-major (what [`pack_codebook_rows`] was given).
pub fn unpack_codebook_codes(p: &CbRows) -> Vec<u8> {
    let (stride, bits) = (cb_strides(p.bits, p.cols).0, p.bits);
    let mut out = vec![0u8; p.rows * p.cols];
    for (row, o) in p.codes.chunks_exact(stride).zip(out.chunks_exact_mut(p.cols)) {
        if bits == 2 {
            for (b, o) in row.iter().zip(o.chunks_exact_mut(4)) {
                o.iter_mut().enumerate().for_each(|(m, x)| *x = (b >> (2 * m)) & 3);
            }
        } else {
            for (b, o) in row.chunks_exact(3).zip(o.chunks_exact_mut(8)) {
                let w = u32::from_le_bytes([b[0], b[1], b[2], 0]);
                o.iter_mut().enumerate().for_each(|(m, x)| *x = ((w >> (3 * m)) & 7) as u8);
            }
        }
    }
    out
}

/// The weights the matrix decodes to, exact in f32 (`scale * codebook[code]`, what the kernels multiply by), row-major [rows x cols]; false where `out` or the matrix is not a layout's.
pub fn decode_codebook_rows(p: &CbRows, out: &mut [f32]) -> bool {
    if !p.is_valid() || out.len() != p.rows * p.cols {
        return false;
    }
    let codes = unpack_codebook_codes(p);
    for (n, (cs, o)) in codes.chunks_exact(p.cols).zip(out.chunks_exact_mut(p.cols)).enumerate() {
        let book: Vec<f32> = (0..1usize << p.bits).map(|c| bf16_f32(p.entry(n, c))).collect();
        for (j, (&c, x)) in cs.iter().zip(o.iter_mut()).enumerate() {
            *x = bf16_f32(p.scale(n, j / 32)) * book[c as usize];
        }
    }
    true
}

/// The weights as bf16, each exact product rounded once to nearest even (what glyd.json's sha256 of a codebook row pack is of, and `glyd-gpu verify` checks); false where `out` or the matrix is not a layout's.
pub fn unpack_codebook_rows(p: &CbRows, out: &mut [u16]) -> bool {
    if !p.is_valid() || out.len() != p.rows * p.cols {
        return false;
    }
    let mut w = vec![0f32; out.len()];
    if !decode_codebook_rows(p, &mut w) {
        return false;
    }
    out.iter_mut().zip(&w).for_each(|(o, &x)| *o = f32_bf16(x));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{rows_bits, INT4_ROW_TAG};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
        }
    }

    fn bits_of(x: f32) -> u16 {
        f32_bf16(x)
    }

    /// Random codes, scales (some zero, some negative) and codebooks (sorted, in units of the scale, both signs) of a matrix.
    fn random(rng: &mut Rng, bits: u32, rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
        let codes = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
        let scales = (0..rows * cols / 32).map(|i| if i % 11 == 0 { 0 } else { bits_of(rng.f(0.001, 0.05) * if i % 5 == 0 { -1.0 } else { 1.0 }) }).collect();
        let codebook = (0..rows << bits).map(|_| bits_of(rng.f(-1.0, 1.0))).collect();
        (codes, scales, codebook)
    }

    /// The bytes of a run, written out by hand from the layout's text: at 2 bits byte b holds codes 4b .. 4b + 3 from its low bits; at 3 bits eight codes are the 24-bit little-endian integer of three bytes, so that the third
    /// code straddles the first two bytes; a run of 32 is 8 and 12 bytes.
    #[test]
    fn the_bytes_of_a_run() {
        // 2 bits: codes 0 1 2 3 | 3 2 1 0 -> 0b11_10_01_00 = 0xE4, 0b00_01_10_11 = 0x1B
        let c2: Vec<u8> = (0..32).map(|i| [0, 1, 2, 3, 3, 2, 1, 0][i % 8]).collect();
        let p = pack_codebook_rows(&c2, &[0x3f80], &[0, 0, 0, 0], 2, 1, 32).unwrap();
        assert_eq!(&p.codes[..2], &[0xE4, 0x1B]);
        assert_eq!(p.codes.len(), 8);
        // 3 bits: codes 1 2 3 4 5 6 7 0 = 1 + 2 << 3 + 3 << 6 + 4 << 9 + 5 << 12 + 6 << 15 + 7 << 18 = 0x1F58D1 -> D1 58 1F (code 2 is 011 at bits 6 .. 8: bits 6 and 7 of byte 0 and bit 0 of byte 1)
        let c3: Vec<u8> = (0..32).map(|i| [1, 2, 3, 4, 5, 6, 7, 0][i % 8]).collect();
        let p = pack_codebook_rows(&c3, &[0x3f80], &[0; 8], 3, 1, 32).unwrap();
        assert_eq!(&p.codes[..3], &[0xD1, 0x58, 0x1F]);
        assert_eq!(p.codes.len(), 12);
        assert_eq!((0..32).map(|j| p.code(0, j)).collect::<Vec<_>>(), c3);
    }

    /// Pack, read back and dequantize: the codes come back (by `code` and by `unpack_codebook_codes`), every weight is scale times entry exactly, the bf16 decode is that product rounded once, at both widths and the shapes the
    /// engine uses (a row of 1,024 values, the real 2,688 rows) and smaller ones.
    #[test]
    fn pack_read_back_dequantize() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for bits in [2u32, 3] {
            for (rows, cols) in [(1, 32), (5, 64), (7, 128), (64, 1024), (2688, 1024)] {
                let (codes, scales, codebook) = random(&mut rng, bits, rows, cols);
                let p = pack_codebook_rows(&codes, &scales, &codebook, bits, rows, cols).unwrap();
                assert!(p.is_valid());
                assert_eq!(p.tiers(), [cb_rows_tag(bits).unwrap() << 24, 0, 0]);
                assert_eq!(unpack_codebook_codes(&p), codes, "{bits} bits {rows} x {cols}");
                let mut w = vec![0f32; rows * cols];
                let mut h = vec![0u16; rows * cols];
                assert!(decode_codebook_rows(&p, &mut w) && unpack_codebook_rows(&p, &mut h));
                for n in (0..rows).step_by(1 + rows / 17) {
                    for j in 0..cols {
                        let want = bf16_f32(scales[n * cols / 32 + j / 32]) * bf16_f32(codebook[(n << bits) + codes[n * cols + j] as usize]);
                        assert_eq!(w[n * cols + j].to_bits(), want.to_bits(), "{bits} bits ({n}, {j})");
                        assert_eq!(p.value(n, j).to_bits(), want.to_bits());
                        assert_eq!(h[n * cols + j], bits_of(want));
                    }
                }
            }
        }
    }

    /// The sizes the brief gives: 328 and 464 bytes a row at 1,024 values, 3.625 and 2.5625 bits a weight, 1,763,328 and 2,494,464 bytes an expert of two matrices of 2,688 rows; the tags are the layout's own.
    #[test]
    fn the_sizes_of_a_row_and_an_expert() {
        for (bits, row, per_weight, expert) in [(2u32, 328usize, 2.5625f64, 1_763_328usize), (3, 464, 3.625, 2_494_464)] {
            let (c, s, b) = cb_strides(bits, 1024);
            assert_eq!(c + s + b, row);
            assert_eq!(row as f64 * 8.0 / 1024.0, per_weight);
            assert_eq!(2 * 2688 * row, expert);
            assert_eq!((c, s, b), (bits as usize * 128, 64, 2 << bits));
        }
        assert_eq!((cb_rows_bits(CB3_ROW_TAG), cb_rows_bits(CB2_ROW_TAG), cb_rows_bits(INT4_ROW_TAG), cb_rows_bits(0)), (Some(3), Some(2), None, None));
        assert_eq!((cb_rows_tag(3), cb_rows_tag(2), cb_rows_tag(4)), (Some(11), Some(12), None));
        assert_eq!((rows_bits(CB3_ROW_TAG), rows_bits(CB2_ROW_TAG)), (None, None), "the codebook tags are not the block-code row layouts'");
    }

    /// What is not a codebook row matrix is refused: bits other than 2 and 3, columns that are not runs of 32, no rows, a length that is not the shape's, a code past the codebook, a scale or an entry that is not finite.
    #[test]
    fn what_is_refused() {
        let mut rng = Rng(5);
        let (codes, scales, book) = random(&mut rng, 3, 4, 64);
        assert!(pack_codebook_rows(&codes, &scales, &book, 3, 4, 64).is_some());
        assert!(pack_codebook_rows(&codes, &scales, &book, 4, 4, 64).is_none() && pack_codebook_rows(&codes, &scales, &book, 1, 4, 64).is_none());
        assert!(pack_codebook_rows(&codes[..4 * 48], &scales[..4 * 3 / 2], &book, 3, 4, 48).is_none(), "48 columns");
        assert!(pack_codebook_rows(&[], &[], &[], 3, 0, 64).is_none());
        assert!(pack_codebook_rows(&codes[1..], &scales, &book, 3, 4, 64).is_none() && pack_codebook_rows(&codes, &scales[1..], &book, 3, 4, 64).is_none() && pack_codebook_rows(&codes, &scales, &book[1..], 3, 4, 64).is_none());
        let mut past = codes.clone();
        past[7] = 8;
        assert!(pack_codebook_rows(&past, &scales, &book, 3, 4, 64).is_none());
        let (mut codes2, scales2, book2) = random(&mut rng, 2, 4, 64);
        assert!(pack_codebook_rows(&codes2, &scales2, &book2, 2, 4, 64).is_some());
        codes2[7] = 4;
        assert!(pack_codebook_rows(&codes2, &scales2, &book2, 2, 4, 64).is_none(), "a code of 4 is past a 2-bit codebook");
        let (mut s2, mut b2) = (scales.clone(), book.clone());
        s2[3] = 0x7F80;
        assert!(pack_codebook_rows(&codes, &s2, &book, 3, 4, 64).is_none());
        b2[9] = 0xFFC0;
        assert!(pack_codebook_rows(&codes, &scales, &b2, 3, 4, 64).is_none());
        let mut p = pack_codebook_rows(&codes, &scales, &book, 3, 4, 64).unwrap();
        p.codes.pop();
        assert!(!p.is_valid() && !unpack_codebook_rows(&p, &mut [0; 256]) && !decode_codebook_rows(&p, &mut [0.0; 256]));
        let p = pack_codebook_rows(&codes, &scales, &book, 3, 4, 64).unwrap();
        assert!(!unpack_codebook_rows(&p, &mut [0; 255]));
    }
}
