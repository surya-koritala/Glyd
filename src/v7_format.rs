//! Format v7 constants, code mappings and the block sub-header.
//!
//! Lengths: values below 16 are their own code; larger values use code
//! 12 + floor(log2(v)) with floor(log2(v)) extra bits holding v - 2^k.
//! Offsets: codes 0..=2 are the three most recent distinct offsets; a real
//! offset o uses code 3 + floor(log2(o)) with that many extra bits.

pub const MAX_OFFSET_BITS: u32 = 21;
pub const MAX_WINDOW: u32 = 1 << MAX_OFFSET_BITS;
pub const MIN_MATCH: u32 = 3;
pub const LL_SYMBOLS: usize = 32; // codes 0..=30 for values up to 2^18
pub const ML_SYMBOLS: usize = 32;
pub const OFF_SYMBOLS: usize = 24; // 3 reps + log2 buckets 0..=20

pub const S_LIT: usize = 0;
pub const S_LL: usize = 1;
pub const S_ML: usize = 2;
pub const S_OFF: usize = 3;
pub const S_EXTRA: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Ll,
    Ml,
    Off,
}

#[inline(always)]
fn log2(v: u32) -> u32 {
    31 - v.leading_zeros()
}

#[inline(always)]
fn len_code(v: u32) -> (u8, u8, u32) {
    if v < 16 {
        (v as u8, 0, 0)
    } else {
        let k = log2(v);
        ((12 + k) as u8, k as u8, v - (1 << k))
    }
}

#[inline(always)]
fn len_value(code: u8, extra: u32) -> u32 {
    if code < 16 {
        code as u32
    } else {
        let k = code as u32 - 12;
        (1 << k) + extra
    }
}

pub fn ll_code(v: u32) -> (u8, u8, u32) {
    len_code(v)
}
pub fn ll_value(code: u8, extra: u32) -> u32 {
    len_value(code, extra)
}
pub fn ml_code(v: u32) -> (u8, u8, u32) {
    debug_assert!(v >= MIN_MATCH);
    len_code(v - MIN_MATCH)
}
pub fn ml_value(code: u8, extra: u32) -> u32 {
    len_value(code, extra) + MIN_MATCH
}
pub fn off_code(offset: u32) -> (u8, u8, u32) {
    debug_assert!(offset >= 1 && offset < MAX_WINDOW);
    let k = log2(offset);
    ((3 + k) as u8, k as u8, offset - (1 << k))
}
pub fn off_value(code: u8, extra: u32) -> u32 {
    let k = code as u32 - 3;
    (1 << k) + extra
}

/// Extra bits carried by a code; the decoder reads this many after the
/// symbol. For length codes below 16 and rep codes it is zero.
#[inline(always)]
pub fn extra_bits_of_code(kind: Kind, code: u8) -> u8 {
    match kind {
        Kind::Ll | Kind::Ml => {
            if code < 16 {
                0
            } else {
                code - 12
            }
        }
        Kind::Off => {
            if code < 3 {
                0
            } else {
                code - 3
            }
        }
    }
}

/// Repeat-offset state, identical on both sides.
#[derive(Clone, Copy)]
pub struct Reps {
    r: [u32; 3],
}

impl Reps {
    pub fn new() -> Self {
        Reps { r: [1, 4, 8] }
    }

    /// Encoder: rep code if `offset` is a repeat, else its real code.
    pub fn code_for(&mut self, offset: u32) -> (u8, u8, u32) {
        if offset == self.r[0] {
            return (0, 0, 0);
        }
        if offset == self.r[1] {
            self.r.swap(0, 1);
            return (1, 0, 0);
        }
        if offset == self.r[2] {
            self.r = [self.r[2], self.r[0], self.r[1]];
            return (2, 0, 0);
        }
        self.r = [offset, self.r[0], self.r[1]];
        off_code(offset)
    }

    /// Decoder: the offset for a code and its extra bits.
    #[inline(always)]
    pub fn resolve(&mut self, code: u8, extra: u32) -> u32 {
        match code {
            0 => self.r[0],
            1 => {
                self.r.swap(0, 1);
                self.r[0]
            }
            2 => {
                self.r = [self.r[2], self.r[0], self.r[1]];
                self.r[0]
            }
            _ => {
                let o = off_value(code, extra);
                self.r = [o, self.r[0], self.r[1]];
                o
            }
        }
    }
}

/// Sits at the start of a v7 payload. `coded` bit i: stream i is entropy
/// coded (stream 4 is always raw bits and ignores the flag). `reuse` bit
/// 0: literal table reused from the previous block; bit 1: the three
/// sequence tables reused. `sizes`: bytes of each stream section
/// (tables, sub-stream size table and data included).
#[derive(Clone, Copy, Debug)]
pub struct SubHeader {
    pub coded: u8,
    pub reuse: u8,
    pub dict_id: u32,
    pub sizes: [u32; 5],
}

impl SubHeader {
    pub const BYTES: usize = 1 + 1 + 4 + 4 * 5;

    pub fn write(&self, out: &mut Vec<u8>) {
        out.push(self.coded);
        out.push(self.reuse);
        out.extend_from_slice(&self.dict_id.to_le_bytes());
        for s in self.sizes {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }

    pub fn parse(src: &[u8]) -> Option<SubHeader> {
        if src.len() < Self::BYTES {
            return None;
        }
        let u = |i: usize| u32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
        Some(SubHeader {
            coded: src[0],
            reuse: src[1],
            dict_id: u(2),
            sizes: [u(6), u(10), u(14), u(18), u(22)],
        })
    }
}
