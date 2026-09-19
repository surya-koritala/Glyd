//! Format v7 constants, code mappings and the block sub-header.
//!
//! Lengths (v8): direct codes for the common values, then buckets of 1-5
//! extra bits, then one code per power of two (`LL_BITS`/`LL_BASE`,
//! `ML_BITS`/`ML_BASE`); v7 blocks used direct codes below 16 and log2
//! buckets above.
//! Offsets: codes 0..=2 are the three most recent distinct offsets; a real
//! offset o uses code 3 + floor(log2(o)) with that many extra bits.

/// v8: offsets below 8 MB. v7 blocks (`OFF_SYMBOLS_V7`) stay below 2 MB.
pub const MAX_OFFSET_BITS: u32 = 23;
pub const MAX_WINDOW: u32 = 1 << MAX_OFFSET_BITS;
pub const MIN_MATCH: u32 = 3;
pub const LL_SYMBOLS: usize = 38;
pub const ML_SYMBOLS: usize = 54;
pub const OFF_SYMBOLS: usize = 26; // 3 reps + log2 buckets 0..=22
pub const OFF_SYMBOLS_V7: usize = 24;

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
const fn log2(v: u32) -> u32 {
    31 - v.leading_zeros()
}

// Length codes, v8: direct codes for the common values, then buckets
// that grow slowly (two or four codes per doubling) before the log2
// buckets take over, so that a match of 20 or a literal run of 30 costs
// its code alone. The tables below give each code's extra bits and base.

/// Literal-length codes: 0-15 direct; 16-24 in steps of 1, 1, 1, 1, 2,
/// 2, 3, 3, 4 bits; 25.. one code per power of two from 64 (6 bits) to
/// 2^18 (18 bits, the run of a whole block).
pub const LL_BITS: [u8; LL_SYMBOLS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
pub const LL_BASE: [u32; LL_SYMBOLS] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
/// Match-length codes: 0-31 direct for lengths 3-34; 32-42 in steps of
/// 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5 bits; 43.. one code per power of two
/// of (length - 3) from 128 (7 bits) to 2^17 (17 bits).
pub const ML_BITS: [u8; ML_SYMBOLS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
pub const ML_BASE: [u32; ML_SYMBOLS] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539, 131075];

const _: () = {
    // Each table is contiguous: a code's range ends where the next begins.
    let mut c = 0;
    while c + 1 < LL_SYMBOLS {
        assert!(LL_BASE[c] + (1 << LL_BITS[c]) == LL_BASE[c + 1]);
        c += 1;
    }
    let mut c = 0;
    while c + 1 < ML_SYMBOLS {
        assert!(ML_BASE[c] + (1 << ML_BITS[c]) == ML_BASE[c + 1]);
        c += 1;
    }
    assert!(ML_BASE[ML_SYMBOLS - 1] + (1 << ML_BITS[ML_SYMBOLS - 1]) > crate::format::MAX_BLOCK_SIZE as u32);
    assert!(LL_BASE[LL_SYMBOLS - 1] <= crate::format::MAX_BLOCK_SIZE as u32);
};

/// The code of a literal run of `v`: (code, extra bits, extra).
#[inline(always)]
pub const fn ll_code(v: u32) -> (u8, u8, u32) {
    if v < 64 {
        let c = LL_SMALL[v as usize];
        (c, LL_BITS[c as usize], v - LL_BASE[c as usize])
    } else {
        let k = log2(v);
        ((19 + k) as u8, k as u8, v - (1 << k))
    }
}

#[inline(always)]
pub const fn ll_value(code: u8, extra: u32) -> u32 {
    LL_BASE[code as usize] + extra
}

/// The code of a match of `v` bytes (at least MIN_MATCH).
#[inline(always)]
pub const fn ml_code(v: u32) -> (u8, u8, u32) {
    debug_assert!(v >= MIN_MATCH);
    if v < 131 {
        let c = ML_SMALL[v as usize];
        (c, ML_BITS[c as usize], v - ML_BASE[c as usize])
    } else {
        let k = log2(v - 3);
        ((36 + k) as u8, k as u8, v - 3 - (1 << k))
    }
}

#[inline(always)]
pub const fn ml_value(code: u8, extra: u32) -> u32 {
    ML_BASE[code as usize] + extra
}

/// Code of each small value, from the tables.
const LL_SMALL: [u8; 64] = small_codes::<64, LL_SYMBOLS>(&LL_BASE);
const ML_SMALL: [u8; 131] = small_codes::<131, ML_SYMBOLS>(&ML_BASE);

const fn small_codes<const N: usize, const S: usize>(base: &[u32; S]) -> [u8; N] {
    let mut t = [0u8; N];
    let mut c = 0;
    let mut v = base[0] as usize;
    while v < N {
        while c + 1 < S && base[c + 1] as usize <= v {
            c += 1;
        }
        t[v] = c as u8;
        v += 1;
    }
    t
}

// The v7 length codes, for decoding v7 blocks: values below 16 are their
// own code, larger ones code 12 + floor(log2(v)) with that many extra bits.
pub const LL_SYMBOLS_V7: usize = 32;
pub const ML_SYMBOLS_V7: usize = 32;

#[inline(always)]
pub const fn len_value_v7(code: u8, extra: u32) -> u32 {
    if code < 16 {
        code as u32
    } else {
        (1 << (code as u32 - 12)) + extra
    }
}

pub fn off_code(offset: u32) -> (u8, u8, u32) {
    debug_assert!(offset >= 1 && offset < MAX_WINDOW);
    let k = log2(offset);
    ((3 + k) as u8, k as u8, offset - (1 << k))
}
pub const fn off_value(code: u8, extra: u32) -> u32 {
    let k = code as u32 - 3;
    (1 << k) + extra
}

/// Extra bits carried by a code; the decoder reads this many after the
/// symbol. For rep codes it is zero. Codes past a field's symbol count
/// read as zero bits.
#[inline(always)]
pub const fn extra_bits_of_code(kind: Kind, code: u8) -> u8 {
    match kind {
        Kind::Ll => {
            if (code as usize) < LL_SYMBOLS {
                LL_BITS[code as usize]
            } else {
                0
            }
        }
        Kind::Ml => {
            if (code as usize) < ML_SYMBOLS {
                ML_BITS[code as usize]
            } else {
                0
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

/// `extra_bits_of_code` for v7 blocks' length codes.
#[inline(always)]
pub const fn extra_bits_of_code_v7(kind: Kind, code: u8) -> u8 {
    match kind {
        Kind::Ll | Kind::Ml => {
            if code < 16 {
                0
            } else {
                code - 12
            }
        }
        Kind::Off => extra_bits_of_code(Kind::Off, code),
    }
}

/// Decoder walk tables, one per field: entry `code` packs
/// `value_base << 32 | mask << 8 | extra_bits`, with `mask` the low
/// `extra_bits` bits set (at most 20 of them, so it fits below bit 32),
/// so a field decodes as one table load, an AND with the mask, an add of
/// the base and a shift by `extra_bits` -- each a single instruction on
/// AArch64 with this layout (the mask and the base come in as shifted
/// operands; building the mask from `extra_bits` would cost two more).
/// For a code without extra bits, extra is 0 and `base` is the value
/// itself; rep codes get base 0 and are resolved by `Reps`. Built from
/// the code functions above. 256 entries so a `u8` code indexes without
/// a check; entries past a field's symbol count are zero (never
/// produced: the code streams are validated to their symbol counts).
pub const LL_WALK: [u64; 256] = walk_table(Kind::Ll, LL_SYMBOLS, false);
pub const ML_WALK: [u64; 256] = walk_table(Kind::Ml, ML_SYMBOLS, false);
pub const OFF_WALK: [u64; 256] = walk_table(Kind::Off, OFF_SYMBOLS, false);
/// The same for v7 blocks' length codes (offsets code alike).
pub const LL_WALK_V7: [u64; 256] = walk_table(Kind::Ll, LL_SYMBOLS_V7, true);
pub const ML_WALK_V7: [u64; 256] = walk_table(Kind::Ml, ML_SYMBOLS_V7, true);

/// The walk entries split into width bytes and base dwords, all six
/// tables behind one base address for the x86-64 walk (a byte and a dword
/// load per field, folded into its arithmetic, no table pointers live).
#[repr(C)]
pub struct WalkSplit {
    pub nb: [[u8; 256]; 3],
    pub base: [[u32; 256]; 3],
}

pub static WALK_SPLIT: WalkSplit = WalkSplit {
    nb: [nb_table(&LL_WALK), nb_table(&ML_WALK), nb_table(&OFF_WALK)],
    base: [base_table(&LL_WALK), base_table(&ML_WALK), base_table(&OFF_WALK)],
};
pub static WALK_SPLIT_V7: WalkSplit = WalkSplit {
    nb: [nb_table(&LL_WALK_V7), nb_table(&ML_WALK_V7), nb_table(&OFF_WALK)],
    base: [base_table(&LL_WALK_V7), base_table(&ML_WALK_V7), base_table(&OFF_WALK)],
};

const fn nb_table(walk: &[u64; 256]) -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 256 {
        t[c] = walk[c] as u8;
        c += 1;
    }
    t
}

const fn base_table(walk: &[u64; 256]) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut c = 0;
    while c < 256 {
        t[c] = (walk[c] >> 32) as u32;
        c += 1;
    }
    t
}

const fn walk_table(kind: Kind, n_symbols: usize, v7: bool) -> [u64; 256] {
    let mut t = [0u64; 256];
    let mut c = 0;
    while c < n_symbols {
        let base = match kind {
            Kind::Ll => {
                if v7 {
                    len_value_v7(c as u8, 0)
                } else {
                    ll_value(c as u8, 0)
                }
            }
            Kind::Ml => {
                if v7 {
                    len_value_v7(c as u8, 0) + MIN_MATCH
                } else {
                    ml_value(c as u8, 0)
                }
            }
            Kind::Off => {
                if c < 3 {
                    0
                } else {
                    off_value(c as u8, 0)
                }
            }
        };
        let nb = if v7 { extra_bits_of_code_v7(kind, c as u8) } else { extra_bits_of_code(kind, c as u8) } as u64;
        t[c] = (base as u64) << 32 | ((1u64 << nb) - 1) << 8 | nb;
        c += 1;
    }
    t
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
        self.update(code, if code < 3 { 0 } else { off_value(code, extra) })
    }

    /// Decoder: the offset for a code whose extra bits already decoded to
    /// `value` (`off_value`; ignored for a rep code), updating the reps.
    /// Selects, not a match: in the decoder's walk the branchy form
    /// measured 0.3 ns/sequence slower on Silesia even with rep codes at
    /// only 4% of offsets (the default parse; a better one raises that).
    /// The three cases code 1 (swap r0 r1), code 2 (rotate r2 to the
    /// front) and a real offset (push) are all "the chosen offset moves
    /// to the front, the entries in front of its old slot shift back one".
    #[inline(always)]
    pub fn update(&mut self, code: u8, value: u32) -> u32 {
        use std::hint::select_unpredictable as sel;
        let [r0, r1, r2] = self.r;
        let o = sel(code < 2, sel(code == 0, r0, r1), sel(code == 2, r2, value));
        self.r = [o, sel(code == 0, r1, r0), sel(code < 2, r2, r1)];
        o
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

    /// The compact form (v9 blocks): sizes as u16.
    pub const COMPACT_BYTES: usize = 1 + 1 + 4 + 2 * 5;

    pub fn write_compact(&self, out: &mut Vec<u8>) {
        out.push(self.coded);
        out.push(self.reuse);
        out.extend_from_slice(&self.dict_id.to_le_bytes());
        for s in self.sizes {
            debug_assert!(s < 65536);
            out.extend_from_slice(&(s as u16).to_le_bytes());
        }
    }

    pub fn parse_compact(src: &[u8]) -> Option<SubHeader> {
        if src.len() < Self::COMPACT_BYTES {
            return None;
        }
        let u = |i: usize| u16::from_le_bytes([src[i], src[i + 1]]) as u32;
        Some(SubHeader {
            coded: src[0],
            reuse: src[1],
            dict_id: u32::from_le_bytes([src[2], src[3], src[4], src[5]]),
            sizes: [u(6), u(8), u(10), u(12), u(14)],
        })
    }
}
