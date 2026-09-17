pub const MAGIC: u32 = 0x53494D44; // "SIMD"
pub const CURRENT_VERSION: u16 = 3;
pub const MAX_BLOCK_SIZE: usize = 65536; // 64 KB
pub const PADDING: usize = 64; // Safe SIMD read/write margin

pub const MIN_MATCH_LEN: usize = 4;

/// Escaped lengths travel in a u16 stream, so both cap at u16::MAX.
pub const MAX_LIT_LEN: usize = 65535;
pub const MAX_MATCH_LEN: usize = 65535;

/// v3 token layout, one byte:
///   bits 0..2  literal code: 0..6 is the literal length, 7 escapes to `extras`
///   bits 3..7  match code:   0 means no match, 1..30 means length 4..33,
///                            31 escapes to `extras`
///
/// Sized from the measured Silesia distribution: literal runs are <= 6 bytes
/// for 94.20% of tokens and match lengths land in 4..33 for 96.56%, so the
/// common token costs one byte instead of the two that v2 always spent.
pub const LIT_DIRECT_MAX: usize = 6;
pub const LIT_CODE_ESCAPE: usize = 7;
pub const MATCH_CODE_ESCAPE: usize = 31;
/// match code 1 encodes length 4, so length = code + MATCH_CODE_BIAS.
pub const MATCH_CODE_BIAS: usize = 3;
pub const MATCH_DIRECT_MAX: usize = 33;

pub const FLAG_COMPRESSED: u16 = 0;
pub const FLAG_RAW_UNCOMPRESSED: u16 = 1;
pub const FLAG_CHAIN_RESET: u16 = 2;

pub const PARALLEL_CHUNK_SIZE: usize = 256 * 1024; // 256 KB parallel chunk unit

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Token(pub u8);

impl Token {
    #[inline(always)]
    pub fn from_codes(lit_code: usize, match_code: usize) -> Self {
        debug_assert!(lit_code <= LIT_CODE_ESCAPE);
        debug_assert!(match_code <= MATCH_CODE_ESCAPE);
        Self((lit_code as u8) | ((match_code as u8) << 3))
    }

    #[inline(always)]
    pub fn lit_code(self) -> usize {
        (self.0 & 0x07) as usize
    }

    #[inline(always)]
    pub fn match_code(self) -> usize {
        (self.0 >> 3) as usize
    }
}

/// Encode a literal run length into a token code, plus an optional `extras`
/// entry when it does not fit the 3-bit field.
#[inline(always)]
pub fn encode_lit(lit_len: usize) -> (usize, Option<u16>) {
    if lit_len <= LIT_DIRECT_MAX {
        (lit_len, None)
    } else {
        debug_assert!(lit_len <= MAX_LIT_LEN);
        (LIT_CODE_ESCAPE, Some(lit_len as u16))
    }
}

/// Encode a match length into a token code, plus an optional `extras` entry.
#[inline(always)]
pub fn encode_match(match_len: usize) -> (usize, Option<u16>) {
    if match_len == 0 {
        (0, None)
    } else if match_len >= MIN_MATCH_LEN && match_len <= MATCH_DIRECT_MAX {
        (match_len - MATCH_CODE_BIAS, None)
    } else {
        debug_assert!(match_len <= MAX_MATCH_LEN);
        (MATCH_CODE_ESCAPE, Some(match_len as u16))
    }
}

/// Decode table indexed by the whole token byte, so the hot loop needs one
/// load instead of two shifts and four comparisons.
///   bits  0..7   literal length (valid unless the literal-escape bit is set)
///   bits  8..15  match length   (valid unless the match-escape bit is set)
///   bit  16      literal length escapes to `extras`
///   bit  17      match length escapes to `extras`
/// `TOKEN_ESCAPE_MASK` is zero for the ~91% of tokens that need no extras.
pub const TOKEN_LIT_ESCAPE: u32 = 1 << 16;
pub const TOKEN_MATCH_ESCAPE: u32 = 1 << 17;
pub const TOKEN_ESCAPE_MASK: u32 = TOKEN_LIT_ESCAPE | TOKEN_MATCH_ESCAPE;

pub static TOKEN_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let lc = i & 0x07;
        let mc = i >> 3;
        let mut v: u32 = 0;
        if lc == LIT_CODE_ESCAPE {
            v |= TOKEN_LIT_ESCAPE;
        } else {
            v |= lc as u32;
        }
        if mc == MATCH_CODE_ESCAPE {
            v |= TOKEN_MATCH_ESCAPE;
        } else if mc != 0 {
            v |= ((mc + MATCH_CODE_BIAS) as u32) << 8;
        }
        t[i] = v;
        i += 1;
    }
    t
};

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct BlockHeader {
    pub magic: u32,
    pub version: u16,
    pub flags: u16,
    pub checksum: u32,
    pub uncompressed_len: u32,
    pub token_count: u32,
    pub offset_count: u32,
    pub extras_count: u32,
    pub literal_len: u32,
}

pub const HEADER_SIZE: usize = std::mem::size_of::<BlockHeader>();

/// Payload size for a compressed block: tokens are 1 byte, offsets and extras
/// are 2 bytes each, literals are raw.
#[inline(always)]
pub fn payload_len(token_count: usize, offset_count: usize, extras_count: usize, literal_len: usize) -> usize {
    token_count + offset_count * 2 + extras_count * 2 + literal_len
}

/// Fast hardware-friendly Adler-like 32-bit checksum for integrity verification.
#[inline(always)]
pub fn compute_checksum(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { crate::x86_checksum::checksum_avx2(data) };
        }
    }
    compute_checksum_scalar(data)
}

#[inline(always)]
pub fn compute_checksum_scalar(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = a.wrapping_add(byte as u32);
        b = b.wrapping_add(a);
    }
    (b << 16) | (a & 0xFFFF)
}
