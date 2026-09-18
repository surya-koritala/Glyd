pub const MAGIC: u32 = 0x53494D44; // "SIMD"
pub const CURRENT_VERSION: u16 = 6;
/// Match window. An offset is 17 bits: 16 in the offset stream plus one in
/// the token, so the format addresses 128 KB. Measured (token_stats): with
/// a 256 KB finder window every offset already fit 18 bits, so the two bits
/// v5 spent on a width class bought nothing; spending one on the offset
/// instead makes the offset stream constant-stride (no per-token width
/// chain in the decoder) and frees a bit for the literal field.
pub const WINDOW_SIZE: usize = 1 << 17;
pub const MAX_OFFSET: usize = WINDOW_SIZE - 1;
/// Output covered by one block header. Larger blocks mean fewer headers and,
/// more importantly, matches that are not cut short at a block boundary.
pub const MAX_BLOCK_SIZE: usize = 256 * 1024;
pub const PADDING: usize = 64; // Safe SIMD read/write margin

/// LZAV's minimum reference: 6-byte matches cost 2 or 3 bytes to encode, so
/// anything shorter is not worth a token. Measured on Silesia: 23% fewer
/// tokens than the 4-byte minimum for 10.7% more literal bytes, net positive.
pub const MIN_MATCH_LEN: usize = 6;
/// Minimum match of the dense retry parse (FLAG_DENSE blocks). Data such as
/// 12-bit images has its redundancy in 4- and 5-byte matches.
pub const MIN_MATCH_LEN_DENSE: usize = 5;

/// v6 token layout, one byte:
///   bits 0..2  literal code: 0..6 is the literal length, 7 escapes to `extras`
///   bits 3..6  match code:   0 means no match, 1..14 means length 6..19,
///                            15 escapes to `extras`
///   bit  7     offset bit 16; bits 0..15 are the two bytes in the offset stream
///
/// Sized from our own token stream (token_stats, 256 KB window): literal runs
/// are <= 6 for 89.4% of tokens (<= 2 was only 77.7%), match lengths <= 19
/// for 87.9%. Escaped tokens fall from 31.7% to 21.5%, and each escape costs
/// the decoder ~5 ns.
pub const LIT_DIRECT_MAX: usize = 6;
pub const LIT_CODE_ESCAPE: usize = 7;
pub const MATCH_CODE_ESCAPE: usize = 15;
/// match code 1 encodes the minimum length, so length = code + bias where
/// bias = minimum - 1: 5 for ordinary blocks, 3 for FLAG_DENSE blocks.
pub const MATCH_CODE_BIAS: usize = MIN_MATCH_LEN - 1;
pub const MATCH_CODE_BIAS_DENSE: usize = MIN_MATCH_LEN_DENSE - 1;
pub const MATCH_DIRECT_MAX: usize = MATCH_CODE_BIAS + 14;

/// Escaped lengths travel in a byte stream: one byte holds `value - base`
/// when that is below 255; the byte 255 means a little-endian u16 follows
/// holding `value - base - 255`.
pub const ESCAPE_BASE_LIT: usize = LIT_DIRECT_MAX + 1;
pub const ESCAPE_BASE_MATCH: usize = MATCH_DIRECT_MAX + 1;
pub const ESCAPE_BASE_MATCH_DENSE: usize = MATCH_CODE_BIAS_DENSE + 15;
pub const ESCAPE_CONT: u8 = 255;
pub const MAX_LIT_LEN: usize = ESCAPE_BASE_LIT + 255 + 65535;
pub const MAX_MATCH_LEN: usize = ESCAPE_BASE_MATCH + 255 + 65535;

pub const FLAG_COMPRESSED: u16 = 0;
pub const FLAG_RAW_UNCOMPRESSED: u16 = 1;
pub const FLAG_CHAIN_RESET: u16 = 2;
/// Token section is Huffman coded (see huffman.rs). Reserved; not yet emitted.
pub const FLAG_HUFF_TOKENS: u16 = 4;
/// Block was parsed by the dense retry: match lengths are biased by
/// MATCH_CODE_BIAS_DENSE and offsets may be as small as 1 with overlap.
pub const FLAG_DENSE: u16 = 8;

pub const PARALLEL_CHUNK_SIZE: usize = 256 * 1024; // 256 KB parallel chunk unit

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Token(pub u8);

impl Token {
    #[inline(always)]
    pub fn from_codes(lit_code: usize, match_code: usize, off_hi: usize) -> Self {
        debug_assert!(lit_code <= LIT_CODE_ESCAPE);
        debug_assert!(match_code <= MATCH_CODE_ESCAPE);
        debug_assert!(off_hi <= 1);
        Self((lit_code as u8) | ((match_code as u8) << 3) | ((off_hi as u8) << 7))
    }

    #[inline(always)]
    pub fn lit_code(self) -> usize {
        (self.0 & 0x07) as usize
    }

    #[inline(always)]
    pub fn match_code(self) -> usize {
        ((self.0 >> 3) & 0x0F) as usize
    }

    /// Bit 16 of the match offset; bits 0..15 are the two bytes in the offset
    /// stream. Meaningful only when `match_code` is non-zero.
    #[inline(always)]
    pub fn off_hi(self) -> usize {
        (self.0 >> 7) as usize
    }
}

/// Append an escaped length to the extras stream.
#[inline(always)]
pub fn push_escape(extras: &mut Vec<u8>, value: usize, base: usize) {
    let v = value - base;
    if v < 255 {
        extras.push(v as u8);
    } else {
        let rest = v - 255;
        debug_assert!(rest <= 65535);
        extras.push(ESCAPE_CONT);
        extras.extend_from_slice(&(rest as u16).to_le_bytes());
    }
}

/// Literal run length to token code, writing an escape when needed.
#[inline(always)]
pub fn encode_lit(lit_len: usize, extras: &mut Vec<u8>) -> usize {
    if lit_len <= LIT_DIRECT_MAX {
        lit_len
    } else {
        debug_assert!(lit_len <= MAX_LIT_LEN);
        push_escape(extras, lit_len, ESCAPE_BASE_LIT);
        LIT_CODE_ESCAPE
    }
}

/// Match length to token code, writing an escape when needed. Zero is "no
/// match". `min_match` is the parse's minimum, which sets the length bias.
#[inline(always)]
pub fn encode_match(match_len: usize, extras: &mut Vec<u8>, min_match: usize) -> usize {
    let bias = min_match - 1;
    if match_len == 0 {
        0
    } else if match_len <= bias + 14 {
        debug_assert!(match_len >= min_match);
        match_len - bias
    } else {
        debug_assert!(match_len <= MAX_MATCH_LEN);
        push_escape(extras, match_len, bias + 15);
        MATCH_CODE_ESCAPE
    }
}

/// Append the low 16 bits of an offset; bit 16 goes in the token.
#[inline(always)]
pub fn push_offset(offsets: &mut Vec<u8>, offset: usize) {
    debug_assert!(offset > 0 && offset <= MAX_OFFSET);
    offsets.extend_from_slice(&(offset as u16).to_le_bytes());
}

/// Bytes of offset stream per match.
pub const OFFSET_BYTES: usize = 2;

/// Decode table indexed by the whole token byte, so the hot loop needs one
/// load instead of shifts and comparisons.
///   bits  0..7   literal length (valid unless the literal-escape bit is set)
///   bits  8..15  match length   (valid unless the match-escape bit is set)
///   bit  16      literal length escapes to `extras`
///   bit  17      match length escapes to `extras`
///   bit  20      offset bit 16
/// `TOKEN_ESCAPE_MASK` is zero for the tokens that need no extras.
pub const TOKEN_LIT_ESCAPE: u32 = 1 << 16;
pub const TOKEN_MATCH_ESCAPE: u32 = 1 << 17;
pub const TOKEN_ESCAPE_MASK: u32 = TOKEN_LIT_ESCAPE | TOKEN_MATCH_ESCAPE;
pub const TOKEN_OFF_SHIFT: u32 = 20;

const fn token_table(bias: usize) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let lc = i & 0x07;
        let mc = (i >> 3) & 0x0F;
        let ow = i >> 7;
        let mut v: u32 = 0;
        if lc == LIT_CODE_ESCAPE {
            v |= TOKEN_LIT_ESCAPE;
        } else {
            v |= lc as u32;
        }
        if mc == MATCH_CODE_ESCAPE {
            v |= TOKEN_MATCH_ESCAPE;
            v |= (ow as u32) << TOKEN_OFF_SHIFT;
        } else if mc != 0 {
            v |= ((mc + bias) as u32) << 8;
            v |= (ow as u32) << TOKEN_OFF_SHIFT;
        }
        t[i] = v;
        i += 1;
    }
    t
}

pub static TOKEN_TABLE: [u32; 256] = token_table(MATCH_CODE_BIAS);
pub static TOKEN_TABLE_DENSE: [u32; 256] = token_table(MATCH_CODE_BIAS_DENSE);

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct BlockHeader {
    pub magic: u32,
    pub version: u16,
    pub flags: u16,
    pub checksum: u32,
    pub uncompressed_len: u32,
    /// Number of tokens, which is the number of (literal run, match) pairs.
    pub token_count: u32,
    /// Bytes the token section occupies on disk. Equals `token_count` unless
    /// the tokens are entropy coded.
    pub token_bytes: u32,
    /// Bytes of the offset section: OFFSET_BYTES per match.
    pub offset_bytes: u32,
    pub extras_bytes: u32,
    pub literal_len: u32,
}

pub const HEADER_SIZE: usize = std::mem::size_of::<BlockHeader>();

impl BlockHeader {
    /// Bytes following the header for this block.
    #[inline(always)]
    pub fn payload_len(&self) -> usize {
        if (self.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            self.uncompressed_len as usize
        } else {
            self.token_bytes as usize
                + self.offset_bytes as usize
                + self.extras_bytes as usize
                + self.literal_len as usize
        }
    }

    /// Reject any block whose section sizes could not have come from the
    /// compressor, before anything is dereferenced.
    #[inline(always)]
    pub fn is_plausible(&self) -> bool {
        let u = self.uncompressed_len as usize;
        u <= MAX_BLOCK_SIZE
            && self.token_count as usize <= u
            && self.token_bytes as usize <= u
            && self.offset_bytes as usize <= 2 * u
            && self.extras_bytes as usize <= 3 * u
            && self.literal_len as usize <= u
    }
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
