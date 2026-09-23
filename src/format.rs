pub const MAGIC: u32 = 0x53494D44; // "SIMD"
pub const CURRENT_VERSION: u16 = 6;
/// Entropy-coded blocks (see v7_format.rs). Same BlockHeader; for this
/// version `token_bytes` is the whole payload length, `token_count` the
/// sequence count and `literal_len` the literal byte count; the other
/// section fields are zero.
pub const VERSION_V7: u16 = 7;
/// v7 with an 8 MB window (26 offset codes), a compact section layout
/// (24-bit sub-stream sizes, one padding per section) and packed tANS
/// counts. Written by every level of format v7's kind from v0.3.0 on;
/// v7 blocks are still decoded.
pub const VERSION_V8: u16 = 8;

/// v8 coding in compact framing for small blocks: a one-byte marker
/// instead of the magic and version, the lengths as varints, a
/// sub-header of one flag byte, the dictionary id when there is one and
/// four varint section sizes, sub-stream sizes as varints, and no
/// padding on disk (the decoder pads its copy). Written for blocks of at
/// most `COMPACT_MAX` bytes.
pub const VERSION_V9: u16 = 9;
pub const COMPACT_MAX: usize = MAX_BLOCK_SIZE;
/// The first byte of a compact block ('G'; the magic's is 'D').
pub const COMPACT_MARKER: u8 = 0x47;
/// The shortest compact header: marker, flags, two one-byte lengths, checksum.
pub const COMPACT_HEADER_MIN: usize = 8;

/// A 7-bits-a-byte little-endian varint, the top bit marking more.
pub fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 128 {
        out.push((v & 127) as u8 | 128);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn varint_len(v: u32) -> usize {
    1 + (31 - (v | 1).leading_zeros() as usize) / 7
}

/// The varint at `src[*pos..]`; None if truncated or over 32 bits.
pub fn get_varint(src: &[u8], pos: &mut usize) -> Option<u32> {
    let (mut v, mut shift) = (0u32, 0u32);
    loop {
        let b = *src.get(*pos)?;
        *pos += 1;
        v |= ((b & 127) as u32) << shift;
        if b < 128 {
            return Some(v);
        }
        shift += 7;
        if shift > 28 {
            return None;
        }
    }
}

/// A block of the entropy-coded family (v7, v8 or v9).
#[inline(always)]
pub fn is_coded_version(version: u16) -> bool {
    version == VERSION_V7 || version == VERSION_V8 || version == VERSION_V9
}

/// Bytes the on-disk header of a coded block takes: the compact one's
/// depends on its lengths.
pub fn coded_header_len(compact: bool, uncompressed_len: usize, payload_len: usize, n_seq: usize, n_lit: usize) -> usize {
    if compact {
        2 + varint_len(uncompressed_len as u32) + varint_len(payload_len as u32) + varint_len(n_seq as u32) + varint_len(n_lit as u32) + 4
    } else {
        HEADER_SIZE
    }
}

impl BlockHeader {
    /// The compact on-disk form (v9): the marker, the flags as one byte,
    /// the uncompressed and payload lengths as varints, for a coded block
    /// the sequence and literal counts as varints, the checksum.
    pub fn write_compact(&self, out: &mut Vec<u8>) {
        debug_assert!(self.version == VERSION_V9 && self.flags < 256);
        out.push(COMPACT_MARKER);
        out.push(self.flags as u8);
        put_varint(out, self.uncompressed_len);
        put_varint(out, self.token_bytes);
        if self.flags & FLAG_RAW_UNCOMPRESSED == 0 {
            put_varint(out, self.token_count);
            put_varint(out, self.literal_len);
        }
        out.extend_from_slice(&self.checksum.to_le_bytes());
    }

    /// The header at the start of `src`, of either kind (compact by its
    /// marker, else the struct behind the magic), and its length; None
    /// if truncated or not a block.
    pub fn read(src: &[u8]) -> Option<(BlockHeader, usize)> {
        if src.first() == Some(&COMPACT_MARKER) {
            return Self::read_compact(src);
        }
        if src.len() < HEADER_SIZE || u32::from_le_bytes([src[0], src[1], src[2], src[3]]) != MAGIC {
            return None;
        }
        Some((unsafe { std::ptr::read_unaligned(src.as_ptr() as *const BlockHeader) }, HEADER_SIZE))
    }

    /// Bytes this header takes on disk.
    pub fn header_len(&self) -> usize {
        if self.version == VERSION_V9 {
            coded_header_len(true, self.uncompressed_len as usize, self.token_bytes as usize, self.token_count as usize, self.literal_len as usize) - if self.flags & FLAG_RAW_UNCOMPRESSED != 0 { varint_len(self.token_count) + varint_len(self.literal_len) } else { 0 }
        } else {
            HEADER_SIZE
        }
    }

    /// Read a compact header (`src` starts at its marker): the header and
    /// its length, or None if truncated.
    pub fn read_compact(src: &[u8]) -> Option<(BlockHeader, usize)> {
        if src.len() < COMPACT_HEADER_MIN || src[0] != COMPACT_MARKER {
            return None;
        }
        let flags = src[1] as u16;
        let mut pos = 2usize;
        let uncompressed_len = get_varint(src, &mut pos)?;
        let token_bytes = get_varint(src, &mut pos)?;
        let (token_count, literal_len) = if flags & FLAG_RAW_UNCOMPRESSED == 0 { (get_varint(src, &mut pos)?, get_varint(src, &mut pos)?) } else { (0, 0) };
        let c = src.get(pos..pos + 4)?;
        let checksum = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let header = BlockHeader { magic: MAGIC, version: VERSION_V9, flags, uncompressed_len, token_count, token_bytes, literal_len, checksum, offset_bytes: 0, extras_bytes: 0 };
        Some((header, pos + 4))
    }
}
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
pub const MIN_MATCH_LEN: usize = 7;
/// Minimum match of the dense retry parse (FLAG_DENSE blocks). Data such as
/// 12-bit images has its redundancy in 4- and 5-byte matches.
pub const MIN_MATCH_LEN_DENSE: usize = 5;
/// Minimum match of the turbo level (FLAG_TURBO blocks). Decode scales
/// with tokens per byte; measured on Silesia (M1 Max, default 6.9 GB/s at
/// 2.19): 8 -> 8.2 GB/s at 2.055, 9 -> 8.8 at 1.975, 10 -> 9.3 at 1.884,
/// 12 -> 10.5 at 1.751. 10 is the +35% point.
pub const MIN_MATCH_LEN_TURBO: usize = 10;

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
pub const MATCH_CODE_BIAS_TURBO: usize = MIN_MATCH_LEN_TURBO - 1;
pub const MATCH_DIRECT_MAX: usize = MATCH_CODE_BIAS + 14;

/// Escaped lengths travel in a byte stream: one byte holds `value - base`
/// when that is below 255; the byte 255 means a little-endian u16 follows
/// holding `value - base - 255`.
pub const ESCAPE_BASE_LIT: usize = LIT_DIRECT_MAX + 1;
pub const ESCAPE_BASE_MATCH: usize = MATCH_DIRECT_MAX + 1;
pub const ESCAPE_BASE_MATCH_DENSE: usize = MATCH_CODE_BIAS_DENSE + 15;
pub const ESCAPE_BASE_MATCH_TURBO: usize = MATCH_CODE_BIAS_TURBO + 15;
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
/// Block was parsed at minimum match 8 (turbo level): match lengths are
/// biased by MATCH_CODE_BIAS_TURBO.
pub const FLAG_TURBO: u16 = 16;
/// The block's checksum is CRC-32C (v0.14.2 on); without this flag it is
/// the Adler-like sum below, which keeps only 16 bits of its weighted
/// half and so misses, for one, two bytes swapped 8 KB apart.
pub const FLAG_CRC32C: u16 = 32;
/// A repeat-offset code after zero literals names the *other* repeats
/// (v0.14.3 on): a match with no literal before it cannot be the last
/// offset going on (that match would have been longer), so code 0 is
/// the second repeat, 1 the third, 2 the last one — what zstd does, and
/// half a bit less per repeat on data whose records alternate sources.
pub const FLAG_LL0_REP: u16 = 64;

/// Units the parallel paths cut an input into: each is compressed on its
/// own (its first block carries FLAG_CHAIN_RESET) and decodes on its own,
/// so decoding runs one unit per core and a unit is the granule of random
/// access. A unit's first bytes have no window and empty tables, so the
/// unit size is a ratio trade per level (Silesia, 10 cores, against the
/// sequential ratio): the v6 levels lose 3% at 256 KB, 0.5% at 2 MB, 0.1%
/// at 8 MB; max 6.6% / 1.7% / 0.4%; ultra 13% / 4% (4 MB) / 0.7% (16 MB).
/// These are the smallest units: a large input takes larger ones, up to
/// `PARALLEL_UNIT_LARGEST`, as long as every core keeps two
/// (`parallel_unit`). On repetitive data the small unit costs more than
/// Silesia says: JSON events lose 4.7% at 8 MB, 1.2% at 32 MB, 0.6% at
/// 64 MB (an SQL dump 0.7% / 0.2% / 0.1%).
pub const PARALLEL_UNIT_V6: usize = 2 * 1024 * 1024;
pub const PARALLEL_UNIT_MAX: usize = 8 * 1024 * 1024;
pub const PARALLEL_UNIT_ULTRA: usize = 16 * 1024 * 1024;
pub const PARALLEL_UNIT_LARGEST: usize = 128 * 1024 * 1024;

/// The unit for an input of `len` bytes on `threads` cores, at least
/// `smallest`: as large as leaves one unit per core (the long-distance
/// matcher reaches 128 MB, and a unit is its window), capped at
/// `PARALLEL_UNIT_LARGEST`, rounded down to a megabyte.
pub fn parallel_unit(len: usize, threads: usize, smallest: usize) -> usize {
    let fair = len / threads.max(1);
    (fair.min(PARALLEL_UNIT_LARGEST) & !((1 << 20) - 1)).max(smallest)
}

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
pub static TOKEN_TABLE_TURBO: [u32; 256] = token_table(MATCH_CODE_BIAS_TURBO);

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
        } else if is_coded_version(self.version) {
            self.token_bytes as usize
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
        if is_coded_version(self.version) {
            return u <= MAX_BLOCK_SIZE
                && self.token_count as usize <= u / 3 + 1
                && self.literal_len as usize <= u
                && self.token_bytes as usize <= 2 * u + 4096
                && self.offset_bytes == 0
                && self.extras_bytes == 0;
        }
        u <= MAX_BLOCK_SIZE
            && self.token_count as usize <= u
            && self.token_bytes as usize <= u
            && self.offset_bytes as usize <= 2 * u
            && self.extras_bytes as usize <= 3 * u
            && self.literal_len as usize <= u
    }
}

/// The checksum a block's flags say it carries.
#[inline]
pub fn block_checksum(flags: u16, data: &[u8]) -> u32 {
    if flags & FLAG_CRC32C != 0 { crc32c(data) } else { compute_checksum(data) }
}

/// CRC-32C (Castagnoli), the hardware instruction on aarch64 and
/// x86-64, a table otherwise; every block written since v0.14.2 is
/// checked with it.
pub fn crc32c(data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            return unsafe { crc32c_aarch64(data) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { crc32c_x86(data) };
        }
    }
    crc32c_table(data)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_aarch64(data: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    let mut c = !0u32;
    let (chunks, tail) = data.as_chunks::<8>();
    for w in chunks {
        c = __crc32cd(c, u64::from_le_bytes(*w));
    }
    for &b in tail {
        c = __crc32cb(c, b);
    }
    !c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_x86(data: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    let mut c = !0u64;
    let (chunks, tail) = data.as_chunks::<8>();
    for w in chunks {
        c = _mm_crc32_u64(c, u64::from_le_bytes(*w));
    }
    let mut c = c as u32;
    for &b in tail {
        c = _mm_crc32_u8(c, b);
    }
    !c
}

pub fn crc32c_table(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0x82F6_3B78 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut c = !0u32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
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
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { crate::neon_checksum::checksum_neon(data) };
    }
    #[allow(unreachable_code)]
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

#[cfg(test)]
mod checksum_tests {
    use super::*;

    #[test]
    fn crc32c_is_castagnoli_and_sees_what_the_sum_missed() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c_table(b"123456789"), 0xE306_9283);
        let mut a: Vec<u8> = (0..20000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let base = compute_checksum(&a);
        let (i, j) = (2297usize, 2297 + 8192);
        (a[i], a[j]) = (0xD8, 0x60);
        let sum_before = compute_checksum(&a);
        let crc_before = crc32c(&a);
        a.swap(i, j);
        assert_eq!(compute_checksum(&a), sum_before, "the Adler-like sum misses this swap");
        assert_ne!(crc32c(&a), crc_before, "the CRC sees it");
        let _ = base;
        for n in [0usize, 1, 7, 8, 9, 63, 64, 1000] {
            assert_eq!(crc32c(&a[..n]), crc32c_table(&a[..n]), "{n}");
        }
    }
}
