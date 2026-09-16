pub const MAGIC: u32 = 0x53494D44; // "SIMD"
pub const CURRENT_VERSION: u16 = 2;
pub const MAX_BLOCK_SIZE: usize = 65536; // 64 KB
pub const PADDING: usize = 64; // Safe SIMD read/write margin

pub const MAX_LIT_LEN: usize = 31;     // 5 bits (0..31)
pub const MAX_MATCH_LEN: usize = 2047; // 11 bits (0..2047)
pub const MIN_MATCH_LEN: usize = 4;

pub const FLAG_COMPRESSED: u16 = 0;
pub const FLAG_RAW_UNCOMPRESSED: u16 = 1;

/// A compact 16-bit descriptor:
/// - bits 0..4:  literal length (0..31)
/// - bits 5..15: match length (0..2047)
/// Format v2: lit_len == 31 && match_len == 0 encodes an extended literal run (> 31 bytes),
/// where the true length is read as one u16 from the offset stream.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Token(pub u16);

impl Token {
    #[inline(always)]
    pub fn new(lit_len: usize, match_len: usize) -> Self {
        debug_assert!(lit_len <= MAX_LIT_LEN);
        debug_assert!(match_len <= MAX_MATCH_LEN);
        Self((lit_len as u16) | ((match_len as u16) << 5))
    }

    #[inline(always)]
    pub fn lit_len(self) -> usize {
        (self.0 & 0x1F) as usize
    }

    #[inline(always)]
    pub fn match_len(self) -> usize {
        ((self.0 >> 5) & 0x7FF) as usize
    }

    #[inline(always)]
    pub fn is_extended_literal(self) -> bool {
        self.0 == 31
    }
}

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
    pub literal_len: u32,
}

pub const HEADER_SIZE: usize = std::mem::size_of::<BlockHeader>();

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
