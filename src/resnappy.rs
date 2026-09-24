//! Snappy streams reproduced: the reference compressor (google/snappy
//! 1.2, level 1) ported step for step, so a stream it wrote is made
//! again from its content and the stream's bytes need not be kept.
//! Parquet's pages are the case: a data lake's files hold their
//! columns as snappy or zstd pages, and opening a page lets its
//! values be modeled instead of its LZ tokens. Builds of the
//! reference differ in the hash of four bytes (a multiply, or the
//! CRC32C instruction where the build had it) and in the table's
//! size (2^14 entries up to 1.1.10, 2^15 since 1.2.0); `reproduce`
//! finds which build wrote a stream (`Build`). A stream no build
//! made (another implementation, level 2) is reported as not
//! reproduced, and its bytes are kept as they are. The Rust `snap`
//! crate (polars' writer) is the same compressor with the older hash,
//! shifted by the table's size (`Hash::Shift`).
//! No dependency: the format is a varint length then literal and copy
//! tokens, sixty lines to decode.

/// Blocks the compressor cuts the input into; each has its own table.
const BLOCK: usize = 1 << 16;
const MIN_TABLE_BITS: u32 = 8;
const MARGIN: usize = 15;

/// Which hash of four bytes the writer's build used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hash {
    /// `(0x1e35a7bd * bytes) >> (31 - table bits)`: builds without a CRC instruction.
    Multiply,
    /// `crc32c(bytes, mask)`: x86 with SSE4.2, ARM with the CRC extension.
    Crc,
    /// `(0x1e35a7bd * bytes) >> (32 - log2(table size))`: the Rust
    /// `snap` crate, as C++ snappy up to 1.1.8. The same entry as the
    /// multiply at the largest table, another one below it (a block
    /// under 8 KB).
    Shift,
}

/// A build of the reference: its hash and its largest table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Build {
    pub hash: Hash,
    /// 14 up to snappy 1.1.10, 15 since 1.2.0.
    pub table_bits: u32,
}

/// The builds `reproduce` tries, the commonest first (pyarrow's
/// wheels: 1.1.10 with the multiply); a container keeps a build as
/// its index here, so new builds go at the end.
pub const BUILDS: [Build; 5] = [
    Build { hash: Hash::Multiply, table_bits: 14 },
    Build { hash: Hash::Multiply, table_bits: 15 },
    Build { hash: Hash::Crc, table_bits: 14 },
    Build { hash: Hash::Crc, table_bits: 15 },
    Build { hash: Hash::Shift, table_bits: 14 },
];

/// The CRC32C (Castagnoli) byte table, as the instruction computes it.
const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0x82F63B78 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}
static CRC: [u32; 256] = crc_table();

/// `_mm_crc32_u32(crc, v)` / `__crc32cw(crc, v)`: the four bytes of
/// `v`, least significant first, folded into `crc`.
fn crc32c_u32(mut crc: u32, v: u32) -> u32 {
    for b in v.to_le_bytes() {
        crc = CRC[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc
}

fn table_size(n: usize, max_bits: u32) -> usize {
    if n > 1 << max_bits {
        1 << max_bits
    } else if n < 1 << MIN_TABLE_BITS {
        1 << MIN_TABLE_BITS
    } else {
        2 << (31 - (n as u32 - 1).leading_zeros())
    }
}

/// The table entry for four bytes: `mask` is twice the table's last
/// index (the reference indexes bytes of a u16 table).
#[inline]
fn entry(build: Build, bytes: u32, mask: u32) -> usize {
    let h = match build.hash {
        Hash::Multiply => 0x1e35a7bdu32.wrapping_mul(bytes) >> (31 - build.table_bits),
        Hash::Crc => crc32c_u32(bytes, mask),
        // `mask` has log2(table size) bits set.
        Hash::Shift => 0x1e35a7bdu32.wrapping_mul(bytes) >> (31 - mask.count_ones()),
    };
    ((h & mask) >> 1) as usize
}

#[inline]
fn load32(s: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([s[at], s[at + 1], s[at + 2], s[at + 3]])
}

fn emit_literal(out: &mut Vec<u8>, lit: &[u8]) {
    let n = lit.len() - 1;
    if n < 60 {
        out.push((n << 2) as u8);
    } else {
        let count = ((31 - (n as u32).leading_zeros()) >> 3) + 1;
        out.push(((59 + count) << 2) as u8);
        out.extend_from_slice(&(n as u32).to_le_bytes()[..count as usize]);
    }
    out.extend_from_slice(lit);
}

fn emit_copy_at_most_64(out: &mut Vec<u8>, offset: usize, len: usize) {
    if len < 12 && offset < 2048 {
        out.push((1 | ((len - 4) << 2) | ((offset >> 8) << 5)) as u8);
        out.push(offset as u8);
    } else {
        out.push((2 | ((len - 1) << 2)) as u8);
        out.extend_from_slice(&(offset as u16).to_le_bytes());
    }
}

fn emit_copy(out: &mut Vec<u8>, offset: usize, mut len: usize) {
    if len < 12 {
        emit_copy_at_most_64(out, offset, len);
        return;
    }
    while len >= 68 {
        emit_copy_at_most_64(out, offset, 64);
        len -= 64;
    }
    if len > 64 {
        emit_copy_at_most_64(out, offset, 60);
        len -= 60;
    }
    emit_copy_at_most_64(out, offset, len);
}

/// Bytes matching from `a` and `b` on, up to `end`.
fn match_len(s: &[u8], a: usize, b: usize, end: usize) -> usize {
    let mut n = 0;
    while b + n < end && s[a + n] == s[b + n] {
        n += 1;
    }
    n
}

/// One block of the reference's `CompressFragment`.
fn compress_block(input: &[u8], build: Build, out: &mut Vec<u8>) {
    let n = input.len();
    let size = table_size(n, build.table_bits);
    let mask = (2 * (size - 1)) as u32;
    let mut table = vec![0u16; size];
    let mut ip = 0usize;
    'blocks: {
        if n < MARGIN {
            break 'blocks;
        }
        let limit = n - MARGIN;
        loop {
            let next_emit = ip;
            ip += 1;
            let mut skip = 32u32;
            let mut candidate = 0usize;
            let mut found = false;
            // Sixteen positions probed one by one before the skipping scan.
            if ip <= limit && limit - ip >= 16 {
                for i in 0..16 {
                    let dword = load32(input, ip + i);
                    let e = entry(build, dword, mask);
                    candidate = table[e] as usize;
                    table[e] = (ip + i) as u16;
                    if load32(input, candidate) == dword {
                        emit_literal(out, &input[next_emit..ip + i]);
                        ip += i;
                        found = true;
                        break;
                    }
                }
                if !found {
                    ip += 16;
                    skip += 16;
                }
            }
            if !found {
                loop {
                    let dword = load32(input, ip);
                    let e = entry(build, dword, mask);
                    let between = (skip >> 5) as usize;
                    skip += between as u32;
                    let next_ip = ip + between;
                    if next_ip > limit {
                        ip = next_emit;
                        break 'blocks;
                    }
                    candidate = table[e] as usize;
                    table[e] = ip as u16;
                    if dword == load32(input, candidate) {
                        break;
                    }
                    ip = next_ip;
                }
                emit_literal(out, &input[next_emit..ip]);
            }
            // A match at ip; copies until the bytes after one do not match.
            loop {
                let base = ip;
                let matched = 4 + match_len(input, candidate + 4, ip + 4, n);
                ip += matched;
                emit_copy(out, base - candidate, matched);
                if ip >= limit {
                    break 'blocks;
                }
                let e = entry(build, load32(input, ip - 1), mask);
                table[e] = (ip - 1) as u16;
                let dword = load32(input, ip);
                let e = entry(build, dword, mask);
                candidate = table[e] as usize;
                table[e] = ip as u16;
                if dword != load32(input, candidate) {
                    break;
                }
            }
        }
    }
    if ip < n {
        emit_literal(out, &input[ip..]);
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// `input` as the reference compressor of `build` writes it.
pub fn compress(input: &[u8], build: Build) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2 + 32);
    put_varint(&mut out, input.len() as u32);
    for block in input.chunks(BLOCK) {
        compress_block(block, build, &mut out);
    }
    out
}

/// The content of a snappy stream, or None for a malformed one.
pub fn decompress(s: &[u8]) -> Option<Vec<u8>> {
    let mut p = 0usize;
    let (mut len, mut shift) = (0u64, 0u32);
    loop {
        let b = *s.get(p)?;
        p += 1;
        len |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return None;
        }
    }
    let len = len as usize;
    let mut out = Vec::with_capacity(len);
    while p < s.len() {
        let tag = s[p];
        p += 1;
        match tag & 3 {
            0 => {
                let mut n = (tag >> 2) as usize;
                if n >= 60 {
                    let count = n - 59;
                    let mut v = 0usize;
                    for i in 0..count {
                        v |= (*s.get(p + i)? as usize) << (8 * i);
                    }
                    p += count;
                    n = v;
                }
                let n = n + 1;
                out.extend_from_slice(s.get(p..p + n)?);
                p += n;
            }
            kind => {
                let (n, offset) = match kind {
                    1 => (4 + ((tag >> 2) & 7) as usize, (((tag >> 5) as usize) << 8) | *s.get(p)? as usize),
                    2 => (1 + (tag >> 2) as usize, u16::from_le_bytes([*s.get(p)?, *s.get(p + 1)?]) as usize),
                    _ => (1 + (tag >> 2) as usize, u32::from_le_bytes([*s.get(p)?, *s.get(p + 1)?, *s.get(p + 2)?, *s.get(p + 3)?]) as usize),
                };
                p += match kind { 1 => 1, 2 => 2, _ => 4 };
                if offset == 0 || offset > out.len() {
                    return None;
                }
                let from = out.len() - offset;
                for i in 0..n {
                    let b = out[from + i];
                    out.push(b);
                }
            }
        }
        if out.len() > len {
            return None;
        }
    }
    (out.len() == len).then_some(out)
}

/// A stream's content and the build whose compressor writes exactly
/// the stream again; None when none does, or the stream is bad.
pub fn reproduce(stream: &[u8]) -> Option<(Vec<u8>, Build)> {
    let plain = decompress(stream)?;
    for build in BUILDS {
        if compress(&plain, build) == stream {
            return Some((plain, build));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/snappy/").to_string() + name).unwrap()
    }

    #[test]
    fn pages_written_by_parquet_cpp_come_back_byte_for_byte() {
        // Three pages of a Parquet file written by pyarrow 21's macOS
        // wheel (snappy 1.1.10, the multiply hash): 2 KB, 2.6 KB and 60
        // KB (a block and a half).
        for i in 0..3 {
            let stream = fixture(&format!("page{i}.snappy"));
            let raw = fixture(&format!("page{i}.raw"));
            assert_eq!(decompress(&stream).as_deref(), Some(&raw[..]), "page{i} decodes");
            let (plain, build) = reproduce(&stream).unwrap_or_else(|| panic!("page{i} is not reproduced by any build"));
            assert_eq!(plain, raw);
            assert_eq!(build, BUILDS[0], "page{i}");
        }
    }

    #[test]
    fn a_stream_the_snap_crate_wrote_comes_back_with_its_hash() {
        // 3000 bytes of words compressed by the Rust `snap` crate 1.1
        // (through cramjam 2.11), as polars writes its pages: a table of
        // 4096 entries indexed by the product's top 12 bits, not by the
        // bits the 1.1.10 multiply keeps.
        let stream = fixture("snap0.snappy");
        let raw = fixture("snap0.raw");
        let (plain, build) = reproduce(&stream).expect("the snap crate's stream is reproduced");
        assert_eq!(plain, raw);
        assert_eq!(build, Build { hash: Hash::Shift, table_bits: 14 });
        assert_ne!(compress(&raw, BUILDS[0]), stream);
        // At the largest table (a block over 8 KB) the two pick the same entries.
        let big: Vec<u8> = raw.iter().enumerate().map(|(i, &b)| b ^ (i % 7 == 0) as u8).cycle().take(20_000).collect();
        assert_eq!(compress(&big, BUILDS[0]), compress(&big, Build { hash: Hash::Shift, table_bits: 14 }));
    }

    #[test]
    fn round_trips_with_every_build_and_long_runs() {
        let mut x = 7u64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let words: Vec<&[u8]> = vec![b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"lazy ", b"dogs ", b"and ", b"cats "];
        let mut d = Vec::new();
        while d.len() < 40_000 { d.extend_from_slice(words[(rnd() % 10) as usize]); }
        d.extend(std::iter::repeat(0u8).take(70_000));
        for _ in 0..30_000 { d.push(rnd() as u8); }
        let head = d[..50_000].to_vec();
        d.extend_from_slice(&head);
        for build in BUILDS {
            let c = compress(&d, build);
            assert!(c.len() < d.len() / 2, "{build:?}: {} of {}", c.len(), d.len());
            assert_eq!(decompress(&c).unwrap(), d);
            // Two builds can write the same bytes; whichever is found writes them again.
            let found = reproduce(&c).unwrap().1;
            assert_eq!(compress(&d, found), c, "{build:?} found as {found:?}");
        }
        assert_eq!(decompress(&compress(b"", BUILDS[0])).unwrap(), b"");
        assert_eq!(decompress(&compress(b"abc", BUILDS[0])).unwrap(), b"abc");
        assert!(decompress(&[5, 0x08, b'a']).is_none(), "a short stream is refused");
    }
}
