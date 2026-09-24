//! Zstd frames reproduced: the reference compressor (zstd 1.5.5,
//! level 1, the one-shot `ZSTD_compress`) ported step for step, so a
//! frame it wrote is made again from its content and the frame's bytes
//! need not be kept. Parquet's pages are the case, as with snappy in
//! `resnappy`: a data lake's files hold their columns as zstd pages,
//! and opening a page lets its values be modeled instead of its LZ
//! tokens. Every decision that shapes the bytes is the reference's:
//! the parameters by input size, the window, the fast match finder
//! with its growing step, the literals' Huffman table or the previous
//! block's, the three sequence tables chosen among predefined, RLE
//! and fresh, and the capacity rules of a `ZSTD_compressBound` buffer.
//! `reproduce` finds which build wrote a frame (`Build`); a frame no
//! build made (another level, another version's rules, a dictionary,
//! a checksum) is reported as not reproduced and its bytes are kept.
//! `decompress` is a plain single-frame decoder for the same purpose.

mod block;
mod decode;
mod fast;
mod fse;
mod huf;

pub use decode::decompress;

/// `ZSTD_compressionParameters` for the fast strategy: a level row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CParams {
    pub window_log: u32,
    pub chain_log: u32,
    pub hash_log: u32,
    pub search_log: u32,
    pub min_match: u32,
    pub target_length: u32,
}

const fn row(window_log: u32, chain_log: u32, hash_log: u32, search_log: u32, min_match: u32, target_length: u32) -> CParams {
    CParams { window_log, chain_log, hash_log, search_log, min_match, target_length }
}

/// A build of the reference: what varies between versions on the
/// level-1 path. The rows are `clevels.h`'s level-1 entries per input
/// size class (over 256 KB, up to 256 KB, up to 128 KB, up to 16 KB).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Build {
    pub version: &'static str,
    pub rows: [CParams; 4],
}

/// The builds `reproduce` tries.
pub const BUILDS: [Build; 1] = [Build {
    version: "1.5.5",
    rows: [row(19, 13, 14, 1, 7, 0), row(18, 13, 14, 1, 6, 0), row(17, 12, 13, 1, 6, 0), row(14, 14, 15, 1, 5, 0)],
}];

const MAGIC: u32 = 0xFD2FB528;
const BLOCK_SIZE_MAX: usize = 1 << 17;
const WINDOW_LOG_ABSOLUTE_MIN: u32 = 10;
const HASH_LOG_MIN: u32 = 6;

impl Build {
    /// `ZSTD_getCParams_internal` then `ZSTD_adjustCParams_internal`
    /// for a known input size and no dictionary: the row by size, the
    /// window shrunk to the input, the hash and chain logs clamped to
    /// it (before the window is raised to the format's minimum).
    pub fn params(&self, n: usize) -> CParams {
        let table = (n <= 256 << 10) as usize + (n <= 128 << 10) as usize + (n <= 16 << 10) as usize;
        let mut p = self.rows[table];
        if n <= 1 << 30 {
            let src_log = if n < 1 << HASH_LOG_MIN { HASH_LOG_MIN } else { fse::highbit(n as u32 - 1) + 1 };
            if p.window_log > src_log {
                p.window_log = src_log;
            }
        }
        if p.hash_log > p.window_log + 1 {
            p.hash_log = p.window_log + 1;
        }
        if p.chain_log > p.window_log {
            p.chain_log -= p.chain_log - p.window_log;
        }
        if p.window_log < WINDOW_LOG_ABSOLUTE_MIN {
            p.window_log = WINDOW_LOG_ABSOLUTE_MIN;
        }
        p
    }
}

/// `ZSTD_compressBound`: the buffer the one-shot API writes into; its
/// remaining room decides a few edge cases of the reference.
fn compress_bound(n: usize) -> usize {
    n + (n >> 8) + if n < 128 << 10 { ((128 << 10) - n) >> 11 } else { 0 }
}

/// `ZSTD_isRLE`.
fn is_rle(s: &[u8]) -> bool {
    s.iter().all(|&b| b == s[0])
}

/// `ZSTD_compressedBlockState_t`: the repcodes and entropy tables a
/// block starts from.
#[derive(Clone, Copy)]
struct BlockState {
    rep: [u32; 3],
    entropy: block::Entropy,
}

/// `input` as the reference compressor of `build` writes it.
pub fn compress(input: &[u8], build: Build) -> Vec<u8> {
    let n = input.len();
    let p = build.params(n);
    let mut cap = compress_bound(n);
    let mut out = Vec::with_capacity(n / 2 + 64);
    // ZSTD_writeFrameHeader: content size, no checksum, no dictionary.
    let window_size = 1u64 << p.window_log;
    let single_segment = window_size >= n as u64;
    let fcs_code = (n >= 256) as u8 + (n >= 65536 + 256) as u8 + (n as u64 >= 0xFFFF_FFFF) as u8;
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.push((fcs_code << 6) + ((single_segment as u8) << 5));
    if !single_segment {
        out.push(((p.window_log - WINDOW_LOG_ABSOLUTE_MIN) << 3) as u8);
    }
    match fcs_code {
        0 => {
            if single_segment {
                out.push(n as u8);
            }
        }
        1 => out.extend_from_slice(&((n - 256) as u16).to_le_bytes()),
        2 => out.extend_from_slice(&(n as u32).to_le_bytes()),
        _ => out.extend_from_slice(&(n as u64).to_le_bytes()),
    }
    cap -= out.len();
    if n == 0 {
        // ZSTD_writeEpilogue: an empty last block.
        out.extend_from_slice(&[1, 0, 0]);
        return out;
    }
    let block_size = BLOCK_SIZE_MAX.min((window_size as usize).min(n).max(1));
    let mut table = vec![0u32; 1 << p.hash_log];
    let mut window = fast::Window::new();
    let mut prev = BlockState { rep: [1, 4, 8], entropy: block::Entropy::fresh() };
    let mut next = prev;
    let mut first = true;
    let mut pos = 0;
    while pos < n {
        let size = block_size.min(n - pos);
        let last = pos + size == n;
        window.enforce_max_dist(pos, p.window_log);
        // ZSTD_compressBlock_internal: 0 is a raw block, 1 an RLE one.
        let src = &input[pos..pos + size];
        let mut coded = Vec::new();
        let mut c_size = 0usize;
        if size >= 7 {
            next.rep = prev.rep;
            let store = fast::compress_block(input, pos, pos + size, &mut table, &p, &window, &mut next.rep);
            if let Some(c) = block::compress(&store, &prev.entropy, &mut next.entropy, cap - 3, size) {
                c_size = c.len();
                coded = c;
            }
            if !first && c_size < 25 && is_rle(src) {
                c_size = 1;
            }
            if c_size > 1 {
                std::mem::swap(&mut prev, &mut next);
            }
            if prev.entropy.fse.of_repeat == fse::Repeat::Valid {
                prev.entropy.fse.of_repeat = fse::Repeat::Check;
            }
        }
        let header = |kind: u32, size: usize| (last as u32 + (kind << 1) + ((size as u32) << 3)).to_le_bytes();
        match c_size {
            0 => {
                out.extend_from_slice(&header(0, size)[..3]);
                out.extend_from_slice(src);
            }
            1 => {
                out.extend_from_slice(&header(1, size)[..3]);
                out.push(src[0]);
            }
            _ => {
                out.extend_from_slice(&header(2, c_size)[..3]);
                out.extend_from_slice(&coded);
            }
        }
        cap -= match c_size {
            0 => 3 + size,
            1 => 4,
            c => 3 + c,
        };
        pos += size;
        first = false;
    }
    out
}

/// A frame's content and the build whose compressor writes exactly
/// the frame again; None when none does, or the frame is bad.
pub fn reproduce(frame: &[u8]) -> Option<(Vec<u8>, Build)> {
    let plain = decompress(frame)?;
    for build in BUILDS {
        if compress(&plain, build) == frame {
            return Some((plain, build));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/zstd/").to_string() + name).unwrap()
    }

    #[test]
    fn frames_written_by_zstd_1_5_5_come_back_byte_for_byte() {
        // Level-1 frames of ZSTD_compress: text and mixed data over one
        // and two blocks, planes, zeros, random bytes, a 50-byte and a
        // 3 KB input, three Parquet pages written by pyarrow's zstd, and
        // zeros with a 150-byte run copied 523,000 bytes later across the
        // block boundary at 512 KB: the repcode carried into the fifth
        // block reaches almost a full window back, which only survives
        // when the window's limit is set from the block's start.
        for name in ["text50", "text3k", "page0", "page1", "page2", "random40k", "zeros100k", "mixed170k", "text200k", "planes300k", "farrep526k"] {
            let frame = fixture(&format!("{name}.zst"));
            let raw = fixture(&format!("{name}.raw"));
            assert_eq!(decompress(&frame).as_deref(), Some(&raw[..]), "{name} decodes");
            let made = compress(&raw, BUILDS[0]);
            if made != frame {
                let at = made.iter().zip(&frame).position(|(a, b)| a != b).unwrap_or(made.len().min(frame.len()));
                panic!("{name}: {} bytes made, {} expected, first difference at {at}", made.len(), frame.len());
            }
            let (plain, build) = reproduce(&frame).unwrap_or_else(|| panic!("{name} is not reproduced by any build"));
            assert_eq!(plain, raw);
            assert_eq!(build, BUILDS[0], "{name}");
        }
    }

    #[test]
    fn round_trips_on_synthetic_data() {
        let mut x = 7u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let words: Vec<&[u8]> = vec![b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"lazy ", b"dogs ", b"and ", b"cats "];
        let mut d = Vec::new();
        while d.len() < 300_000 {
            d.extend_from_slice(words[(rnd() % 10) as usize]);
        }
        d.extend(std::iter::repeat(0u8).take(200_000));
        for _ in 0..50_000 {
            d.push(rnd() as u8);
        }
        let head = d[..100_000].to_vec();
        d.extend_from_slice(&head);
        // Every size class, several blocks, a window smaller than the input.
        for len in [0, 1, 5, 7, 50, 300, 1000, 5000, 16 << 10, 40_000, 128 << 10, (128 << 10) + 1, 200_000, 300_000, d.len()] {
            let s = &d[..len];
            let c = compress(s, BUILDS[0]);
            assert_eq!(decompress(&c).as_deref(), Some(s), "{len} bytes round-trip");
            assert_eq!(reproduce(&c).map(|(_, b)| b), Some(BUILDS[0]), "{len} bytes reproduce");
        }
        let zeros = vec![0u8; 300_000];
        let c = compress(&zeros, BUILDS[0]);
        assert!(c.len() < 40, "zeros: {} bytes", c.len());
        assert_eq!(decompress(&c).unwrap(), zeros);
        assert!(decompress(&c[..c.len() - 1]).is_none(), "a truncated frame is refused");
        assert!(decompress(b"\x28\xb5\x2f\xfd\x20\x05\x01\x00\x00").is_none(), "a frame short of its content size is refused");
    }
}
