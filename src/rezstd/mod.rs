//! Zstd frames reproduced: the reference compressor (zstd 1.5.2 to
//! 1.5.7 at level 1 and level 3, the one-shot `ZSTD_compress` and the
//! CLI's stream and jobs) ported step for step, so a frame it wrote is made again
//! from its content and the frame's bytes need not be kept. Parquet's
//! pages are the case, as with snappy in `resnappy`: a data lake's
//! files hold their columns as zstd pages, and opening a page lets its
//! values be modeled instead of its LZ tokens; a `.zst` file is the
//! other. Every decision that shapes the bytes is the reference's: the
//! parameters by input size, the window, the fast and double-fast
//! match finders with their growing steps, the literals' Huffman table
//! or the previous block's, the three sequence tables chosen among
//! predefined, RLE and fresh, the capacity rules of the buffer written
//! into, and for the CLI the ring its input goes through (a window plus
//! a block: past its end the finders run their `extDict` variants) and
//! the checksum. `reproduce` finds which build wrote a frame (`Build`);
//! a frame no build made (another level or version, a dictionary) is
//! reported as not reproduced and its bytes are kept. `decompress` is
//! a plain single-frame decoder for the same purpose.

mod block;
mod decode;
mod dfast;
mod fast;
mod fse;
mod huf;
mod split;
mod xxh64;

pub use decode::decompress;

/// `ZSTD_strategy`, as far as ported: `ZSTD_fast` (level 1) and
/// `ZSTD_dfast` (level 3); the numeric values are the reference's,
/// which its rules use in arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Fast = 1,
    DoubleFast = 2,
}

/// `ZSTD_compressionParameters`: a level row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CParams {
    pub window_log: u32,
    pub chain_log: u32,
    pub hash_log: u32,
    pub search_log: u32,
    pub min_match: u32,
    pub target_length: u32,
    pub strategy: Strategy,
}

const fn row(window_log: u32, chain_log: u32, hash_log: u32, search_log: u32, min_match: u32, target_length: u32, strategy: Strategy) -> CParams {
    CParams { window_log, chain_log, hash_log, search_log, min_match, target_length, strategy }
}

/// The compression levels ported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    One = 1,
    Three = 3,
}

/// The reference's versions whose level-1 and level-3 paths differ,
/// in order. 1.5.2's fast finder writes its pipelined table entry
/// after a match rather than on finding it, keeps one saved repcode
/// for both invalidated ones, and runs the older one-position loop
/// past the stream ring's wrap. `V1_5_5` stands for 1.5.4, 1.5.5 and
/// 1.5.6, which write the same bytes on these paths. 1.5.7 cuts full
/// blocks where their content changes (`split.rs`) and lets the
/// double-fast finder accept a candidate at the window's lowest index
/// and keep the short match unless the long match a byte ahead is
/// strictly longer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Version {
    V1_5_2,
    V1_5_5,
    V1_5_7,
}

/// The finder rules that changed between versions (`Build::rules`).
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    /// The fast finder writes the entry for the position after the
    /// match's start when it finds the match (1.5.4), not after it.
    pub hash1_in_search: bool,
    /// Two saved repcodes, the first rotating into the second when
    /// repcode 1 was replaced (1.5.4); before, one shared value.
    pub saved_reps_rotate: bool,
    /// The fast `extDict` loop is the pipelined one (1.5.4).
    pub fast_ext_pipelined: bool,
    /// The double-fast finder's 1.5.7 rules.
    pub dfast_longer_wins: bool,
    /// Full blocks cut where their content changes (1.5.7).
    pub splits_blocks: bool,
}

/// How the input reached the compressor. `OneShot` is `ZSTD_compress`
/// in one call with a `ZSTD_compressBound` buffer. `Stream` is the CLI
/// with `--single-thread`: `ZSTD_compressStream2` fed 128 KB at a time,
/// each chunk into a fresh `ZSTD_CStreamOutSize` buffer, the input
/// copied through a ring of a window plus a block whose wrap turns the
/// finders to their `extDict` variants. `Cli` is the CLI's default
/// (any number of workers): inputs up to 512 KB go as `Stream`; larger
/// ones are cut into jobs of `1 << max(20, windowLog + 2)` bytes, each
/// compressed by a fresh context whose tables are first filled from the
/// `1 << (windowLog - 3)` bytes before the job (every third position),
/// with repcodes zeroed, a `ZSTD_compressBound(jobSize)` buffer, and no
/// RLE for the job's first block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Writer {
    OneShot,
    Stream,
    Cli,
}

/// A build of the reference: its version, level, how it was fed and
/// whether it appended the content's checksum (the CLI does; the
/// one-shot API does not).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Build {
    pub version: Version,
    pub level: Level,
    pub writer: Writer,
    pub checksum: bool,
}

const fn build(version: Version, level: Level, writer: Writer) -> Build {
    Build { version, level, writer, checksum: false }
}

/// The builds `reproduce` tries (with the checksum read from the
/// frame), newest version first.
pub const BUILDS: [Build; 18] = [
    build(Version::V1_5_7, Level::One, Writer::OneShot),
    build(Version::V1_5_7, Level::Three, Writer::OneShot),
    build(Version::V1_5_7, Level::One, Writer::Cli),
    build(Version::V1_5_7, Level::Three, Writer::Cli),
    build(Version::V1_5_7, Level::One, Writer::Stream),
    build(Version::V1_5_7, Level::Three, Writer::Stream),
    build(Version::V1_5_5, Level::One, Writer::OneShot),
    build(Version::V1_5_5, Level::Three, Writer::OneShot),
    build(Version::V1_5_5, Level::One, Writer::Cli),
    build(Version::V1_5_5, Level::Three, Writer::Cli),
    build(Version::V1_5_5, Level::One, Writer::Stream),
    build(Version::V1_5_5, Level::Three, Writer::Stream),
    build(Version::V1_5_2, Level::One, Writer::OneShot),
    build(Version::V1_5_2, Level::Three, Writer::OneShot),
    build(Version::V1_5_2, Level::One, Writer::Cli),
    build(Version::V1_5_2, Level::Three, Writer::Cli),
    build(Version::V1_5_2, Level::One, Writer::Stream),
    build(Version::V1_5_2, Level::Three, Writer::Stream),
];

/// `clevels.h`: the rows of a level by input size class (over 256 KB,
/// up to 256 KB, up to 128 KB, up to 16 KB).
const LEVEL1_ROWS: [CParams; 4] = [
    row(19, 13, 14, 1, 7, 0, Strategy::Fast),
    row(18, 13, 14, 1, 6, 0, Strategy::Fast),
    row(17, 12, 13, 1, 6, 0, Strategy::Fast),
    row(14, 14, 15, 1, 5, 0, Strategy::Fast),
];
const LEVEL3_ROWS: [CParams; 4] = [
    row(21, 16, 17, 1, 5, 0, Strategy::DoubleFast),
    row(18, 16, 16, 1, 4, 0, Strategy::DoubleFast),
    row(17, 15, 16, 2, 5, 0, Strategy::DoubleFast),
    row(14, 14, 15, 2, 4, 0, Strategy::DoubleFast),
];

const MAGIC: u32 = 0xFD2FB528;
const BLOCK_SIZE_MAX: usize = 1 << 17;
const WINDOW_LOG_ABSOLUTE_MIN: u32 = 10;
const HASH_LOG_MIN: u32 = 6;

/// `ZSTD_compressBound`: the buffer the one-shot API writes into; its
/// remaining room decides a few edge cases of the reference.
fn compress_bound(n: usize) -> usize {
    n + (n >> 8) + if n < 128 << 10 { ((128 << 10) - n) >> 11 } else { 0 }
}

/// `ZSTD_CStreamOutSize`: the CLI's output buffer for each chunk.
const CSTREAM_OUT_SIZE: usize = 131072 + (131072 >> 8) + 3 + 4;
/// `ZSTDMT_JOBSIZE_MIN`: inputs up to this size are not cut into jobs.
const JOB_SIZE_MIN: usize = 512 << 10;

impl Build {
    /// `ZSTD_getCParams_internal` then `ZSTD_adjustCParams_internal`
    /// for a known input size and no dictionary: the row by size, the
    /// window shrunk to the input, the hash and chain logs clamped to
    /// it (before the window is raised to the format's minimum).
    pub fn params(&self, n: usize) -> CParams {
        let table = (n <= 256 << 10) as usize + (n <= 128 << 10) as usize + (n <= 16 << 10) as usize;
        let mut p = match self.level {
            Level::One => LEVEL1_ROWS[table],
            Level::Three => LEVEL3_ROWS[table],
        };
        if n <= 1 << 30 {
            let src_log = if n < 1 << HASH_LOG_MIN { HASH_LOG_MIN } else { fse::highbit(n as u32 - 1) + 1 };
            if p.window_log > src_log {
                p.window_log = src_log;
            }
        }
        if p.hash_log > p.window_log + 1 {
            p.hash_log = p.window_log + 1;
        }
        // The chain log's cycle (the chain log itself for these
        // strategies) is brought down to the window log.
        if p.chain_log > p.window_log {
            p.chain_log = p.window_log;
        }
        if p.window_log < WINDOW_LOG_ABSOLUTE_MIN {
            p.window_log = WINDOW_LOG_ABSOLUTE_MIN;
        }
        p
    }

    /// The rules of this build's version.
    pub fn rules(&self) -> Rules {
        Rules {
            hash1_in_search: self.version >= Version::V1_5_5,
            saved_reps_rotate: self.version >= Version::V1_5_5,
            fast_ext_pipelined: self.version >= Version::V1_5_5,
            dfast_longer_wins: self.version >= Version::V1_5_7,
            splits_blocks: self.version >= Version::V1_5_7,
        }
    }
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

/// One compression context's pass over `input[start..end]`: its
/// blocks appended to `out`. `base` is where the context's history
/// begins (a job's overlap prefix, whose tables are filled first);
/// the input reaches the context in calls of `chunk` bytes
/// (`ZSTD_compressContinue`); `cap` is the room in the buffer written
/// into, renewed per call for the CLI's stream, whose ring of `ring`
/// bytes restarts the window at each wrap; `reps` are the repcodes the
/// context starts from; `header` is the frame header's size when this
/// context wrote it, which its savings count from the second call on.
struct Run {
    base: usize,
    start: usize,
    end: usize,
    chunk: usize,
    cap: usize,
    per_chunk: bool,
    ring: Option<usize>,
    reps: [u32; 3],
    header: usize,
}

/// `ZSTD_optimalBlockSize`: the next block's size. Before 1.5.7 a
/// block is as large as it can be; since then a full block is cut by
/// `split::split_block` once the frame's savings reach three bytes
/// (so never the first block, nor incompressible data).
fn block_size_at(src: &[u8], remaining: usize, block_size_max: usize, rules: &Rules, strategy: Strategy, savings: i64) -> usize {
    if !rules.splits_blocks || remaining < BLOCK_SIZE_MAX || block_size_max < BLOCK_SIZE_MAX {
        return remaining.min(block_size_max);
    }
    if savings < 3 {
        return BLOCK_SIZE_MAX;
    }
    split::split_block(src, if strategy == Strategy::Fast { 0 } else { 1 })
}

fn run(input: &[u8], n: usize, b: &Build, p: &CParams, r: Run, out: &mut Vec<u8>) {
    let rules = b.rules();
    let s = &input[r.base..r.end];
    let (start, end) = (r.start - r.base, r.end - r.base);
    let window_size = (1u64 << p.window_log).min((end - start) as u64).max(1) as usize;
    let block_size = BLOCK_SIZE_MAX.min(window_size);
    let mut table = vec![0u32; 1 << p.hash_log];
    let mut small = vec![0u32; if p.strategy == Strategy::DoubleFast { 1 << p.chain_log } else { 0 }];
    let mut window = fast::Window::new();
    // ZSTD_fillHashTable / ZSTD_fillDoubleHashTable (ZSTD_dtlm_fast) over
    // the prefix: every third position up to nine bytes before its end.
    if start > fast::HASH_READ_SIZE {
        let mut q = 0;
        while q + 9 < start {
            let curr = fast::idx(q);
            table[fast::hash(s, q, p.hash_log, if p.strategy == Strategy::DoubleFast { 8 } else { p.min_match })] = curr;
            if p.strategy == Strategy::DoubleFast {
                small[fast::hash(s, q, p.chain_log, p.min_match)] = curr;
            }
            q += 3;
        }
    }
    let mut prev = BlockState { rep: r.reps, entropy: block::Entropy::fresh() };
    let mut next = prev;
    let mut cap = r.cap;
    let mut first = true;
    let mut ring_pos = 0usize;
    let (mut consumed, mut produced) = (0i64, 0i64);
    let mut call_start = start;
    while call_start < end {
        let call_end = (call_start + r.chunk).min(end);
        if !first {
            if r.per_chunk {
                cap = CSTREAM_OUT_SIZE;
            }
            if r.ring.is_some() && ring_pos == 0 {
                window.restart(call_start);
            }
        }
        // ZSTD_compress_frameChunk: the savings so far decide the splits.
        let mut savings = consumed - produced;
        let call_out = out.len();
        let mut pos = call_start;
        while pos < call_end {
        let size = block_size_at(&s[pos..], call_end - pos, block_size, &rules, p.strategy, savings);
        let last = r.base + pos + size == n;
        window.enforce_max_dist(pos, p.window_log);
        // ZSTD_compressBlock_internal: 0 is a raw block, 1 an RLE one.
        let src = &s[pos..pos + size];
        let mut coded = Vec::new();
        let mut c_size = 0usize;
        if size >= 7 {
            next.rep = prev.rep;
            let store = match (p.strategy, window.has_ext_dict()) {
                (Strategy::Fast, false) => fast::compress_block(s, pos, pos + size, &mut table, p, &window, &mut next.rep, &rules),
                (Strategy::Fast, true) => fast::compress_block_ext(s, pos, pos + size, &mut table, p, &window, &mut next.rep, &rules),
                (Strategy::DoubleFast, false) => dfast::compress_block(s, pos, pos + size, &mut table, &mut small, p, &window, &mut next.rep, &rules),
                (Strategy::DoubleFast, true) => dfast::compress_block_ext(s, pos, pos + size, &mut table, &mut small, p, &window, &mut next.rep, &rules),
            };
            if let Some(c) = block::compress(&store, &prev.entropy, &mut next.entropy, cap - 3, size, p.strategy) {
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
        let written = match c_size {
            0 => 3 + size,
            1 => 4,
            c => 3 + c,
        };
        cap -= written;
        savings += size as i64 - written as i64;
        pos += size;
        first = false;
        }
        consumed += (call_end - call_start) as i64;
        produced += (out.len() - call_out) as i64 + if call_start == start { r.header as i64 } else { 0 };
        if let Some(ring) = r.ring {
            ring_pos += call_end - call_start;
            if ring_pos + block_size > ring {
                ring_pos = 0;
            }
        }
        call_start = call_end;
    }
}

/// The single-thread stream: 128 KB chunks through the input ring.
fn stream(input: &[u8], n: usize, b: &Build, p: &CParams, fh_size: usize, out: &mut Vec<u8>) {
    let window_size = (1u64 << p.window_log).min(n as u64).max(1) as usize;
    let block_size = BLOCK_SIZE_MAX.min(window_size);
    // An input of one block takes the one-shot shortcut: a single call.
    let chunk = if n <= BLOCK_SIZE_MAX { n } else { block_size };
    run(input, n, b, p, Run { base: 0, start: 0, end: n, chunk, cap: CSTREAM_OUT_SIZE - fh_size, per_chunk: true, ring: Some(window_size + block_size), reps: [1, 4, 8], header: fh_size }, out);
}

/// `input` as the reference compressor of `build` writes it.
pub fn compress(input: &[u8], build: Build) -> Vec<u8> {
    let n = input.len();
    let p = build.params(n);
    let mut out = Vec::with_capacity(n / 2 + 64);
    // ZSTD_writeFrameHeader: content size, no dictionary.
    let single_segment = (1u64 << p.window_log) >= n as u64;
    let fcs_code = (n >= 256) as u8 + (n >= 65536 + 256) as u8 + (n as u64 >= 0xFFFF_FFFF) as u8;
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.push((fcs_code << 6) + ((single_segment as u8) << 5) + ((build.checksum as u8) << 2));
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
    let fh_size = out.len();
    if n == 0 {
        // ZSTD_writeEpilogue: an empty last block.
        out.extend_from_slice(&[1, 0, 0]);
    } else {
        match build.writer {
            Writer::OneShot => run(input, n, &build, &p, Run { base: 0, start: 0, end: n, chunk: n, cap: compress_bound(n) - fh_size, per_chunk: false, ring: None, reps: [1, 4, 8], header: fh_size }, &mut out),
            Writer::Stream => stream(input, n, &build, &p, fh_size, &mut out),
            Writer::Cli if n <= JOB_SIZE_MIN => stream(input, n, &build, &p, fh_size, &mut out),
            Writer::Cli => {
                // ZSTDMT_computeTargetJobLog and ZSTDMT_computeOverlapSize
                // for the fast and double-fast strategies.
                let job = 1usize << (20.max(p.window_log + 2)).min(30);
                let overlap = 1usize << (p.window_log - 3);
                let mut start = 0;
                while start < n {
                    let end = (start + job).min(n);
                    let first = start == 0;
                    let base = if first { 0 } else { start - overlap };
                    let cap = compress_bound(job) - if first { fh_size } else { 0 };
                    // ZSTDMT_compressionJob feeds its context four blocks at a time.
                    run(input, n, &build, &p, Run { base, start, end, chunk: 4 * BLOCK_SIZE_MAX, cap, per_chunk: false, ring: None, reps: if first { [1, 4, 8] } else { [0, 0, 0] }, header: if first { fh_size } else { 0 } }, &mut out);
                    start = end;
                }
            }
        }
    }
    if build.checksum {
        out.extend_from_slice(&(xxh64::xxh64(input) as u32).to_le_bytes());
    }
    out
}

/// A frame's content and the build whose compressor writes exactly
/// the frame again; None when none does, or the frame is bad. The
/// checksum is read from the frame's descriptor; a frame carrying one
/// is tried as the CLI's first.
pub fn reproduce(frame: &[u8]) -> Option<(Vec<u8>, Build)> {
    let (plain, blocks) = decode::decompress_blocks(frame)?;
    let checksum = frame[4] & 4 != 0;
    // A block short of 128 KB before the last one is the splitter's.
    let split = blocks.iter().rev().skip(1).any(|&size| size != BLOCK_SIZE_MAX);
    let mut builds: Vec<Build> = BUILDS.iter().filter(|b| !split || b.rules().splits_blocks).map(|b| Build { checksum, ..*b }).collect();
    if checksum {
        builds.sort_by_key(|b| b.writer == Writer::OneShot);
    }
    for build in builds {
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

    const NAMES: [&str; 11] = ["text50", "text3k", "page0", "page1", "page2", "random40k", "zeros100k", "mixed170k", "text200k", "planes300k", "farrep526k"];

    const V155_L1: Build = build(Version::V1_5_5, Level::One, Writer::OneShot);
    const V155_L3: Build = build(Version::V1_5_5, Level::Three, Writer::OneShot);
    const V157_L1: Build = build(Version::V1_5_7, Level::One, Writer::OneShot);
    const V157_L3: Build = build(Version::V1_5_7, Level::Three, Writer::OneShot);

    /// `frame` is `raw` as `build` writes it, and comes back through `reproduce`.
    fn check(name: &str, raw: &[u8], frame: &[u8], build: Build) {
        assert_eq!(decompress(frame).as_deref(), Some(raw), "{name} decodes");
        let made = compress(raw, build);
        if made != frame {
            let at = made.iter().zip(frame).position(|(a, b)| a != b).unwrap_or(made.len().min(frame.len()));
            panic!("{name}: {} bytes made, {} expected, first difference at {at}", made.len(), frame.len());
        }
        let (plain, found) = reproduce(frame).unwrap_or_else(|| panic!("{name} is not reproduced by any build"));
        assert_eq!(plain, raw);
        // An earlier build may write the same bytes (a raw block at any level).
        assert!(compress(raw, found) == frame, "{name}: reproduced as {found:?}");
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
        for name in NAMES {
            let raw = fixture(&format!("{name}.raw"));
            check(name, &raw, &fixture(&format!("{name}.zst")), V155_L1);
        }
    }

    #[test]
    fn level_3_frames_come_back_byte_for_byte() {
        // The same inputs through ZSTD_compress at level 3 (double fast).
        for name in NAMES {
            let raw = fixture(&format!("{name}.raw"));
            check(name, &raw, &fixture(&format!("{name}.l3.zst")), V155_L3);
        }
    }

    #[test]
    fn frames_written_by_zstd_1_5_7_come_back_byte_for_byte() {
        // 1.5.7's zc at level 1 splits planes300k's second block at 32 KB
        // (structured bytes, then zeros); at level 3 page1 parses
        // differently (the double-fast rules) and planes300k splits; the
        // CLI at level 1 splits within its 128 KB chunk (32 KB + 96 KB).
        let planes = fixture("planes300k.raw");
        check("planes300k.v157", &planes, &fixture("planes300k.v157.zst"), V157_L1);
        check("planes300k.v157.l3", &planes, &fixture("planes300k.v157.l3.zst"), V157_L3);
        check("planes300k.v157.cli", &planes, &fixture("planes300k.v157.cli.zst"), Build { checksum: true, ..build(Version::V1_5_7, Level::One, Writer::Cli) });
        let page1 = fixture("page1.raw");
        check("page1.v157.l3", &page1, &fixture("page1.v157.l3.zst"), V157_L3);
        // The 1.5.5 frames still come back as 1.5.5's, not as any 1.5.7 build.
        assert_eq!(reproduce(&fixture("planes300k.zst")).unwrap().1, V155_L1);
        assert_eq!(reproduce(&fixture("page1.l3.zst")).unwrap().1, V155_L3);
    }

    #[test]
    fn frames_written_by_zstd_1_5_2_come_back_byte_for_byte() {
        // A 35 KB slice of a binary where 1.5.2's fast finder, writing the
        // pipelined entry after the match, keeps one 1.5.4 drops (a match
        // found five or more positions past the last, long enough); and
        // zeros to 640 KB then 20 KB of text: the text lands past the
        // single-thread stream's ring wrap, where 1.5.2 runs the older
        // extDict loop. 1.5.4 to 1.5.7 write the same bytes for both.
        let moz = fixture("moz35k.raw");
        check("moz35k", &moz, &fixture("moz35k.zst"), V155_L1);
        check("moz35k.v152", &moz, &fixture("moz35k.v152.zst"), build(Version::V1_5_2, Level::One, Writer::OneShot));
        assert_eq!(reproduce(&fixture("moz35k.v152.zst")).unwrap().1.version, Version::V1_5_2);
        let wrap = fixture("wrap676k.raw");
        let stream = |version| Build { checksum: true, ..build(version, Level::One, Writer::Stream) };
        check("wrap676k.st1", &wrap, &fixture("wrap676k.st1.zst"), stream(Version::V1_5_5));
        check("wrap676k.v152.st1", &wrap, &fixture("wrap676k.v152.st1.zst"), stream(Version::V1_5_2));
        assert_eq!(reproduce(&fixture("wrap676k.v152.st1.zst")).unwrap().1, stream(Version::V1_5_2));
        // The 1.5.5 frame also comes back as a CLI job of 1.5.7 or 1.5.5: a
        // single 676 KB job has no ring, and its contiguous parse of the
        // text is the same as the stream's extDict one here.
        assert!(reproduce(&fixture("wrap676k.st1.zst")).unwrap().1.version >= Version::V1_5_5);
    }

    #[test]
    fn cli_frames_carry_a_checksum() {
        // `zstd -1 --single-thread -T1 text3k.raw`: content size and checksum.
        let raw = fixture("text3k.raw");
        let frame = fixture("text3k.cli.zst");
        assert_eq!(frame[4] & 4, 4, "the descriptor's checksum bit");
        check("text3k.cli", &raw, &frame, Build { checksum: true, ..build(Version::V1_5_5, Level::One, Writer::Cli) });
        let (_, found) = reproduce(&frame).unwrap();
        assert_eq!((found.level, found.writer, found.checksum), (Level::One, Writer::Cli, true));
        let mut bad = frame.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert!(decompress(&bad).is_none(), "a wrong checksum is refused");
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
        // Every size class, several blocks, a window smaller than the
        // input, and for the CLI the ring wrapping (past 640 KB at level 1).
        for len in [0, 1, 5, 7, 50, 300, 1000, 5000, 16 << 10, 40_000, 128 << 10, (128 << 10) + 1, 200_000, 300_000, d.len()] {
            let s = &d[..len];
            for build in BUILDS.iter().flat_map(|b| [Build { checksum: false, ..*b }, Build { checksum: true, ..*b }]) {
                let c = compress(s, build);
                assert_eq!(decompress(&c).as_deref(), Some(s), "{len} bytes round-trip with {build:?}");
                let found = reproduce(&c).map(|(_, b)| b).unwrap_or_else(|| panic!("{len} bytes with {build:?} reproduce"));
                assert_eq!(compress(s, found), c, "{len} bytes with {build:?} found as {found:?}");
            }
        }
        let zeros = vec![0u8; 300_000];
        let c = compress(&zeros, BUILDS[0]);
        assert!(c.len() < 40, "zeros: {} bytes", c.len());
        assert_eq!(decompress(&c).unwrap(), zeros);
        assert!(decompress(&c[..c.len() - 1]).is_none(), "a truncated frame is refused");
        assert!(decompress(b"\x28\xb5\x2f\xfd\x20\x05\x01\x00\x00").is_none(), "a frame short of its content size is refused");
    }
}
