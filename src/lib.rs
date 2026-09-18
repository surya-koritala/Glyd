pub mod error;
pub mod format;
pub mod huffman;
pub mod finder;
pub mod fallback;
#[cfg(target_arch = "x86_64")]
pub mod x86_decompress;
#[cfg(target_arch = "x86_64")]
pub mod x86_compress;
#[cfg(target_arch = "x86_64")]
pub mod x86_checksum;
#[cfg(target_arch = "aarch64")]
pub mod neon_decompress;
pub mod streaming;
pub mod c_api;

pub use streaming::{AlatirokReader, AlatirokWriter};
pub use format::compute_checksum;

use error::{CodecError, Result};
use finder::{new_table, Dense, HashTable, Lzav, Mode};
use format::*;
use rayon::prelude::*;
use std::cell::RefCell;

struct CompressScratch {
    tokens: Vec<u8>,
    offsets: Vec<u8>,
    extras: Vec<u8>,
    literals: Vec<u8>,
    table: Box<HashTable>,
}

thread_local! {
    static COMPRESS_SCRATCH: RefCell<CompressScratch> = RefCell::new(CompressScratch {
        tokens: Vec::with_capacity(4096),
        offsets: Vec::with_capacity(8192),
        extras: Vec::with_capacity(1024),
        literals: Vec::with_capacity(MAX_BLOCK_SIZE + 64),
        table: new_table(),
    });
}

#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Run the match finder over one block of `full_input` with parse `P`.
#[inline(always)]
fn find_block<P: Mode>(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86_compress::compress_chained_avx2::<P>(
                    full_input, block_start, block_len, table, tokens, offsets, extras, literals,
                );
            }
            return;
        }
    }
    fallback::compress_chained_fallback::<P>(
        full_input, block_start, block_len, table, tokens, offsets, extras, literals,
    );
}

/// The dense retry is opt-in (ALATIROK_DENSE=1). GOAL3 section 3: it bought
/// x-ray 1.08 for 23 ms of compression and 2.7 ms of decode, which the speed
/// target cannot afford; liblz4 gets 1.01 there at memcpy speed. The fast
/// level (GOAL3 Tier S3) is the intended replacement. Cached, never per block.
fn dense_enabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("ALATIROK_DENSE").is_ok())
}

/// Streams would not save at least 4% of the chunk.
#[inline(always)]
fn not_worth_it(chunk_len: usize, tokens: &[u8], offsets: &[u8], extras: &[u8], literals: &[u8]) -> bool {
    tokens.len() + offsets.len() + extras.len() + literals.len() >= chunk_len - (chunk_len / 25)
}

/// Parse one block: the ordinary parse first, then the dense retry when it
/// would have been stored raw. Returns the flag bits describing the result.
#[inline(always)]
fn parse_block(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) -> u16 {
    find_block::<Lzav>(full_input, block_start, block_len, table, tokens, offsets, extras, literals);
    if !not_worth_it(block_len, tokens, offsets, extras, literals) {
        return FLAG_COMPRESSED;
    }
    if !dense_enabled() {
        // Keep the ordinary parse whenever it beats a raw store at all.
        let payload = tokens.len() + offsets.len() + extras.len() + literals.len();
        return if payload < block_len { FLAG_COMPRESSED } else { FLAG_RAW_UNCOMPRESSED };
    }
    tokens.clear();
    offsets.clear();
    extras.clear();
    literals.clear();
    find_block::<Dense>(full_input, block_start, block_len, table, tokens, offsets, extras, literals);
    if not_worth_it(block_len, tokens, offsets, extras, literals) {
        return FLAG_RAW_UNCOMPRESSED;
    }
    FLAG_DENSE
}

/// Append one block (header plus payload) for `chunk`, given its streams and
/// the parse result flags (raw, ordinary or dense).
fn write_block(
    chunk: &[u8],
    parse_flags: u16,
    chain_flag: u16,
    tokens: &[u8],
    offsets: &[u8],
    extras: &[u8],
    literals: &[u8],
    output: &mut Vec<u8>,
) {
    let checksum = compute_checksum(chunk);

    if (parse_flags & FLAG_RAW_UNCOMPRESSED) != 0 {
        let header = BlockHeader {
            magic: MAGIC,
            version: CURRENT_VERSION,
            flags: FLAG_RAW_UNCOMPRESSED | chain_flag,
            checksum,
            uncompressed_len: chunk.len() as u32,
            token_count: 0,
            token_bytes: 0,
            offset_bytes: 0,
            extras_bytes: 0,
            literal_len: chunk.len() as u32,
        };
        output.extend_from_slice(header_bytes(&header));
        output.extend_from_slice(chunk);
        return;
    }

    let header = BlockHeader {
        magic: MAGIC,
        version: CURRENT_VERSION,
        flags: parse_flags | chain_flag,
        checksum,
        uncompressed_len: chunk.len() as u32,
        token_count: tokens.len() as u32,
        token_bytes: tokens.len() as u32,
        offset_bytes: offsets.len() as u32,
        extras_bytes: extras.len() as u32,
        literal_len: literals.len() as u32,
    };
    output.extend_from_slice(header_bytes(&header));
    output.extend_from_slice(tokens);
    output.extend_from_slice(offsets);
    output.extend_from_slice(extras);
    output.extend_from_slice(literals);
}

#[inline(always)]
fn header_bytes(header: &BlockHeader) -> &[u8] {
    unsafe { std::slice::from_raw_parts(header as *const BlockHeader as *const u8, HEADER_SIZE) }
}

/// Compress a single block (at most MAX_BLOCK_SIZE bytes) with no history
/// into a pre-allocated vector.
pub fn compress_block_into(chunk: &[u8], output: &mut Vec<u8>) {
    assert!(chunk.len() <= MAX_BLOCK_SIZE, "block exceeds MAX_BLOCK_SIZE");
    COMPRESS_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        let CompressScratch {
            ref mut tokens,
            ref mut offsets,
            ref mut extras,
            ref mut literals,
            ref mut table,
        } = *scratch;

        tokens.clear();
        offsets.clear();
        extras.clear();
        literals.clear();
        finder::init_table(table, chunk);
        let flags = parse_block(chunk, 0, chunk.len(), table, tokens, offsets, extras, literals);
        write_block(chunk, flags, FLAG_CHAIN_RESET, tokens, offsets, extras, literals, output);
    });
}

/// Compress an arbitrary byte slice sequentially using a single core.
pub fn compress(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() / 2 + 1024);
    compress_into(input, &mut output);
    output
}

/// Compress an input slice into a destination vector with cross-block history lookback.
pub fn compress_into(input: &[u8], output: &mut Vec<u8>) {
    let mut table = new_table();
    finder::init_table(&mut table, input);
    let mut tokens = Vec::with_capacity(4096);
    let mut offsets = Vec::with_capacity(8192);
    let mut extras = Vec::with_capacity(1024);
    let mut literals = Vec::with_capacity(MAX_BLOCK_SIZE);

    let mut offset = 0;
    while offset < input.len() {
        let chunk_len = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let chunk = &input[offset..offset + chunk_len];

        tokens.clear();
        offsets.clear();
        extras.clear();
        literals.clear();

        let flags = parse_block(input, offset, chunk_len, &mut table, &mut tokens, &mut offsets, &mut extras, &mut literals);

        let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };
        write_block(chunk, flags, chain_flag, &tokens, &offsets, &extras, &literals, output);
        offset += chunk_len;
    }
}

/// Compress across all CPU cores in parallel into a pre-allocated destination vector.
pub fn compress_parallel_into(input: &[u8], output: &mut Vec<u8>) {
    if input.len() <= PARALLEL_CHUNK_SIZE {
        compress_into(input, output);
        return;
    }

    let chunks: Vec<&[u8]> = input.chunks(PARALLEL_CHUNK_SIZE).collect();
    let compressed_chunks: Vec<Vec<u8>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut chunk_out = Vec::with_capacity(chunk.len() / 2 + 1024);
            compress_into(chunk, &mut chunk_out);
            chunk_out
        })
        .collect();

    let total_len: usize = compressed_chunks.iter().map(|c| c.len()).sum();
    output.reserve(total_len);
    for chunk in compressed_chunks {
        output.extend_from_slice(&chunk);
    }
}

/// Compress across all CPU cores in parallel using Rayon.
pub fn compress_parallel(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() / 2 + 1024);
    compress_parallel_into(input, &mut output);
    output
}

/// Read and validate the block header at `cursor`. Returns the header and
/// the cursor of the next block. Nothing in the payload is touched.
#[inline(always)]
fn parse_header(compressed: &[u8], cursor: usize) -> Result<(BlockHeader, usize)> {
    let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
    if header.magic != MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    if header.version != CURRENT_VERSION {
        return Err(CodecError::UnsupportedVersion(header.version));
    }
    if !header.is_plausible() {
        return Err(CodecError::CorruptedBitstream("Implausible block header"));
    }
    let next = cursor + HEADER_SIZE + header.payload_len();
    if next > compressed.len() {
        return Err(CodecError::CorruptedBitstream("Truncated compressed block payload"));
    }
    Ok((header, next))
}

/// Decode one block's payload into `dst` (the block's output region plus
/// padding), with the match window starting at `buffer_start`.
#[inline(always)]
unsafe fn decode_block(
    header: &BlockHeader,
    payload: &[u8],
    dst: &mut [u8],
    buffer_start: *const u8,
    avx2: bool,
) -> Result<()> {
    let uncomp_len = header.uncompressed_len as usize;
    if uncomp_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall { required: uncomp_len, provided: dst.len() });
    }
    if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
        dst[..uncomp_len].copy_from_slice(&payload[..uncomp_len]);
        return Ok(());
    }
    if (header.flags & FLAG_HUFF_TOKENS) != 0 {
        return Err(CodecError::CorruptedBitstream("Huffman token blocks not supported yet"));
    }
    let token_count = header.token_count as usize;
    let token_bytes = header.token_bytes as usize;
    if token_bytes != token_count {
        return Err(CodecError::CorruptedBitstream("Token section length mismatch"));
    }
    let mut c = 0usize;
    let tokens = payload.as_ptr().add(c);
    c += token_bytes;
    let offsets = payload.as_ptr().add(c);
    let offsets_len = header.offset_bytes as usize;
    c += offsets_len;
    let extras = payload.as_ptr().add(c);
    let extras_len = header.extras_bytes as usize;
    c += extras_len;
    let literals = &payload[c..c + header.literal_len as usize];
    let dense = (header.flags & FLAG_DENSE) != 0;
    let min_match = if dense { MIN_MATCH_LEN_DENSE } else { MIN_MATCH_LEN };

    #[cfg(target_arch = "x86_64")]
    {
        if avx2 {
            let (table, esc) = if dense {
                (&TOKEN_TABLE_DENSE, ESCAPE_BASE_MATCH_DENSE)
            } else {
                (&TOKEN_TABLE, ESCAPE_BASE_MATCH)
            };
            x86_decompress::decompress_avx2(
                tokens, token_count, offsets, offsets_len, extras, extras_len,
                literals, dst, buffer_start, uncomp_len, table, esc,
            )?;
            return Ok(());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let (table, esc) = if dense {
            (&TOKEN_TABLE_DENSE, ESCAPE_BASE_MATCH_DENSE)
        } else {
            (&TOKEN_TABLE, ESCAPE_BASE_MATCH)
        };
        neon_decompress::decompress_neon(
            tokens, token_count, offsets, offsets_len, extras, extras_len,
            payload.len() - (c - extras_len), literals, dst, buffer_start, uncomp_len, table, esc,
        )?;
        return Ok(());
    }
    #[allow(unreachable_code)]
    let _ = avx2;
    fallback::decompress_fallback_raw(
        tokens, token_count, offsets, offsets_len, extras, extras_len,
        literals, dst, buffer_start, uncomp_len, min_match,
    )?;
    Ok(())
}

/// Walk every header, validating framing, and return the total output size.
fn total_uncompressed_len(compressed: &[u8]) -> Result<usize> {
    let mut total = 0usize;
    let mut cursor = 0usize;
    while cursor + HEADER_SIZE <= compressed.len() {
        let (header, next) = parse_header(compressed, cursor)?;
        total += header.uncompressed_len as usize;
        cursor = next;
    }
    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block header"));
    }
    Ok(total)
}

/// Decompress an entire SIMD-stream payload sequentially into a freshly allocated vector.
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
    let total = total_uncompressed_len(compressed)?;
    let mut output = vec![0u8; total + PADDING * 2];
    let written = decompress_into(compressed, &mut output)?;
    output.truncate(written);
    Ok(output)
}

fn decompress_sequential(compressed: &[u8], dst: &mut [u8], verify: bool) -> Result<usize> {
    let mut cursor = 0usize;
    let mut dst_offset = 0usize;
    let buffer_start = dst.as_ptr();
    let avx2 = has_avx2();

    while cursor + HEADER_SIZE <= compressed.len() {
        let (header, next) = parse_header(compressed, cursor)?;
        let uncomp_len = header.uncompressed_len as usize;
        if dst_offset + uncomp_len > dst.len() {
            return Err(CodecError::OutputBufferTooSmall {
                required: dst_offset + uncomp_len,
                provided: dst.len(),
            });
        }
        let payload = &compressed[cursor + HEADER_SIZE..next];
        let dst_slice = &mut dst[dst_offset..];
        unsafe {
            decode_block(&header, payload, dst_slice, buffer_start, avx2)?;
        }
        if verify {
            let actual = compute_checksum(&dst[dst_offset..dst_offset + uncomp_len]);
            if actual != header.checksum {
                return Err(CodecError::ChecksumMismatch { expected: header.checksum, computed: actual });
            }
        }
        cursor = next;
        dst_offset += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }
    Ok(dst_offset)
}

/// Decompress into a pre-allocated buffer sequentially with checksum validation.
pub fn decompress_into(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    decompress_sequential(compressed, dst, true)
}

/// Decompress into pre-allocated buffer without verifying checksum (raw codec speed).
pub fn decompress_into_raw(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    decompress_sequential(compressed, dst, false)
}

struct BlockInfo {
    block_offset: usize,
    block_size: usize,
    uncomp_offset: usize,
    uncomp_len: usize,
}

struct ParallelUnit {
    first_block_idx: usize,
    block_count: usize,
    uncomp_offset: usize,
    uncomp_len: usize,
}

/// Decompress in parallel across all CPU cores into a freshly allocated vector.
pub fn decompress_parallel(compressed: &[u8]) -> Result<Vec<u8>> {
    let total = total_uncompressed_len(compressed)?;
    let mut output = vec![0u8; total + PADDING * 2];
    let written = decompress_parallel_into(compressed, &mut output)?;
    output.truncate(written);
    Ok(output)
}

fn decompress_parallel_impl(compressed: &[u8], dst: &mut [u8], verify: bool) -> Result<usize> {
    let mut blocks = Vec::new();
    let mut units: Vec<ParallelUnit> = Vec::new();
    let mut cursor = 0usize;
    let mut total_uncomp = 0usize;

    while cursor + HEADER_SIZE <= compressed.len() {
        let (header, next) = parse_header(compressed, cursor)?;
        let uncomp_len = header.uncompressed_len as usize;
        let block_idx = blocks.len();
        blocks.push(BlockInfo {
            block_offset: cursor,
            block_size: next - cursor,
            uncomp_offset: total_uncomp,
            uncomp_len,
        });
        if (header.flags & FLAG_CHAIN_RESET) != 0 || units.is_empty() {
            units.push(ParallelUnit {
                first_block_idx: block_idx,
                block_count: 1,
                uncomp_offset: total_uncomp,
                uncomp_len,
            });
        } else {
            let u = units.last_mut().unwrap();
            u.block_count += 1;
            u.uncomp_len += uncomp_len;
        }
        cursor = next;
        total_uncomp += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }
    if dst.len() < total_uncomp {
        return Err(CodecError::OutputBufferTooSmall { required: total_uncomp, provided: dst.len() });
    }
    if units.len() <= 1 {
        return decompress_sequential(compressed, dst, verify);
    }

    let output_ptr = dst.as_mut_ptr() as usize;
    let avx2 = has_avx2();

    units.par_iter().try_for_each(|unit| -> Result<()> {
        let unit_buffer_start = (output_ptr + unit.uncomp_offset) as *const u8;
        for i in 0..unit.block_count {
            let b = &blocks[unit.first_block_idx + i];
            let block_slice = &compressed[b.block_offset..b.block_offset + b.block_size];
            let header = unsafe { std::ptr::read_unaligned(block_slice.as_ptr() as *const BlockHeader) };
            let payload = &block_slice[HEADER_SIZE..];
            // Blocks cover disjoint output ranges, so these slices never alias.
            let dst_slice = unsafe {
                let ptr = (output_ptr + b.uncomp_offset) as *mut u8;
                std::slice::from_raw_parts_mut(ptr, b.uncomp_len)
            };
            unsafe {
                decode_block(&header, payload, dst_slice, unit_buffer_start, avx2)?;
            }
            if verify {
                let actual = compute_checksum(&dst_slice[..b.uncomp_len]);
                if actual != header.checksum {
                    return Err(CodecError::ChecksumMismatch { expected: header.checksum, computed: actual });
                }
            }
        }
        Ok(())
    })?;

    Ok(total_uncomp)
}

/// Decompress in parallel across all CPU cores into a pre-allocated buffer with checksum verification.
pub fn decompress_parallel_into(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    decompress_parallel_impl(compressed, dst, true)
}

/// Decompress in parallel across all CPU cores into a pre-allocated buffer without verifying checksum (raw codec speed).
pub fn decompress_parallel_into_raw(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    decompress_parallel_impl(compressed, dst, false)
}
