pub mod error;
pub mod format;
pub mod fallback;
pub mod x86_decompress;
pub mod x86_compress;
pub mod x86_checksum;
pub mod streaming;
pub mod c_api;

pub use streaming::{AlatirokReader, AlatirokWriter};
pub use format::compute_checksum;

use error::{CodecError, Result};
use format::*;
use rayon::prelude::*;
use std::cell::RefCell;

struct CompressScratch {
    tokens: Vec<Token>,
    offsets: Vec<u16>,
    extras: Vec<u16>,
    literals: Vec<u8>,
    table: [u32; x86_compress::HASH_SIZE],
}

thread_local! {
    static COMPRESS_SCRATCH: RefCell<CompressScratch> = RefCell::new(CompressScratch {
        tokens: Vec::with_capacity(4096),
        offsets: Vec::with_capacity(4096),
        extras: Vec::with_capacity(1024),
        literals: Vec::with_capacity(MAX_BLOCK_SIZE + 64),
        table: [0u32; x86_compress::HASH_SIZE],
    });
}

/// Compress a single 64 KB block into a pre-allocated vector.
pub fn compress_block_into(chunk: &[u8], output: &mut Vec<u8>) {
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
        table.fill(0);

        let checksum = compute_checksum(chunk);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2") {
                unsafe {
                    x86_compress::compress_chained_avx2(
                        chunk,
                        0,
                        chunk.len(),
                        table,
                        tokens,
                        offsets,
                        extras,
                        literals,
                    );
                }
            } else {
                fallback::compress_fallback(chunk, tokens, offsets, extras, literals);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            fallback::compress_fallback(chunk, tokens, offsets, extras, literals);
        }

        let token_bytes_len = tokens.len() * std::mem::size_of::<Token>();
        let offset_bytes_len = offsets.len() * std::mem::size_of::<u16>();
        let extras_bytes_len = extras.len() * std::mem::size_of::<u16>();
        let compressed_payload_len =
            token_bytes_len + offset_bytes_len + extras_bytes_len + literals.len();

        // Incompressibility check: If compressed payload doesn't save at least 2% space,
        // bypass compression entirely and store raw bytes!
        if compressed_payload_len >= chunk.len() - (chunk.len() / 25) {
            let header = BlockHeader {
                magic: MAGIC,
                version: CURRENT_VERSION,
                flags: FLAG_RAW_UNCOMPRESSED | FLAG_CHAIN_RESET,
                checksum,
                uncompressed_len: chunk.len() as u32,
                token_count: 0,
                offset_count: 0,
                extras_count: 0,
                literal_len: chunk.len() as u32,
            };

            let header_slice = unsafe {
                std::slice::from_raw_parts(
                    &header as *const BlockHeader as *const u8,
                    HEADER_SIZE,
                )
            };
            output.extend_from_slice(header_slice);
            output.extend_from_slice(chunk);
            return;
        }

        let header = BlockHeader {
            magic: MAGIC,
            version: CURRENT_VERSION,
            flags: FLAG_COMPRESSED | FLAG_CHAIN_RESET,
            checksum,
            uncompressed_len: chunk.len() as u32,
            token_count: tokens.len() as u32,
            offset_count: offsets.len() as u32,
            extras_count: extras.len() as u32,
            literal_len: literals.len() as u32,
        };

        let header_slice = unsafe {
            std::slice::from_raw_parts(
                &header as *const BlockHeader as *const u8,
                HEADER_SIZE,
            )
        };
        output.extend_from_slice(header_slice);

        // Write tokens
        let token_bytes = unsafe {
            std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                token_bytes_len,
            )
        };
        output.extend_from_slice(token_bytes);

        // Write offsets
        let offset_bytes = unsafe {
            std::slice::from_raw_parts(
                offsets.as_ptr() as *const u8,
                offset_bytes_len,
            )
        };
        output.extend_from_slice(offset_bytes);

        // Write extras
        let extras_bytes = unsafe {
            std::slice::from_raw_parts(extras.as_ptr() as *const u8, extras_bytes_len)
        };
        output.extend_from_slice(extras_bytes);

        // Write literals
        output.extend_from_slice(literals);
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
    let mut table = [0u32; x86_compress::HASH_SIZE];
    let mut tokens = Vec::with_capacity(4096);
    let mut offsets = Vec::with_capacity(4096);
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

        let checksum = compute_checksum(chunk);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2") {
                unsafe {
                    x86_compress::compress_chained_avx2(
                        input,
                        offset,
                        chunk_len,
                        &mut table,
                        &mut tokens,
                        &mut offsets,
                        &mut extras,
                        &mut literals,
                    );
                }
            } else {
                fallback::compress_fallback(chunk, &mut tokens, &mut offsets, &mut extras, &mut literals);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            fallback::compress_fallback(chunk, &mut tokens, &mut offsets, &mut extras, &mut literals);
        }

        let token_bytes_len = tokens.len() * std::mem::size_of::<Token>();
        let offset_bytes_len = offsets.len() * std::mem::size_of::<u16>();
        let extras_bytes_len = extras.len() * std::mem::size_of::<u16>();
        let compressed_payload_len =
            token_bytes_len + offset_bytes_len + extras_bytes_len + literals.len();

        let chain_flag = if offset == 0 { FLAG_CHAIN_RESET } else { 0 };

        if compressed_payload_len >= chunk_len - (chunk_len / 25) {
            let header = BlockHeader {
                magic: MAGIC,
                version: CURRENT_VERSION,
                flags: FLAG_RAW_UNCOMPRESSED | chain_flag,
                checksum,
                uncompressed_len: chunk_len as u32,
                token_count: 0,
                offset_count: 0,
                extras_count: 0,
                literal_len: chunk_len as u32,
            };

            let header_slice = unsafe {
                std::slice::from_raw_parts(
                    &header as *const BlockHeader as *const u8,
                    HEADER_SIZE,
                )
            };
            output.extend_from_slice(header_slice);
            output.extend_from_slice(chunk);
            offset += chunk_len;
            continue;
        }

        let header = BlockHeader {
            magic: MAGIC,
            version: CURRENT_VERSION,
            flags: FLAG_COMPRESSED | chain_flag,
            checksum,
            uncompressed_len: chunk_len as u32,
            token_count: tokens.len() as u32,
            offset_count: offsets.len() as u32,
            extras_count: extras.len() as u32,
            literal_len: literals.len() as u32,
        };

        let header_slice = unsafe {
            std::slice::from_raw_parts(
                &header as *const BlockHeader as *const u8,
                HEADER_SIZE,
            )
        };
        output.extend_from_slice(header_slice);

        let token_bytes = unsafe {
            std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                token_bytes_len,
            )
        };
        output.extend_from_slice(token_bytes);

        let offset_bytes = unsafe {
            std::slice::from_raw_parts(
                offsets.as_ptr() as *const u8,
                offset_bytes_len,
            )
        };
        output.extend_from_slice(offset_bytes);

        let extras_bytes = unsafe {
            std::slice::from_raw_parts(extras.as_ptr() as *const u8, extras_bytes_len)
        };
        output.extend_from_slice(extras_bytes);

        output.extend_from_slice(&literals);
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

/// Decompress an entire SIMD-stream payload sequentially into a freshly allocated vector.
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut total_uncompressed_len = 0usize;
    let mut cursor = 0usize;

    while cursor + HEADER_SIZE <= compressed.len() {
        let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        if header.version > CURRENT_VERSION {
            return Err(CodecError::UnsupportedVersion(header.version));
        }
        total_uncompressed_len += header.uncompressed_len as usize;

        let block_payload_len = if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            header.uncompressed_len as usize
        } else {
            payload_len(
                header.token_count as usize,
                header.offset_count as usize,
                header.extras_count as usize,
                header.literal_len as usize,
            )
        };

        cursor += HEADER_SIZE + block_payload_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block header"));
    }

    let mut output = vec![0u8; total_uncompressed_len + PADDING * 2];
    let written = decompress_into(compressed, &mut output)?;
    output.truncate(written);
    Ok(output)
}

/// Decompress into a pre-allocated buffer sequentially with checksum validation.
pub fn decompress_into(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    let mut cursor = 0usize;
    let mut dst_offset = 0usize;
    let buffer_start = dst.as_ptr();

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2");

    while cursor + HEADER_SIZE <= compressed.len() {
        let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        if header.version > CURRENT_VERSION {
            return Err(CodecError::UnsupportedVersion(header.version));
        }
        cursor += HEADER_SIZE;

        let uncomp_len = header.uncompressed_len as usize;
        let expected_checksum = header.checksum;

        if dst_offset + uncomp_len > dst.len() {
            return Err(CodecError::OutputBufferTooSmall {
                required: dst_offset + uncomp_len,
                provided: dst.len(),
            });
        }

        // Raw uncompressed bypass path
        if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            if cursor + uncomp_len > compressed.len() {
                return Err(CodecError::CorruptedBitstream("Unexpected end in raw block"));
            }
            dst[dst_offset..dst_offset + uncomp_len]
                .copy_from_slice(&compressed[cursor..cursor + uncomp_len]);

            let actual_checksum = compute_checksum(&dst[dst_offset..dst_offset + uncomp_len]);
            if actual_checksum != expected_checksum {
                return Err(CodecError::ChecksumMismatch {
                    expected: expected_checksum,
                    computed: actual_checksum,
                });
            }

            cursor += uncomp_len;
            dst_offset += uncomp_len;
            continue;
        }

        let token_count = header.token_count as usize;
        let offset_count = header.offset_count as usize;
        let extras_count = header.extras_count as usize;
        let lit_len = header.literal_len as usize;

        let token_bytes_len = token_count * std::mem::size_of::<Token>();
        let offset_bytes_len = offset_count * std::mem::size_of::<u16>();
        let extras_bytes_len = extras_count * std::mem::size_of::<u16>();

        if cursor + token_bytes_len + offset_bytes_len + extras_bytes_len + lit_len
            > compressed.len()
        {
            return Err(CodecError::CorruptedBitstream("Truncated compressed block payload"));
        }

        let tokens_ptr = unsafe { compressed.as_ptr().add(cursor) as *const Token };
        cursor += token_bytes_len;

        let offsets_ptr = unsafe { compressed.as_ptr().add(cursor) as *const u16 };
        cursor += offset_bytes_len;

        let extras_ptr = unsafe { compressed.as_ptr().add(cursor) as *const u16 };
        cursor += extras_bytes_len;

        let raw_literals = &compressed[cursor..cursor + lit_len];
        cursor += lit_len;

        let dst_slice = &mut dst[dst_offset..];

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2 {
                unsafe {
                    x86_decompress::decompress_avx2(
                        tokens_ptr,
                        token_count,
                        offsets_ptr,
                        offset_count,
                        extras_ptr,
                        extras_count,
                        raw_literals,
                        dst_slice,
                        buffer_start,
                        uncomp_len,
                    )?;
                }
            } else {
                unsafe {
                    fallback::decompress_fallback_raw(
                        tokens_ptr,
                        token_count,
                        offsets_ptr,
                        offset_count,
                        extras_ptr,
                        extras_count,
                        raw_literals,
                        dst,
                        dst_offset,
                        uncomp_len,
                    )?;
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unsafe {
                fallback::decompress_fallback_raw(
                    tokens_ptr,
                    token_count,
                    offsets_ptr,
                    offset_count,
                    raw_literals,
                    dst,
                    dst_offset,
                    uncomp_len,
                )?;
            }
        }

        let actual_checksum = compute_checksum(&dst[dst_offset..dst_offset + uncomp_len]);
        if actual_checksum != expected_checksum {
            return Err(CodecError::ChecksumMismatch {
                expected: expected_checksum,
                computed: actual_checksum,
            });
        }

        dst_offset += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }

    Ok(dst_offset)
}

/// Decompress into pre-allocated buffer without verifying checksum (raw codec speed).
pub fn decompress_into_raw(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    let mut cursor = 0usize;
    let mut dst_offset = 0usize;
    let buffer_start = dst.as_ptr();

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2");

    while cursor + HEADER_SIZE <= compressed.len() {
        let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        cursor += HEADER_SIZE;

        let uncomp_len = header.uncompressed_len as usize;

        if dst_offset + uncomp_len > dst.len() {
            return Err(CodecError::OutputBufferTooSmall {
                required: dst_offset + uncomp_len,
                provided: dst.len(),
            });
        }

        if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            if cursor + uncomp_len > compressed.len() {
                return Err(CodecError::CorruptedBitstream("Unexpected end in raw block"));
            }
            dst[dst_offset..dst_offset + uncomp_len]
                .copy_from_slice(&compressed[cursor..cursor + uncomp_len]);
            cursor += uncomp_len;
            dst_offset += uncomp_len;
            continue;
        }

        let token_count = header.token_count as usize;
        let offset_count = header.offset_count as usize;
        let extras_count = header.extras_count as usize;
        let lit_len = header.literal_len as usize;

        let token_bytes_len = token_count * std::mem::size_of::<Token>();
        let offset_bytes_len = offset_count * std::mem::size_of::<u16>();
        let extras_bytes_len = extras_count * std::mem::size_of::<u16>();

        if cursor + token_bytes_len + offset_bytes_len + extras_bytes_len + lit_len
            > compressed.len()
        {
            return Err(CodecError::CorruptedBitstream("Truncated compressed block payload"));
        }

        let tokens_ptr = unsafe { compressed.as_ptr().add(cursor) as *const Token };
        cursor += token_bytes_len;

        let offsets_ptr = unsafe { compressed.as_ptr().add(cursor) as *const u16 };
        cursor += offset_bytes_len;

        let extras_ptr = unsafe { compressed.as_ptr().add(cursor) as *const u16 };
        cursor += extras_bytes_len;

        let raw_literals = &compressed[cursor..cursor + lit_len];
        cursor += lit_len;

        let dst_slice = &mut dst[dst_offset..];

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2 {
                unsafe {
                    x86_decompress::decompress_avx2(
                        tokens_ptr,
                        token_count,
                        offsets_ptr,
                        offset_count,
                        extras_ptr,
                        extras_count,
                        raw_literals,
                        dst_slice,
                        buffer_start,
                        uncomp_len,
                    )?;
                }
            } else {
                unsafe {
                    fallback::decompress_fallback_raw(
                        tokens_ptr,
                        token_count,
                        offsets_ptr,
                        offset_count,
                        extras_ptr,
                        extras_count,
                        raw_literals,
                        dst,
                        dst_offset,
                        uncomp_len,
                    )?;
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unsafe {
                fallback::decompress_fallback_raw(
                    tokens_ptr,
                    token_count,
                    offsets_ptr,
                    offset_count,
                    raw_literals,
                    dst,
                    dst_offset,
                    uncomp_len,
                )?;
            }
        }

        dst_offset += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }

    Ok(dst_offset)
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
    let mut total_uncomp = 0usize;
    let mut cursor = 0usize;

    while cursor + HEADER_SIZE <= compressed.len() {
        let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        if header.version > CURRENT_VERSION {
            return Err(CodecError::UnsupportedVersion(header.version));
        }
        let uncomp_len = header.uncompressed_len as usize;
        let payload_len = if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            uncomp_len
        } else {
            payload_len(
                header.token_count as usize,
                header.offset_count as usize,
                header.extras_count as usize,
                header.literal_len as usize,
            )
        };
        cursor += HEADER_SIZE + payload_len;
        total_uncomp += uncomp_len;
    }

    let mut output = vec![0u8; total_uncomp + PADDING * 2];
    let written = decompress_parallel_into(compressed, &mut output)?;
    output.truncate(written);
    Ok(output)
}

/// Decompress in parallel across all CPU cores into a pre-allocated buffer with checksum verification.
pub fn decompress_parallel_into(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    let mut blocks = Vec::new();
    let mut units: Vec<ParallelUnit> = Vec::new();
    let mut cursor = 0usize;
    let mut total_uncomp = 0usize;

    while cursor + HEADER_SIZE <= compressed.len() {
        let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        if header.version > CURRENT_VERSION {
            return Err(CodecError::UnsupportedVersion(header.version));
        }
        let uncomp_len = header.uncompressed_len as usize;

        let payload_len = if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            uncomp_len
        } else {
            payload_len(
                header.token_count as usize,
                header.offset_count as usize,
                header.extras_count as usize,
                header.literal_len as usize,
            )
        };

        let total_block_len = HEADER_SIZE + payload_len;
        let block_idx = blocks.len();
        blocks.push(BlockInfo {
            block_offset: cursor,
            block_size: total_block_len,
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

        cursor += total_block_len;
        total_uncomp += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }

    if dst.len() < total_uncomp {
        return Err(CodecError::OutputBufferTooSmall {
            required: total_uncomp,
            provided: dst.len(),
        });
    }

    if units.len() <= 1 {
        return decompress_into(compressed, dst);
    }

    let output_ptr = dst.as_mut_ptr() as usize;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2");

    units.par_iter().try_for_each(|unit| -> Result<()> {
        let unit_slice = unsafe {
            let ptr = (output_ptr + unit.uncomp_offset) as *mut u8;
            std::slice::from_raw_parts_mut(ptr, unit.uncomp_len + PADDING)
        };
        let unit_buffer_start = unit_slice.as_ptr();

        for i in 0..unit.block_count {
            let b = &blocks[unit.first_block_idx + i];
            let block_slice = &compressed[b.block_offset..b.block_offset + b.block_size];
            let header = unsafe { std::ptr::read_unaligned(block_slice.as_ptr() as *const BlockHeader) };

            let block_offset_in_unit = b.uncomp_offset - unit.uncomp_offset;
            let dst_slice = unsafe {
                let ptr = (output_ptr + b.uncomp_offset) as *mut u8;
                std::slice::from_raw_parts_mut(ptr, b.uncomp_len + PADDING)
            };

            if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        block_slice.as_ptr().add(HEADER_SIZE),
                        dst_slice.as_mut_ptr(),
                        b.uncomp_len,
                    );
                }
            } else {
                let token_count = header.token_count as usize;
                let offset_count = header.offset_count as usize;
                let extras_count = header.extras_count as usize;
                let lit_len = header.literal_len as usize;

                let mut c = HEADER_SIZE;
                let token_bytes_len = token_count * std::mem::size_of::<Token>();
                let offset_bytes_len = offset_count * std::mem::size_of::<u16>();
                let extras_bytes_len = extras_count * std::mem::size_of::<u16>();

                let tokens_ptr = unsafe { block_slice.as_ptr().add(c) as *const Token };
                c += token_bytes_len;

                let offsets_ptr = unsafe { block_slice.as_ptr().add(c) as *const u16 };
                c += offset_bytes_len;

                let extras_ptr = unsafe { block_slice.as_ptr().add(c) as *const u16 };
                c += extras_bytes_len;

                let raw_literals = &block_slice[c..c + lit_len];

                #[cfg(target_arch = "x86_64")]
                {
                    unsafe {
                    if has_avx2 {
                            x86_decompress::decompress_avx2(tokens_ptr, token_count, offsets_ptr, offset_count, extras_ptr, extras_count, raw_literals, dst_slice, unit_buffer_start, b.uncomp_len)?;
                        } else {
                            fallback::decompress_fallback_raw(tokens_ptr, token_count, offsets_ptr, offset_count, extras_ptr, extras_count, raw_literals, unit_slice, block_offset_in_unit, b.uncomp_len)?;
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    unsafe {
                        fallback::decompress_fallback_raw(tokens_ptr, token_count, offsets_ptr, offset_count, extras_ptr, extras_count, raw_literals, unit_slice, block_offset_in_unit, b.uncomp_len)?;
                    }
                }
            }

            let actual = compute_checksum(&dst_slice[..b.uncomp_len]);
            if actual != header.checksum {
                return Err(CodecError::ChecksumMismatch {
                    expected: header.checksum,
                    computed: actual,
                });
            }
        }

        Ok(())
    })?;

    Ok(total_uncomp)
}

/// Decompress in parallel across all CPU cores into a pre-allocated buffer without verifying checksum (raw codec speed).
pub fn decompress_parallel_into_raw(compressed: &[u8], dst: &mut [u8]) -> Result<usize> {
    let mut blocks = Vec::new();
    let mut units: Vec<ParallelUnit> = Vec::new();
    let mut cursor = 0usize;
    let mut total_uncomp = 0usize;

    while cursor + HEADER_SIZE <= compressed.len() {
        let header = unsafe { std::ptr::read_unaligned(compressed.as_ptr().add(cursor) as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(CodecError::InvalidMagic);
        }
        let uncomp_len = header.uncompressed_len as usize;

        let payload_len = if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            uncomp_len
        } else {
            payload_len(
                header.token_count as usize,
                header.offset_count as usize,
                header.extras_count as usize,
                header.literal_len as usize,
            )
        };

        let total_block_len = HEADER_SIZE + payload_len;
        let block_idx = blocks.len();
        blocks.push(BlockInfo {
            block_offset: cursor,
            block_size: total_block_len,
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

        cursor += total_block_len;
        total_uncomp += uncomp_len;
    }

    if cursor != compressed.len() {
        return Err(CodecError::CorruptedBitstream("Trailing unparsed bytes or truncated block"));
    }

    if dst.len() < total_uncomp {
        return Err(CodecError::OutputBufferTooSmall {
            required: total_uncomp,
            provided: dst.len(),
        });
    }

    if units.len() <= 1 {
        return decompress_into_raw(compressed, dst);
    }

    let output_ptr = dst.as_mut_ptr() as usize;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2");

    units.par_iter().try_for_each(|unit| -> Result<()> {
        let unit_slice = unsafe {
            let ptr = (output_ptr + unit.uncomp_offset) as *mut u8;
            std::slice::from_raw_parts_mut(ptr, unit.uncomp_len + PADDING)
        };
        let unit_buffer_start = unit_slice.as_ptr();

        for i in 0..unit.block_count {
            let b = &blocks[unit.first_block_idx + i];
            let block_slice = &compressed[b.block_offset..b.block_offset + b.block_size];
            let header = unsafe { std::ptr::read_unaligned(block_slice.as_ptr() as *const BlockHeader) };

            let block_offset_in_unit = b.uncomp_offset - unit.uncomp_offset;
            let dst_slice = unsafe {
                let ptr = (output_ptr + b.uncomp_offset) as *mut u8;
                std::slice::from_raw_parts_mut(ptr, b.uncomp_len + PADDING)
            };

            if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        block_slice.as_ptr().add(HEADER_SIZE),
                        dst_slice.as_mut_ptr(),
                        b.uncomp_len,
                    );
                }
            } else {
                let token_count = header.token_count as usize;
                let offset_count = header.offset_count as usize;
                let extras_count = header.extras_count as usize;
                let lit_len = header.literal_len as usize;

                let mut c = HEADER_SIZE;
                let token_bytes_len = token_count * std::mem::size_of::<Token>();
                let offset_bytes_len = offset_count * std::mem::size_of::<u16>();
                let extras_bytes_len = extras_count * std::mem::size_of::<u16>();

                let tokens_ptr = unsafe { block_slice.as_ptr().add(c) as *const Token };
                c += token_bytes_len;

                let offsets_ptr = unsafe { block_slice.as_ptr().add(c) as *const u16 };
                c += offset_bytes_len;

                let extras_ptr = unsafe { block_slice.as_ptr().add(c) as *const u16 };
                c += extras_bytes_len;

                let raw_literals = &block_slice[c..c + lit_len];

                #[cfg(target_arch = "x86_64")]
                {
                    unsafe {
                    if has_avx2 {
                            x86_decompress::decompress_avx2(tokens_ptr, token_count, offsets_ptr, offset_count, extras_ptr, extras_count, raw_literals, dst_slice, unit_buffer_start, b.uncomp_len)?;
                        } else {
                            fallback::decompress_fallback_raw(tokens_ptr, token_count, offsets_ptr, offset_count, extras_ptr, extras_count, raw_literals, unit_slice, block_offset_in_unit, b.uncomp_len)?;
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    unsafe {
                        fallback::decompress_fallback_raw(tokens_ptr, token_count, offsets_ptr, offset_count, extras_ptr, extras_count, raw_literals, unit_slice, block_offset_in_unit, b.uncomp_len)?;
                    }
                }
            }
        }

        Ok(())
    })?;

    Ok(total_uncomp)
}
