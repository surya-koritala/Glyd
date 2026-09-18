use crate::error::{CodecError, Result};
use crate::finder::{find_matches, init_table, HashTable, Mode, ScalarMatch, Streams};
use crate::format::{
    Token, ESCAPE_BASE_LIT, ESCAPE_CONT, LIT_CODE_ESCAPE, MATCH_CODE_ESCAPE,
};

/// Read one escaped length from the extras stream.
#[inline(always)]
pub unsafe fn read_escape(
    extras: *const u8,
    extras_len: usize,
    idx: &mut usize,
    base: usize,
) -> Result<usize> {
    if *idx >= extras_len {
        return Err(CodecError::CorruptedBitstream("Insufficient extras in bitstream"));
    }
    let v = *extras.add(*idx);
    *idx += 1;
    if v != ESCAPE_CONT {
        return Ok(base + v as usize);
    }
    if *idx + 2 > extras_len {
        return Err(CodecError::CorruptedBitstream("Insufficient extras in bitstream"));
    }
    let w = u16::from_le_bytes([*extras.add(*idx), *extras.add(*idx + 1)]) as usize;
    *idx += 2;
    Ok(base + 255 + w)
}

/// Universal portable decompressor. Every access is bounds-checked, so it is
/// also the reference for the SIMD decoder's tail phase.
///
/// `dst` is the block's own output region (plus padding); `buffer_start` is
/// where the match window begins, at or before `dst`. `min_match` selects
/// the length bias (6 for ordinary blocks, 4 for FLAG_DENSE).
pub unsafe fn decompress_fallback_raw(
    tokens: *const u8,
    num_tokens: usize,
    offsets: *const u8,
    offsets_len: usize,
    extras: *const u8,
    extras_len: usize,
    literals: &[u8],
    dst: &mut [u8],
    buffer_start: *const u8,
    uncompressed_len: usize,
    min_match: usize,
) -> Result<usize> {
    let bias = min_match - 1;
    if uncompressed_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall {
            required: uncompressed_len,
            provided: dst.len(),
        });
    }
    let block = dst.as_mut_ptr();
    let mut dst_pos = 0usize;
    let mut lit_pos = 0usize;
    let mut off_pos = 0usize;
    let mut extra_idx = 0usize;

    for token_idx in 0..num_tokens {
        let token = Token(*tokens.add(token_idx));
        let lc = token.lit_code();
        let mc = token.match_code();

        let lit_len = if lc == LIT_CODE_ESCAPE {
            read_escape(extras, extras_len, &mut extra_idx, ESCAPE_BASE_LIT)?
        } else {
            lc
        };
        let match_len = if mc == 0 {
            0
        } else if mc == MATCH_CODE_ESCAPE {
            read_escape(extras, extras_len, &mut extra_idx, bias + 15)?
        } else {
            mc + bias
        };

        // A corrupted token must never write past the declared block length.
        if lit_len + match_len > uncompressed_len - dst_pos {
            return Err(CodecError::CorruptedBitstream(
                "Token output exceeds declared block length",
            ));
        }

        if lit_len > 0 {
            if lit_pos + lit_len > literals.len() {
                return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
            }
            std::ptr::copy_nonoverlapping(literals.as_ptr().add(lit_pos), block.add(dst_pos), lit_len);
            lit_pos += lit_len;
            dst_pos += lit_len;
        }

        if match_len > 0 {
            let width = token.off_width();
            if off_pos + width > offsets_len {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let mut offset = 0usize;
            for i in 0..width {
                offset |= (*offsets.add(off_pos + i) as usize) << (8 * i);
            }
            off_pos += width;

            let available = block.add(dst_pos).offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let mut s = block.add(dst_pos).sub(offset);
            let mut d = block.add(dst_pos);
            for _ in 0..match_len {
                *d = *s;
                s = s.add(1);
                d = d.add(1);
            }
            dst_pos += match_len;
        }
    }

    if dst_pos != uncompressed_len {
        return Err(CodecError::CorruptedBitstream("Decompressed length mismatch"));
    }
    Ok(dst_pos)
}

/// Safe wrapper: decode a standalone block whose window starts at `dst[0]`.
pub fn decompress_fallback(
    tokens: &[u8],
    offsets: &[u8],
    extras: &[u8],
    literals: &[u8],
    dst: &mut [u8],
    uncompressed_len: usize,
    min_match: usize,
) -> Result<usize> {
    let buffer_start = dst.as_ptr();
    unsafe {
        decompress_fallback_raw(
            tokens.as_ptr(),
            tokens.len(),
            offsets.as_ptr(),
            offsets.len(),
            extras.as_ptr(),
            extras.len(),
            literals,
            dst,
            buffer_start,
            uncompressed_len,
            min_match,
        )
    }
}

/// Portable chained compressor: same parse as the AVX2 build.
pub fn compress_chained_fallback<P: Mode>(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) {
    let mut out = Streams { min_match: P::MIN_MATCH, tokens, offsets, extras, literals };
    unsafe {
        find_matches::<ScalarMatch, P>(full_input, block_start, block_len, table, &mut out);
    }
}

/// Portable single-block compressor with the ordinary parse.
pub fn compress_fallback(
    src: &[u8],
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) {
    init_table(table, src);
    compress_chained_fallback::<crate::finder::Lzav>(src, 0, src.len(), table, tokens, offsets, extras, literals);
}
