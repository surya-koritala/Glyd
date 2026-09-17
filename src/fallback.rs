use crate::error::{CodecError, Result};
use crate::format::{Token, MAX_BLOCK_SIZE, MAX_LIT_LEN, MAX_MATCH_LEN, MIN_MATCH_LEN};

/// Universal portable decompressor with raw unaligned pointer support.
pub unsafe fn decompress_fallback_raw(
    tokens: *const Token,
    num_tokens: usize,
    offsets: *const u16,
    num_offsets: usize,
    literals: &[u8],
    full_dst: &mut [u8],
    block_offset: usize,
    uncompressed_len: usize,
) -> Result<usize> {
    let mut dst_pos = 0usize;
    let mut lit_pos = 0usize;
    let mut offset_idx = 0usize;

    for token_idx in 0..num_tokens {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        let (lit_len, match_len) = if token.is_extended_literal() {
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Missing offset for extended literal"));
            }
            let extended_len = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;
            (extended_len, 0)
        } else {
            (token.lit_len(), token.match_len())
        };

        // 1. Literal copy
        if lit_len > 0 {
            if lit_pos + lit_len > literals.len() {
                return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
            }
            let target_start = block_offset + dst_pos;
            if target_start + lit_len > full_dst.len() {
                return Err(CodecError::OutputBufferTooSmall {
                    required: target_start + lit_len,
                    provided: full_dst.len(),
                });
            }

            full_dst[target_start..target_start + lit_len]
                .copy_from_slice(&literals[lit_pos..lit_pos + lit_len]);
            lit_pos += lit_len;
            dst_pos += lit_len;
        }

        // 2. Match copy (supports cross-block lookback)
        if match_len > 0 {
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets"));
            }
            let offset = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;

            let available = block_offset + dst_pos;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }
            let target_start = block_offset + dst_pos;
            if target_start + match_len > full_dst.len() {
                return Err(CodecError::OutputBufferTooSmall {
                    required: target_start + match_len,
                    provided: full_dst.len(),
                });
            }

            let match_start = target_start - offset;
            for i in 0..match_len {
                full_dst[target_start + i] = full_dst[match_start + i];
            }
            dst_pos += match_len;
        }
    }

    if dst_pos != uncompressed_len {
        return Err(CodecError::CorruptedBitstream("Decompressed length mismatch"));
    }

    Ok(dst_pos)
}

/// Universal portable decompressor without any SIMD intrinsics requirement.
pub fn decompress_fallback(
    tokens: &[Token],
    offsets: &[u16],
    literals: &[u8],
    full_dst: &mut [u8],
    block_offset: usize,
    uncompressed_len: usize,
) -> Result<usize> {
    unsafe {
        decompress_fallback_raw(
            tokens.as_ptr(),
            tokens.len(),
            offsets.as_ptr(),
            offsets.len(),
            literals,
            full_dst,
            block_offset,
            uncompressed_len,
        )
    }
}

const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;

#[inline(always)]
fn hash4(v: u32) -> usize {
    ((v.wrapping_mul(0x9E3779B1)) >> (32 - HASH_BITS)) as usize
}

/// Universal portable compressor.
pub fn compress_fallback(
    src: &[u8],
    tokens: &mut Vec<Token>,
    offsets: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    let src_len = src.len();
    if src_len < MIN_MATCH_LEN {
        if src_len > MAX_LIT_LEN {
            tokens.push(Token::new(31, 0));
            offsets.push(src_len as u16);
            literals.extend_from_slice(src);
        } else if src_len > 0 {
            tokens.push(Token::new(src_len, 0));
            literals.extend_from_slice(src);
        }
        return;
    }

    let mut table = [0u16; HASH_SIZE];
    let mut anchor = 0usize;
    let mut pos = 0usize;
    let limit = src_len - MIN_MATCH_LEN;
    let mut forward_step = 1usize;
    let mut step_skip = 1usize;

    while pos < limit {
        let val = u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]]);
        let h = hash4(val);
        let candidate = table[h] as usize;
        table[h] = pos as u16;

        let offset = pos.wrapping_sub(candidate);
        if offset > 0 && offset < MAX_BLOCK_SIZE && candidate < pos {
            let cand_val = u32::from_le_bytes([
                src[candidate],
                src[candidate + 1],
                src[candidate + 2],
                src[candidate + 3],
            ]);
            if val == cand_val {
                let mut match_len = 4;
                while pos + match_len < src_len && src[pos + match_len] == src[candidate + match_len] {
                    match_len += 1;
                }

                let mut match_offset = offset;
                // Lazy matching: probe pos + 1
                if pos + 1 < limit {
                    let val2 = u32::from_le_bytes([src[pos + 1], src[pos + 2], src[pos + 3], src[pos + 4]]);
                    let h2 = hash4(val2);
                    let cand2 = table[h2] as usize;
                    let off2 = (pos + 1).wrapping_sub(cand2);
                    if off2 > 0 && off2 < MAX_BLOCK_SIZE && cand2 < pos + 1 {
                        let cand2_val = u32::from_le_bytes([
                            src[cand2],
                            src[cand2 + 1],
                            src[cand2 + 2],
                            src[cand2 + 3],
                        ]);
                        if val2 == cand2_val {
                            let mut m2 = 4;
                            while pos + 1 + m2 < src_len && src[pos + 1 + m2] == src[cand2 + m2] {
                                m2 += 1;
                            }
                            if m2 > match_len {
                                table[h2] = (pos + 1) as u16;
                                pos += 1;
                                match_len = m2;
                                match_offset = off2;
                            }
                        }
                    }
                }

                let lit_count = pos - anchor;
                let lit_src = anchor;

                let (first_lit_len, rem_lit_src) = if lit_count > MAX_LIT_LEN {
                    tokens.push(Token::new(31, 0));
                    offsets.push(lit_count as u16);
                    literals.extend_from_slice(&src[lit_src..lit_src + lit_count]);
                    (0, lit_src + lit_count)
                } else {
                    (lit_count, lit_src)
                };

                let first_match_chunk = match_len.min(MAX_MATCH_LEN);
                tokens.push(Token::new(first_lit_len, first_match_chunk));
                offsets.push(match_offset as u16);
                if first_lit_len > 0 {
                    literals.extend_from_slice(&src[rem_lit_src..rem_lit_src + first_lit_len]);
                }

                let mut rem_match = match_len - first_match_chunk;
                while rem_match > 0 {
                    let chunk = rem_match.min(MAX_MATCH_LEN);
                    tokens.push(Token::new(0, chunk));
                    offsets.push(match_offset as u16);
                    rem_match -= chunk;
                }

                pos += match_len;
                anchor = pos;
                forward_step = 1;
                step_skip = 1;
                continue;
            }
        }

        pos += forward_step;
        step_skip += 1;
        forward_step = (step_skip >> 6).max(1);
    }

    let trailing = src_len - anchor;
    if trailing > MAX_LIT_LEN {
        tokens.push(Token::new(31, 0));
        offsets.push(trailing as u16);
        literals.extend_from_slice(&src[anchor..src_len]);
    } else if trailing > 0 {
        tokens.push(Token::new(trailing, 0));
        literals.extend_from_slice(&src[anchor..anchor + trailing]);
    }
}
