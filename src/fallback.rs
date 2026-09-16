use crate::error::{CodecError, Result};
use crate::format::{Token, MAX_BLOCK_SIZE, MAX_LIT_LEN, MAX_MATCH_LEN, MIN_MATCH_LEN};

/// Universal portable decompressor without any SIMD intrinsics requirement.
pub fn decompress_fallback(
    tokens: &[Token],
    offsets: &[u16],
    literals: &[u8],
    full_dst: &mut [u8],
    block_offset: usize,
    uncompressed_len: usize,
) -> Result<usize> {
    let mut dst_pos = 0usize;
    let mut lit_pos = 0usize;
    let mut offset_idx = 0usize;

    for &token in tokens {
        let lit_len = token.lit_len();
        let match_len = token.match_len();

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
            if offset_idx >= offsets.len() {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets"));
            }
            let offset = offsets[offset_idx] as usize;
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
        let mut rem = src;
        while !rem.is_empty() {
            let chunk = rem.len().min(MAX_LIT_LEN);
            tokens.push(Token::new(chunk, 0));
            literals.extend_from_slice(&rem[..chunk]);
            rem = &rem[chunk..];
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

                let mut lit_count = pos - anchor;
                let mut lit_src = anchor;
                while lit_count > MAX_LIT_LEN {
                    tokens.push(Token::new(MAX_LIT_LEN, 0));
                    literals.extend_from_slice(&src[lit_src..lit_src + MAX_LIT_LEN]);
                    lit_src += MAX_LIT_LEN;
                    lit_count -= MAX_LIT_LEN;
                }

                let first_match_chunk = match_len.min(MAX_MATCH_LEN);
                tokens.push(Token::new(lit_count, first_match_chunk));
                offsets.push(offset as u16);
                if lit_count > 0 {
                    literals.extend_from_slice(&src[lit_src..lit_src + lit_count]);
                }

                let mut rem_match = match_len - first_match_chunk;
                while rem_match > 0 {
                    let chunk = rem_match.min(MAX_MATCH_LEN);
                    tokens.push(Token::new(0, chunk));
                    offsets.push(offset as u16);
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
        forward_step = (step_skip >> 5).max(1);
    }

    let mut trailing = src_len - anchor;
    let mut lit_src = anchor;
    while trailing > 0 {
        let chunk = trailing.min(MAX_LIT_LEN);
        tokens.push(Token::new(chunk, 0));
        literals.extend_from_slice(&src[lit_src..lit_src + chunk]);
        lit_src += chunk;
        trailing -= chunk;
    }
}
