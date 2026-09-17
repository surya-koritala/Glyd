use crate::error::{CodecError, Result};
use crate::format::{encode_lit, encode_match, Token, MATCH_CODE_BIAS, MATCH_CODE_ESCAPE,
    LIT_CODE_ESCAPE, MAX_LIT_LEN, MAX_MATCH_LEN, MIN_MATCH_LEN, WINDOW_SIZE};

/// Universal portable decompressor with raw unaligned pointer support.
pub unsafe fn decompress_fallback_raw(
    tokens: *const Token,
    num_tokens: usize,
    offsets: *const u16,
    num_offsets: usize,
    extras: *const u16,
    num_extras: usize,
    literals: &[u8],
    full_dst: &mut [u8],
    block_offset: usize,
    uncompressed_len: usize,
) -> Result<usize> {
    let mut dst_pos = 0usize;
    let mut lit_pos = 0usize;
    let mut offset_idx = 0usize;
    let mut extra_idx = 0usize;

    for token_idx in 0..num_tokens {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        let lc = token.lit_code();
        let mc = token.match_code();

        let lit_len = if lc == LIT_CODE_ESCAPE {
            if extra_idx >= num_extras {
                return Err(CodecError::CorruptedBitstream("Missing extra for literal length"));
            }
            let v = std::ptr::read_unaligned(extras.add(extra_idx)) as usize;
            extra_idx += 1;
            v
        } else {
            lc
        };

        let match_len = if mc == 0 {
            0
        } else if mc == MATCH_CODE_ESCAPE {
            if extra_idx >= num_extras {
                return Err(CodecError::CorruptedBitstream("Missing extra for match length"));
            }
            let v = std::ptr::read_unaligned(extras.add(extra_idx)) as usize;
            extra_idx += 1;
            v
        } else {
            mc + MATCH_CODE_BIAS
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
    extras: &[u16],
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
            extras.as_ptr(),
            extras.len(),
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
    extras: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    let src_len = src.len();
    if src_len < MIN_MATCH_LEN {
        if src_len > 0 {
            emit_literal_run_fallback(src, tokens, extras, literals);
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
        if offset > 0 && offset < WINDOW_SIZE && candidate < pos {
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
                    if off2 > 0 && off2 < WINDOW_SIZE && cand2 < pos + 1 {
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
                    let head = lit_count - MAX_LIT_LEN;
                    emit_literal_run_fallback(&src[lit_src..lit_src + head], tokens, extras, literals);
                    (MAX_LIT_LEN, lit_src + head)
                } else {
                    (lit_count, lit_src)
                };

                let first_match_chunk = match_len.min(MAX_MATCH_LEN);
                let (lc, le) = encode_lit(first_lit_len);
                let (mc, me) = encode_match(first_match_chunk);
                tokens.push(Token::from_codes(lc, mc));
                if let Some(v) = le { extras.push(v); }
                if let Some(v) = me { extras.push(v); }
                offsets.push(match_offset as u16);
                if first_lit_len > 0 {
                    literals.extend_from_slice(&src[rem_lit_src..rem_lit_src + first_lit_len]);
                }

                let mut rem_match = match_len - first_match_chunk;
                while rem_match > 0 {
                    let chunk = rem_match.min(MAX_MATCH_LEN);
                    let (mc2, me2) = encode_match(chunk);
                    tokens.push(Token::from_codes(0, mc2));
                    if let Some(v) = me2 { extras.push(v); }
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
        forward_step = (step_skip >> 5).max(1);
    }

    let trailing = src_len - anchor;
    if trailing > 0 {
        emit_literal_run_fallback(&src[anchor..src_len], tokens, extras, literals);
    }
}

/// Portable counterpart of `x86_compress::emit_literal_run`.
#[inline]
fn emit_literal_run_fallback(
    mut run: &[u8],
    tokens: &mut Vec<Token>,
    extras: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    while !run.is_empty() {
        let n = run.len().min(MAX_LIT_LEN);
        let (lc, le) = encode_lit(n);
        tokens.push(Token::from_codes(lc, 0));
        if let Some(v) = le {
            extras.push(v);
        }
        literals.extend_from_slice(&run[..n]);
        run = &run[n..];
    }
}
