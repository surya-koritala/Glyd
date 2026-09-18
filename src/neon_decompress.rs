//! NEON port of `x86_decompress.rs`. Same structure: a 32-token pre-pass
//! (two 16-lane vectors), escape patching from the extras stream, a
//! copy-only loop, a careful one-token path for chunk tails.
//!
//! Wide copies use two 16-byte NEON loads/stores; rustc emits ldp/stp q for
//! them. `movemask` has no NEON equivalent: lanes are ANDed with a bit-select
//! pattern and folded with three pairwise adds into a 32-bit mask.
use std::arch::aarch64::*;
use crate::error::{CodecError, Result};
use crate::fallback::read_escape;
use crate::format::{ESCAPE_BASE_LIT, ESCAPE_CONT, OFFSET_BYTES,
    TOKEN_LIT_ESCAPE, TOKEN_MATCH_ESCAPE, TOKEN_OFF_SHIFT};

#[inline(always)]
unsafe fn copy32(s: *const u8, d: *mut u8) {
    vst1q_u8(d, vld1q_u8(s));
    vst1q_u8(d.add(16), vld1q_u8(s.add(16)));
}

#[inline(always)]
unsafe fn copy16(s: *const u8, d: *mut u8) {
    vst1q_u8(d, vld1q_u8(s));
}

/// 32-bit lane mask from two 16-lane compare results (all-ones per lane).
#[inline(always)]
unsafe fn movemask32(lo: uint8x16_t, hi: uint8x16_t, bitsel: uint8x16_t) -> u32 {
    let p = vpaddq_u8(vandq_u8(lo, bitsel), vandq_u8(hi, bitsel));
    let p = vpaddq_u8(p, p);
    let p = vpaddq_u8(p, p);
    vgetq_lane_u32(vreinterpretq_u32_u8(p), 0)
}

/// Bounds-checked NEON decoder; see `decompress_avx2` for the contract.
pub unsafe fn decompress_neon(
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
    table: &[u32; 256],
    esc_base_match: usize,
) -> Result<usize> {
    if uncompressed_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall {
            required: uncompressed_len,
            provided: dst.len(),
        });
    }
    let mut dst_ptr = dst.as_mut_ptr();
    let block_start = dst_ptr;
    let block_end = dst_ptr.add(uncompressed_len);
    let safe_limit = if uncompressed_len >= 64 { block_end.sub(64) } else { block_start };

    let mut lit_ptr = literals.as_ptr();
    let lit_limit = literals.as_ptr().add(literals.len());
    let mut off_pos = 0usize;
    let mut extra_idx = 0usize;
    let mut token_idx = 0;

    const CHUNK: usize = 32;
    let bias = (esc_base_match - 15) as u8;
    let seven = vdupq_n_u8(7);
    let fifteen = vdupq_n_u8(15);
    let v_bias = vdupq_n_u8(bias);
    let bitsel = vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());

    let mut litl = [0u16; CHUNK];
    let mut mll = [0u16; CHUNK];
    let mut careful = 0usize;

    'tokens: loop {
        'chunk: {
            if careful > 0 {
                careful -= 1;
                break 'chunk;
            }
            if token_idx + CHUNK > num_tokens || dst_ptr > safe_limit {
                break 'chunk;
            }
            let v0 = vld1q_u8(tokens.add(token_idx));
            let v1 = vld1q_u8(tokens.add(token_idx + 16));
            let lit0 = vandq_u8(v0, seven);
            let lit1 = vandq_u8(v1, seven);
            let mc0 = vandq_u8(vshrq_n_u8(v0, 3), fifteen);
            let mc1 = vandq_u8(vshrq_n_u8(v1, 3), fifteen);
            let lit_esc0 = vceqq_u8(lit0, seven);
            let lit_esc1 = vceqq_u8(lit1, seven);
            let m_esc0 = vceqq_u8(mc0, fifteen);
            let m_esc1 = vceqq_u8(mc1, fifteen);
            let m_zero0 = vceqzq_u8(mc0);
            let m_zero1 = vceqzq_u8(mc1);
            let ml0 = vbicq_u8(vaddq_u8(mc0, v_bias), m_zero0);
            let ml1 = vbicq_u8(vaddq_u8(mc1, v_bias), m_zero1);
            let lit_esc_mask = movemask32(lit_esc0, lit_esc1, bitsel);
            let m_esc_mask = movemask32(m_esc0, m_esc1, bitsel);
            let esc_mask = lit_esc_mask | m_esc_mask;
            let n_match = CHUNK - movemask32(m_zero0, m_zero1, bitsel).count_ones() as usize;

            vst1q_u16(litl.as_mut_ptr(), vmovl_u8(vget_low_u8(lit0)));
            vst1q_u16(litl.as_mut_ptr().add(8), vmovl_high_u8(lit0));
            vst1q_u16(litl.as_mut_ptr().add(16), vmovl_u8(vget_low_u8(lit1)));
            vst1q_u16(litl.as_mut_ptr().add(24), vmovl_high_u8(lit1));
            vst1q_u16(mll.as_mut_ptr(), vmovl_u8(vget_low_u8(ml0)));
            vst1q_u16(mll.as_mut_ptr().add(8), vmovl_high_u8(ml0));
            vst1q_u16(mll.as_mut_ptr().add(16), vmovl_u8(vget_low_u8(ml1)));
            vst1q_u16(mll.as_mut_ptr().add(24), vmovl_high_u8(ml1));

            let mut lit_sum = vaddlvq_u8(lit0) as usize + vaddlvq_u8(lit1) as usize;
            let mut ml_sum = vaddlvq_u8(ml0) as usize + vaddlvq_u8(ml1) as usize;

            let mut e = extra_idx;
            if esc_mask != 0 {
                let fields = esc_mask.count_ones() as usize
                    + (lit_esc_mask & m_esc_mask).count_ones() as usize;
                if e + 3 * fields > extras_len {
                    careful = CHUNK - 1;
                    break 'chunk;
                }
                let mut bits = esc_mask;
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if (lit_esc_mask >> i) & 1 != 0 {
                        let mut v = *extras.add(e) as usize;
                        e += 1;
                        if v == ESCAPE_CONT as usize {
                            v += std::ptr::read_unaligned(extras.add(e) as *const u16) as usize;
                            e += 2;
                            if ESCAPE_BASE_LIT + v > 0xFFFF {
                                careful = CHUNK - 1;
                                break 'chunk;
                            }
                        }
                        lit_sum += v;
                        *litl.get_unchecked_mut(i) = (ESCAPE_BASE_LIT + v) as u16;
                    }
                    if (m_esc_mask >> i) & 1 != 0 {
                        let mut v = *extras.add(e) as usize;
                        e += 1;
                        if v == ESCAPE_CONT as usize {
                            v += std::ptr::read_unaligned(extras.add(e) as *const u16) as usize;
                            e += 2;
                            if esc_base_match + v > 0xFFFF {
                                careful = CHUNK - 1;
                                break 'chunk;
                            }
                        }
                        ml_sum += v;
                        *mll.get_unchecked_mut(i) = (esc_base_match + v) as u16;
                    }
                }
            }

            let remaining = block_end.offset_from(dst_ptr) as usize;
            if lit_sum + ml_sum + 64 > remaining
                || lit_ptr.add(lit_sum + 64) > lit_limit
                || off_pos + n_match * OFFSET_BYTES > offsets_len
            {
                careful = CHUNK - 1;
                break 'chunk;
            }

            let (nl, nd, no) = copy_chunk(
                &litl, &mll, tokens.add(token_idx), lit_ptr, dst_ptr, offsets, off_pos, buffer_start,
            )?;
            lit_ptr = nl;
            dst_ptr = nd;
            off_pos = no;
            token_idx += CHUNK;
            extra_idx = e;
            continue 'tokens;
        }

        // Careful step: one token, every access checked.
        if token_idx >= num_tokens {
            break 'tokens;
        }
        let tv = *table.get_unchecked(*tokens.add(token_idx) as usize);
        token_idx += 1;

        let mut lit_len = (tv & 0xFF) as usize;
        let mut match_len = ((tv >> 8) & 0xFF) as usize;
        if tv & TOKEN_LIT_ESCAPE != 0 {
            lit_len = read_escape(extras, extras_len, &mut extra_idx, ESCAPE_BASE_LIT)?;
        }
        if tv & TOKEN_MATCH_ESCAPE != 0 {
            match_len = read_escape(extras, extras_len, &mut extra_idx, esc_base_match)?;
        }

        let remaining = if dst_ptr <= block_end {
            block_end.offset_from(dst_ptr) as usize
        } else {
            0
        };
        if lit_len + match_len > remaining {
            return Err(CodecError::CorruptedBitstream(
                "Token output exceeds declared block length",
            ));
        }

        if lit_len > 0 {
            if lit_ptr.add(lit_len) > lit_limit {
                return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
            }
            if lit_ptr.add(lit_len + 32) <= lit_limit && remaining >= lit_len + 32 {
                let mut n = 0usize;
                loop {
                    copy32(lit_ptr.add(n), dst_ptr.add(n));
                    n += 32;
                    if n >= lit_len {
                        break;
                    }
                }
            } else {
                std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
            }
            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        if match_len > 0 {
            if off_pos + OFFSET_BYTES > offsets_len {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let lo = u16::from_le_bytes([*offsets.add(off_pos), *offsets.add(off_pos + 1)]) as usize;
            let offset = lo | ((((tv >> TOKEN_OFF_SHIFT) & 1) as usize) << 16);
            off_pos += OFFSET_BYTES;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);
            let match_end = dst_ptr.add(match_len);
            let mut s = match_src;
            let mut d = dst_ptr;
            if remaining >= lit_len + match_len + 32 && offset >= 32 {
                while d < match_end {
                    copy32(s, d);
                    s = s.add(32);
                    d = d.add(32);
                }
            } else if remaining >= lit_len + match_len + 16 && offset >= 16 {
                while d < match_end {
                    copy16(s, d);
                    s = s.add(16);
                    d = d.add(16);
                }
            } else if remaining >= lit_len + match_len + 8 && offset >= 8 {
                while d < match_end {
                    std::ptr::copy_nonoverlapping(s, d, 8);
                    s = s.add(8);
                    d = d.add(8);
                }
            } else {
                while d < match_end {
                    *d = *s;
                    d = d.add(1);
                    s = s.add(1);
                }
            }
            dst_ptr = dst_ptr.add(match_len);
        }
    }

    let written = dst_ptr.offset_from(block_start) as usize;
    if written != uncompressed_len {
        return Err(CodecError::CorruptedBitstream("Decompressed length mismatch"));
    }
    Ok(written)
}

/// Pass 2: apply one chunk of decoded lengths. Bounds were established by
/// the caller from the chunk's totals; only the offset is validated here.
#[inline(never)]
unsafe fn copy_chunk(
    litl: &[u16; 32],
    mll: &[u16; 32],
    tok_base: *const u8,
    mut lit_ptr: *const u8,
    mut dst_ptr: *mut u8,
    offsets: *const u8,
    mut off_pos: usize,
    buffer_start: *const u8,
) -> Result<(*const u8, *mut u8, usize)> {
    for i in 0..32 {
        let lit = *litl.get_unchecked(i) as usize;
        let ml = *mll.get_unchecked(i) as usize;

        copy32(lit_ptr, dst_ptr);
        if lit > 32 {
            let mut n = 32usize;
            while n < lit {
                copy32(lit_ptr.add(n), dst_ptr.add(n));
                n += 32;
            }
        }
        lit_ptr = lit_ptr.add(lit);
        dst_ptr = dst_ptr.add(lit);

        if ml != 0 {
            let lo = std::ptr::read_unaligned(offsets.add(off_pos) as *const u16) as usize;
            let offset = lo | (((*tok_base.add(i) >> 7) as usize) << 16);
            off_pos += OFFSET_BYTES;
            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }
            let match_src = dst_ptr.sub(offset);
            if offset >= 32 {
                copy32(match_src, dst_ptr);
                if ml > 32 {
                    let mut n = 32usize;
                    while n < ml {
                        copy32(match_src.add(n), dst_ptr.add(n));
                        n += 32;
                    }
                }
            } else if offset >= 16 {
                let match_end = dst_ptr.add(ml);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    copy16(s, d);
                    s = s.add(16);
                    d = d.add(16);
                }
            } else if offset >= 8 {
                let match_end = dst_ptr.add(ml);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    std::ptr::copy_nonoverlapping(s, d, 8);
                    s = s.add(8);
                    d = d.add(8);
                }
            } else {
                let match_end = dst_ptr.add(ml);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    *d = *s;
                    d = d.add(1);
                    s = s.add(1);
                }
            }
            dst_ptr = dst_ptr.add(ml);
        }
    }
    Ok((lit_ptr, dst_ptr, off_pos))
}
