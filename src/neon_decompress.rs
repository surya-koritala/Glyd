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

/// Widen 16 lanes to u16, replace escaped lanes with `base + v`, store,
/// and return the two halves summed (for the chunk total).
#[inline(always)]
unsafe fn blend16(esc: uint8x16_t, v: uint8x16_t, base: uint16x8_t, cur: uint8x16_t, out: *mut u16) -> uint16x8_t {
    let ff = vdupq_n_u16(0xFF);
    let e_lo = vceqq_u16(vmovl_u8(vget_low_u8(esc)), ff);
    let e_hi = vceqq_u16(vmovl_high_u8(esc), ff);
    let lo = vbslq_u16(e_lo, vaddq_u16(vmovl_u8(vget_low_u8(v)), base), vmovl_u8(vget_low_u8(cur)));
    let hi = vbslq_u16(e_hi, vaddq_u16(vmovl_high_u8(v), base), vmovl_high_u8(cur));
    vst1q_u16(out, lo);
    vst1q_u16(out.add(8), hi);
    vaddq_u16(lo, hi)
}

#[derive(Clone, Copy)]
struct Lens {
    litl: [u16; 32],
    mll: [u16; 32],
    lit_sum: usize,
    ml_sum: usize,
    n_match: usize,
    e_end: usize,
}

struct Prepass {
    tokens: *const u8,
    extras: *const u8,
    extras_len: usize,
    extras_readable: usize,
    esc_base_match: usize,
    seven: uint8x16_t,
    fifteen: uint8x16_t,
    v_bias: uint8x16_t,
    bitsel: uint8x16_t,
    zero: uint8x16_t,
    one: uint8x16_t,
    ff: uint8x16_t,
    v_base_lit: uint16x8_t,
    v_base_match: uint16x8_t,
}

impl Prepass {
    /// Decode 32 tokens at `t` into `out`. Returns false when the chunk must
    /// be taken carefully (extras too short, a continuation past u16).
    #[inline(always)]
    unsafe fn run(&self, t: usize, e0: usize, out: &mut Lens) -> bool {
        let (tokens, extras, extras_len) = (self.tokens, self.extras, self.extras_len);
        let (seven, fifteen, v_bias, zero, one, ff) = (self.seven, self.fifteen, self.v_bias, self.zero, self.one, self.ff);
        let esc_base_match = self.esc_base_match;
        let v0 = vld1q_u8(tokens.add(t));
        let v1 = vld1q_u8(tokens.add(t + 16));
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
        out.n_match = 32 - (vaddlvq_u8(vandq_u8(m_zero0, one)) + vaddlvq_u8(vandq_u8(m_zero1, one))) as usize;

        // Escapes, loop-free. 31% of Silesia tokens carry one; a
        // data-dependent loop over them mispredicts at exit every chunk
        // and the flush exposes the whole vector chain. Instead: fields
        // per lane (0..2) -> exclusive prefix sum -> tbl gather from the
        // next 64 extras bytes -> u16 blend. A 255 continuation in any
        // consumed field (0.3% of tokens) takes the scalar loop below.
        let mut e = e0;
        if e + 64 > self.extras_readable {
            return false;
        }
        let f0 = vreinterpretq_u8_s8(vnegq_s8(vaddq_s8(vreinterpretq_s8_u8(lit_esc0), vreinterpretq_s8_u8(m_esc0))));
        let f1 = vreinterpretq_u8_s8(vnegq_s8(vaddq_s8(vreinterpretq_s8_u8(lit_esc1), vreinterpretq_s8_u8(m_esc1))));
        let mut p0 = f0;
        p0 = vaddq_u8(p0, vextq_u8(zero, p0, 15));
        p0 = vaddq_u8(p0, vextq_u8(zero, p0, 14));
        p0 = vaddq_u8(p0, vextq_u8(zero, p0, 12));
        p0 = vaddq_u8(p0, vextq_u8(zero, p0, 8));
        let mut p1 = f1;
        p1 = vaddq_u8(p1, vextq_u8(zero, p1, 15));
        p1 = vaddq_u8(p1, vextq_u8(zero, p1, 14));
        p1 = vaddq_u8(p1, vextq_u8(zero, p1, 12));
        p1 = vaddq_u8(p1, vextq_u8(zero, p1, 8));
        p1 = vaddq_u8(p1, vdupq_laneq_u8(p0, 15));
        let fields = vgetq_lane_u8(p1, 15) as usize;
        if e + fields > extras_len {
            return false;
        }
        let x0 = vsubq_u8(p0, f0);
        let x1 = vsubq_u8(p1, f1);
        let il0 = vorrq_u8(x0, vmvnq_u8(lit_esc0));
        let il1 = vorrq_u8(x1, vmvnq_u8(lit_esc1));
        let im0 = vorrq_u8(vsubq_u8(x0, lit_esc0), vmvnq_u8(m_esc0));
        let im1 = vorrq_u8(vsubq_u8(x1, lit_esc1), vmvnq_u8(m_esc1));
        let tab = uint8x16x4_t(
            vld1q_u8(extras.add(e)), vld1q_u8(extras.add(e + 16)),
            vld1q_u8(extras.add(e + 32)), vld1q_u8(extras.add(e + 48)),
        );
        let vl0 = vqtbl4q_u8(tab, il0);
        let vl1 = vqtbl4q_u8(tab, il1);
        let vm0 = vqtbl4q_u8(tab, im0);
        let vm1 = vqtbl4q_u8(tab, im1);
        let cont = vorrq_u8(
            vorrq_u8(vandq_u8(vceqq_u8(vl0, ff), lit_esc0), vandq_u8(vceqq_u8(vl1, ff), lit_esc1)),
            vorrq_u8(vandq_u8(vceqq_u8(vm0, ff), m_esc0), vandq_u8(vceqq_u8(vm1, ff), m_esc1)),
        );
        let litl = out.litl.as_mut_ptr();
        let mll = out.mll.as_mut_ptr();
        if vmaxvq_u8(cont) == 0 {
            e += fields;
            let ls0 = blend16(lit_esc0, vl0, self.v_base_lit, lit0, litl);
            let ls1 = blend16(lit_esc1, vl1, self.v_base_lit, lit1, litl.add(16));
            let ms0 = blend16(m_esc0, vm0, self.v_base_match, ml0, mll);
            let ms1 = blend16(m_esc1, vm1, self.v_base_match, ml1, mll.add(16));
            out.lit_sum = vaddvq_u16(vaddq_u16(ls0, ls1)) as usize;
            out.ml_sum = vaddvq_u16(vaddq_u16(ms0, ms1)) as usize;
        } else {
            vst1q_u16(litl, vmovl_u8(vget_low_u8(lit0)));
            vst1q_u16(litl.add(8), vmovl_high_u8(lit0));
            vst1q_u16(litl.add(16), vmovl_u8(vget_low_u8(lit1)));
            vst1q_u16(litl.add(24), vmovl_high_u8(lit1));
            vst1q_u16(mll, vmovl_u8(vget_low_u8(ml0)));
            vst1q_u16(mll.add(8), vmovl_high_u8(ml0));
            vst1q_u16(mll.add(16), vmovl_u8(vget_low_u8(ml1)));
            vst1q_u16(mll.add(24), vmovl_high_u8(ml1));
            let mut ls = vaddlvq_u8(lit0) as usize + vaddlvq_u8(lit1) as usize;
            let mut ms = vaddlvq_u8(ml0) as usize + vaddlvq_u8(ml1) as usize;
            let lit_esc_mask = movemask32(lit_esc0, lit_esc1, self.bitsel);
            let m_esc_mask = movemask32(m_esc0, m_esc1, self.bitsel);
            let both = (lit_esc_mask & m_esc_mask).count_ones() as usize;
            if e + 3 * (fields + both) > extras_len {
                return false;
            }
            let mut bits = lit_esc_mask | m_esc_mask;
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
                            return false;
                        }
                    }
                    ls += v;
                    *litl.add(i) = (ESCAPE_BASE_LIT + v) as u16;
                }
                if (m_esc_mask >> i) & 1 != 0 {
                    let mut v = *extras.add(e) as usize;
                    e += 1;
                    if v == ESCAPE_CONT as usize {
                        v += std::ptr::read_unaligned(extras.add(e) as *const u16) as usize;
                        e += 2;
                        if esc_base_match + v > 0xFFFF {
                            return false;
                        }
                    }
                    ms += v;
                    *mll.add(i) = (esc_base_match + v) as u16;
                }
            }
            out.lit_sum = ls;
            out.ml_sum = ms;
        }
        out.e_end = e;
        true
    }
}

/// Bounds-checked NEON decoder; see `decompress_avx2` for the contract.
pub unsafe fn decompress_neon(
    tokens: *const u8,
    num_tokens: usize,
    offsets: *const u8,
    offsets_len: usize,
    extras: *const u8,
    extras_len: usize,
    extras_readable: usize,
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
    let pp = Prepass {
        tokens, extras, extras_len, extras_readable, esc_base_match,
        seven: vdupq_n_u8(7),
        fifteen: vdupq_n_u8(15),
        v_bias: vdupq_n_u8((esc_base_match - 15) as u8),
        bitsel: vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr()),
        zero: vdupq_n_u8(0),
        one: vdupq_n_u8(1),
        ff: vdupq_n_u8(255),
        v_base_lit: vdupq_n_u16(ESCAPE_BASE_LIT as u16),
        v_base_match: vdupq_n_u16(esc_base_match as u16),
    };

    // Two length buffers: chunk k+1 is pre-passed before chunk k is copied,
    // so the vector chain's latency (and the store->load of the arrays)
    // overlaps the previous copy loop instead of stalling every chunk.
    let mut bufs = [Lens { litl: [0u16; CHUNK], mll: [0u16; CHUNK], lit_sum: 0, ml_sum: 0, n_match: 0, e_end: 0 }; 2];
    let mut cur = 0usize;
    let mut prepared = false;
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
            if !prepared {
                if !pp.run(token_idx, extra_idx, &mut bufs[cur]) {
                    careful = CHUNK - 1;
                    break 'chunk;
                }
            }
            prepared = false;
            let (lit_sum, ml_sum, n_match, e_end) = {
                let b = &bufs[cur];
                (b.lit_sum, b.ml_sum, b.n_match, b.e_end)
            };

            // Chunk bounds: wild stores need 64 bytes past the chunk's last
            // byte; literal loads need 64 past the chunk's literals; offsets
            // need OFFSET_BYTES per match.
            let remaining = block_end.offset_from(dst_ptr) as usize;
            if lit_sum + ml_sum + 64 > remaining
                || lit_ptr.add(lit_sum + 64) > lit_limit
                || off_pos + n_match * OFFSET_BYTES > offsets_len
            {
                careful = CHUNK - 1;
                break 'chunk;
            }

            // Pre-pass the next chunk while this one's copies are in flight.
            let next = cur ^ 1;
            if token_idx + 2 * CHUNK <= num_tokens {
                prepared = pp.run(token_idx + CHUNK, e_end, &mut bufs[next]);
            }

            let b = &bufs[cur];
            let (nl, nd, no) = copy_chunk(
                &b.litl, &b.mll, tokens.add(token_idx), lit_ptr, dst_ptr, offsets, off_pos, buffer_start,
            )?;
            lit_ptr = nl;
            dst_ptr = nd;
            off_pos = no;
            token_idx += CHUNK;
            extra_idx = e_end;
            cur = next;
            continue 'tokens;
        }
        prepared = false;

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

/// Overlapping match copy for offsets under 32 (2.2% of matches).
#[cold]
#[inline(never)]
unsafe fn short_match(s: *const u8, d: *mut u8, offset: usize, ml: usize) {
    let end = d.add(ml);
    let (mut s, mut d) = (s, d);
    if offset >= 16 {
        while d < end { copy16(s, d); s = s.add(16); d = d.add(16); }
    } else if offset >= 8 {
        while d < end { std::ptr::copy_nonoverlapping(s, d, 8); s = s.add(8); d = d.add(8); }
    } else {
        while d < end { *d = *s; d = d.add(1); s = s.add(1); }
    }
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

        // Rare paths (lit > 32: 1.9% of tokens, ml > 32: 6.6%, offset < 32:
        // 2.2%) each cost one mispredict; measured additive. Tails up to
        // 128 bytes are copied with a fixed three stores so the inner loop
        // and its exit mispredict only run past that.
        copy32(lit_ptr, dst_ptr);
        if lit > 32 {
            copy32(lit_ptr.add(32), dst_ptr.add(32));
            copy32(lit_ptr.add(64), dst_ptr.add(64));
            copy32(lit_ptr.add(96), dst_ptr.add(96));
            if lit > 128 {
                let mut n = 128usize;
                while n < lit {
                    copy32(lit_ptr.add(n), dst_ptr.add(n));
                    n += 32;
                }
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
                    copy32(match_src.add(32), dst_ptr.add(32));
                    copy32(match_src.add(64), dst_ptr.add(64));
                    copy32(match_src.add(96), dst_ptr.add(96));
                    if ml > 128 {
                        let mut n = 128usize;
                        while n < ml {
                            copy32(match_src.add(n), dst_ptr.add(n));
                            n += 32;
                        }
                    }
                }
            } else {
                short_match(match_src, dst_ptr, offset, ml);
            }
            dst_ptr = dst_ptr.add(ml);
        }
    }
    Ok((lit_ptr, dst_ptr, off_pos))
}
