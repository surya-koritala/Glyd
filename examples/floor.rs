// Where is the physical floor? Portable ablation ladder over Silesia's real
// streams. Each block is pre-decoded once into flat (lit, ml, off) arrays, so
// every rung below runs the same token sequence with one more physical cost
// removed. Output is wrong for the reduced rungs; only the time matters.
// Best of 5 per rung.
//
//   lib        the real decoder (NEON on arm64)
//   copy       pre-decoded arrays, real copy loop: no token parse, no escapes,
//              no bounds checks; only the copies and the offset-width branches
//   copy32     same, but every copy is one unconditional 32-byte load+store
//              per token (lengths <= 32 only, longer ones truncated): no
//              branches at all, keeps the store->load dependency of short
//              offsets
//   far        copy32 with offset forced >= 32: removes the store->load
//              forwarding stalls of short offsets
//   noload     one 32-byte store per token (literal only, no match load)
//   ptrchain   no memory traffic: just dst += lit + ml and one byte store
//              per token, i.e. the serial pointer chain
//   memcpy     the output, copied once
use std::time::Instant;
use simd_stream_codec::format::*;

struct Block {
    tokens: Vec<u8>,
    extras: Vec<u8>,
    lit: Vec<u32>,
    ml: Vec<u32>,
    off: Vec<u32>,
    literals: Vec<u8>,
    out_len: usize,
    file_base: usize,
    dst_pos: usize,
}

fn predecode(tokens: &[u8], offsets: &[u8], extras: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut lit, mut ml, mut off) = (Vec::new(), Vec::new(), Vec::new());
    let (mut e, mut o) = (0usize, 0usize);
    let rd = |e: &mut usize, base: usize| -> usize {
        let v = extras[*e] as usize;
        *e += 1;
        if v != ESCAPE_CONT as usize {
            return base + v;
        }
        let w = u16::from_le_bytes([extras[*e], extras[*e + 1]]) as usize;
        *e += 2;
        base + v + w
    };
    for &t in tokens {
        let tv = TOKEN_TABLE[t as usize];
        let mut l = (tv & 0xFF) as usize;
        let mut m = ((tv >> 8) & 0xFF) as usize;
        if tv & TOKEN_LIT_ESCAPE != 0 { l = rd(&mut e, ESCAPE_BASE_LIT); }
        if tv & TOKEN_MATCH_ESCAPE != 0 { m = rd(&mut e, ESCAPE_BASE_MATCH); }
        let mut of = 0usize;
        if m != 0 {
            of = u16::from_le_bytes([offsets[o], offsets[o + 1]]) as usize
                | ((((tv >> TOKEN_OFF_SHIFT) & 1) as usize) << 16);
            o += 2;
        }
        lit.push(l as u32);
        ml.push(m as u32);
        off.push(of as u32);
    }
    (lit, ml, off)
}

#[inline(always)]
unsafe fn cp32(s: *const u8, d: *mut u8) {
    std::ptr::copy_nonoverlapping(s, d, 32);
}

/// Real copy loop over pre-decoded arrays (the `copy` rung).
#[inline(never)]
unsafe fn copy_real(b: &Block, dst: *mut u8) -> usize {
    let mut lp = b.literals.as_ptr();
    let mut d = dst;
    let n = b.lit.len();
    let end = dst.add(b.out_len.saturating_sub(64));
    for i in 0..n {
        if d > end { break; }
        let lit = *b.lit.get_unchecked(i) as usize;
        let ml = *b.ml.get_unchecked(i) as usize;
        cp32(lp, d);
        if lit > 32 {
            let mut k = 32;
            while k < lit { cp32(lp.add(k), d.add(k)); k += 32; }
        }
        lp = lp.add(lit);
        d = d.add(lit);
        if ml != 0 {
            let off = *b.off.get_unchecked(i) as usize;
            let s = d.sub(off);
            if off >= 32 {
                cp32(s, d);
                if ml > 32 {
                    let mut k = 32;
                    while k < ml { cp32(s.add(k), d.add(k)); k += 32; }
                }
            } else if off >= 16 {
                let e = d.add(ml);
                let (mut s, mut dd) = (s, d);
                while dd < e { std::ptr::copy_nonoverlapping(s, dd, 16); s = s.add(16); dd = dd.add(16); }
            } else if off >= 8 {
                let e = d.add(ml);
                let (mut s, mut dd) = (s, d);
                while dd < e { std::ptr::copy_nonoverlapping(s, dd, 8); s = s.add(8); dd = dd.add(8); }
            } else {
                let e = d.add(ml);
                let (mut s, mut dd) = (s, d);
                while dd < e { *dd = *s; s = s.add(1); dd = dd.add(1); }
            }
            d = d.add(ml);
        }
    }
    d.offset_from(dst) as usize
}

/// Copy-loop variants. V=1 merged rare branch (off<32 | ml>32) with a
/// generic slow path; V=2 unconditional 64-byte match copy, slow path only
/// for off<32 | ml>64; V=3 like 2 but literal also 64 unconditional.
#[inline(never)]
unsafe fn copy_var<const V: u8>(b: &Block, dst: *mut u8) -> usize {
    let mut lp = b.literals.as_ptr();
    let mut d = dst;
    let n = b.lit.len();
    let end = dst.add(b.out_len.saturating_sub(256));
    for i in 0..n {
        if d > end { break; }
        let lit = *b.lit.get_unchecked(i) as usize;
        let ml = *b.ml.get_unchecked(i) as usize;
        cp32(lp, d);
        if V == 5 { if lit > 32 { cp32(lp.add(32), d.add(32)); cp32(lp.add(64), d.add(64)); cp32(lp.add(96), d.add(96)); if lit > 128 { let mut k = 128; while k < lit { cp32(lp.add(k), d.add(k)); k += 32; } } } }
        else if V == 3 { cp32(lp.add(32), d.add(32)); if lit > 64 { let mut k = 64; while k < lit { cp32(lp.add(k), d.add(k)); k += 32; } } }
        else if lit > 32 { let mut k = 32; while k < lit { cp32(lp.add(k), d.add(k)); k += 32; } }
        lp = lp.add(lit);
        d = d.add(lit);
        let off = *b.off.get_unchecked(i) as usize;
        let s = d.sub(off);
        if V == 4 {
            if off < 32 {
                slow_match(s, d, off, ml);
            } else {
                cp32(s, d);
                if ml > 32 {
                    cp32(s.add(32), d.add(32));
                    cp32(s.add(64), d.add(64));
                    cp32(s.add(96), d.add(96));
                    if ml > 128 { let mut k = 128; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } }
                }
            }
        } else if V == 5 {
            // V4 plus: literal tail also fixed 3x32 before looping
            if off < 32 {
                slow_match(s, d, off, ml);
            } else {
                cp32(s, d);
                if ml > 32 {
                    cp32(s.add(32), d.add(32));
                    cp32(s.add(64), d.add(64));
                    cp32(s.add(96), d.add(96));
                    if ml > 128 { let mut k = 128; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } }
                }
            }
        } else if V == 1 {
            if (off < 32) | (ml > 32) {
                slow_match(s, d, off, ml);
            } else {
                cp32(s, d);
            }
        } else {
            if off < 32 {
                slow_match(s, d, off, ml);
            } else {
                cp32(s, d);
                cp32(s.add(32), d.add(32));
                if ml > 64 { let mut k = 64; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } }
            }
        }
        d = d.add(ml);
    }
    d.offset_from(dst) as usize
}

#[inline(never)]
#[cold]
unsafe fn slow_match(s: *const u8, d: *mut u8, off: usize, ml: usize) {
    if off >= 32 {
        let mut k = 0; while k < ml { cp32(s.add(k), d.add(k)); k += 32; }
    } else if off >= 16 {
        let mut k = 0; while k < ml { std::ptr::copy_nonoverlapping(s.add(k), d.add(k), 16); k += 16; }
    } else if off >= 8 {
        let mut k = 0; while k < ml { std::ptr::copy_nonoverlapping(s.add(k), d.add(k), 8); k += 8; }
    } else {
        for k in 0..ml { *d.add(k) = *s.add(k); }
    }
}

/// Two independent blocks decoded in one interleaved loop (v5 copy rules):
/// two serial chains overlap on one core.
#[inline(never)]
unsafe fn copy_dual(a: &Block, da: *mut u8, b: &Block, db: *mut u8) -> usize {
    #[inline(always)]
    unsafe fn step(bk: &Block, i: usize, lp: &mut *const u8, d: &mut *mut u8) {
        let lit = *bk.lit.get_unchecked(i) as usize;
        let ml = *bk.ml.get_unchecked(i) as usize;
        cp32(*lp, *d);
        if lit > 32 { cp32(lp.add(32), d.add(32)); cp32(lp.add(64), d.add(64)); cp32(lp.add(96), d.add(96)); if lit > 128 { let mut k = 128; while k < lit { cp32(lp.add(k), d.add(k)); k += 32; } } }
        *lp = lp.add(lit); *d = d.add(lit);
        let off = *bk.off.get_unchecked(i) as usize;
        let s = d.sub(off);
        if off < 32 { slow_match(s, *d, off, ml); } else {
            cp32(s, *d);
            if ml > 32 { cp32(s.add(32), d.add(32)); cp32(s.add(64), d.add(64)); cp32(s.add(96), d.add(96)); if ml > 128 { let mut k = 128; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } } }
        }
        *d = d.add(ml);
    }
    let (mut lpa, mut lpb) = (a.literals.as_ptr(), b.literals.as_ptr());
    let (mut pa, mut pb) = (da, db);
    let enda = da.add(a.out_len.saturating_sub(256));
    let endb = db.add(b.out_len.saturating_sub(256));
    let n = a.lit.len().min(b.lit.len());
    let mut i = 0;
    while i < n {
        if pa > enda || pb > endb { break; }
        step(a, i, &mut lpa, &mut pa);
        step(b, i, &mut lpb, &mut pb);
        i += 1;
    }
    let mut j = i;
    while j < a.lit.len() { if pa > enda { break; } step(a, j, &mut lpa, &mut pa); j += 1; }
    let mut j = i;
    while j < b.lit.len() { if pb > endb { break; } step(b, j, &mut lpb, &mut pb); j += 1; }
    (pa.offset_from(da) + pb.offset_from(db)) as usize
}

/// copy32 plus exactly one of the rare-path branches (timing only).
#[inline(never)]
unsafe fn copy_iso<const W: u8>(b: &Block, dst: *mut u8) -> usize {
    let mut lp = b.literals.as_ptr();
    let mut d = dst;
    let n = b.lit.len();
    let end = dst.add(b.out_len.saturating_sub(256));
    for i in 0..n {
        if d > end { break; }
        let lit = *b.lit.get_unchecked(i) as usize;
        let ml = *b.ml.get_unchecked(i) as usize;
        cp32(lp, d);
        if W == 1 && lit > 32 { let mut k = 32; while k < lit { cp32(lp.add(k), d.add(k)); k += 32; } }
        lp = lp.add(lit);
        d = d.add(lit);
        let off = *b.off.get_unchecked(i) as usize;
        let s = d.sub(off);
        if W == 2 && off < 32 { slow_match(s, d, off, ml); } else {
            cp32(s, d);
            if W == 3 && ml > 32 { cp32(s.add(32), d.add(32)); cp32(s.add(64), d.add(64)); cp32(s.add(96), d.add(96)); if ml > 128 { let mut k = 128; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } } }
            if W == 4 && ml > 32 { let mut k = 32; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } }
            if W == 5 { let r = ml.saturating_sub(32); let mut k = 32; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } let _ = r; }
            if W == 6 { // branchless: always copy 32 more from a valid address (dst itself when short)
                let extra = (ml > 32) as usize;
                let src2 = if extra != 0 { s.add(32) } else { s };
                let dst2 = if extra != 0 { d.add(32) } else { d };
                cp32(src2, dst2);
                if ml > 64 { let mut k = 64; while k < ml { cp32(s.add(k), d.add(k)); k += 32; } }
            }
        }
        d = d.add(ml);
    }
    d.offset_from(dst) as usize
}

/// One unconditional 32-byte copy per literal run and per match.
/// FAR forces offset >= 32; LOAD=false skips the match load entirely.
#[inline(never)]
unsafe fn copy_flat<const FAR: bool, const LOAD: bool, const VFAR: bool>(b: &Block, dst: *mut u8) -> usize {
    let mut lp = b.literals.as_ptr();
    let mut d = dst;
    let n = b.lit.len();
    let end = dst.add(b.out_len.saturating_sub(64));
    for i in 0..n {
        if d > end { break; }
        let lit = *b.lit.get_unchecked(i) as usize;
        let ml = *b.ml.get_unchecked(i) as usize;
        cp32(lp, d);
        lp = lp.add(lit);
        d = d.add(lit);
        if LOAD {
            let mut off = *b.off.get_unchecked(i) as usize;
            if FAR { off |= 32; }
            if FAR && VFAR { off |= 65536; }
            let s = d.sub(off);
            cp32(s, d);
        }
        d = d.add(ml);
    }
    d.offset_from(dst) as usize
}

/// No copies: the serial pointer chain plus one byte store per token.
#[inline(never)]
unsafe fn ptrchain(b: &Block, dst: *mut u8) -> usize {
    let mut d = dst;
    let n = b.lit.len();
    let end = dst.add(b.out_len.saturating_sub(64));
    for i in 0..n {
        if d > end { break; }
        let lit = *b.lit.get_unchecked(i) as usize;
        let ml = *b.ml.get_unchecked(i) as usize;
        *d = lit as u8;
        d = d.add(lit + ml);
    }
    d.offset_from(dst) as usize
}

/// Pass 1 over one block: tokens -> u16 lit/ml arrays, escapes patched.
/// Returns tokens processed. ESC selects the escape loop variant.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn pass1<const ESC: u8>(tokens: &[u8], extras: &[u8], litl: &mut [u16], mll: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    let seven = vdupq_n_u8(7);
    let fifteen = vdupq_n_u8(15);
    let v_bias = vdupq_n_u8((ESCAPE_BASE_MATCH - 15) as u8);
    let bitsel = vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());
    let n = tokens.len() & !31;
    let tp = tokens.as_ptr();
    let ex = extras.as_ptr();
    let mut e = 0usize;
    let mut t = 0usize;
    let lp = litl.as_mut_ptr();
    let mp = mll.as_mut_ptr();
    let mut sum = 0usize;
    while t < n {
        let v0 = vld1q_u8(tp.add(t));
        let v1 = vld1q_u8(tp.add(t + 16));
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
        let mm = |lo: uint8x16_t, hi: uint8x16_t| -> u32 {
            let p = vpaddq_u8(vandq_u8(lo, bitsel), vandq_u8(hi, bitsel));
            let p = vpaddq_u8(p, p);
            let p = vpaddq_u8(p, p);
            vgetq_lane_u32(vreinterpretq_u32_u8(p), 0)
        };
        let lit_esc_mask = mm(lit_esc0, lit_esc1);
        let m_esc_mask = mm(m_esc0, m_esc1);
        let esc_mask = lit_esc_mask | m_esc_mask;
        let l = lp.add(t);
        let m = mp.add(t);
        vst1q_u16(l, vmovl_u8(vget_low_u8(lit0)));
        vst1q_u16(l.add(8), vmovl_high_u8(lit0));
        vst1q_u16(l.add(16), vmovl_u8(vget_low_u8(lit1)));
        vst1q_u16(l.add(24), vmovl_high_u8(lit1));
        vst1q_u16(m, vmovl_u8(vget_low_u8(ml0)));
        vst1q_u16(m.add(8), vmovl_high_u8(ml0));
        vst1q_u16(m.add(16), vmovl_u8(vget_low_u8(ml1)));
        vst1q_u16(m.add(24), vmovl_high_u8(ml1));
        sum += vaddlvq_u8(lit0) as usize + vaddlvq_u8(ml0) as usize;
        if ESC == 0 {
            // no escape handling at all (wrong output): pre-pass floor
        } else if ESC == 1 {
            // current: bit loop, two branches per lane
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if (lit_esc_mask >> i) & 1 != 0 {
                    let mut v = *ex.add(e) as usize; e += 1;
                    if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                    *l.add(i) = (ESCAPE_BASE_LIT + v) as u16;
                }
                if (m_esc_mask >> i) & 1 != 0 {
                    let mut v = *ex.add(e) as usize; e += 1;
                    if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                    *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16;
                }
            }
        } else if ESC == 2 {
            // two loops: literal escapes cannot be split from match escapes
            // in the stream, so this walks the merged mask with branchless
            // cursor advance and unconditional stores of selected values.
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let nl = ((lit_esc_mask >> i) & 1) as usize;
                let nm = ((m_esc_mask >> i) & 1) as usize;
                let vl = *ex.add(e) as usize;
                let vm = *ex.add(e + nl) as usize;
                if ((vl == 255) & (nl != 0)) | ((vm == 255) & (nm != 0)) {
                    if nl != 0 { let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *l.add(i) = (ESCAPE_BASE_LIT + v) as u16; }
                    if nm != 0 { let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16; }
                    continue;
                }
                e += nl + nm;
                let cl = *l.add(i) as usize; let cm = *m.add(i) as usize;
                *l.add(i) = (if nl != 0 { ESCAPE_BASE_LIT + vl } else { cl }) as u16;
                *m.add(i) = (if nm != 0 { ESCAPE_BASE_MATCH + vm } else { cm }) as u16;
            }
        } else if ESC == 4 {
            // select, but the unescaped value comes from the token byte, not
            // from a read-back of the array just stored by the vector pass.
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let tk = *tp.add(t + i) as usize;
                let nl = ((lit_esc_mask >> i) & 1) as usize;
                let nm = ((m_esc_mask >> i) & 1) as usize;
                let vl = *ex.add(e) as usize;
                let vm = *ex.add(e + nl) as usize;
                if ((vl == 255) & (nl != 0)) | ((vm == 255) & (nm != 0)) {
                    if nl != 0 { let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *l.add(i) = (ESCAPE_BASE_LIT + v) as u16; }
                    if nm != 0 { let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16; }
                    continue;
                }
                e += nl + nm;
                let mc = (tk >> 3) & 15;
                let cl = tk & 7;
                let cm = if mc == 0 { 0 } else { mc + MATCH_CODE_BIAS };
                *l.add(i) = (if nl != 0 { ESCAPE_BASE_LIT + vl } else { cl }) as u16;
                *m.add(i) = (if nm != 0 { ESCAPE_BASE_MATCH + vm } else { cm }) as u16;
            }
        } else if ESC == 5 {
            // like 4, without the u16 vector stores at all: every lane is
            // written by this scalar loop from the token byte (32 stores).
            for i in 0..32 {
                let tk = *tp.add(t + i) as usize;
                let nl = ((tk & 7) == 7) as usize;
                let nm = (((tk >> 3) & 15) == 15) as usize;
                let vl = *ex.add(e) as usize;
                let vm = *ex.add(e + nl) as usize;
                if ((vl == 255) & (nl != 0)) | ((vm == 255) & (nm != 0)) {
                    if nl != 0 { let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *l.add(i) = (ESCAPE_BASE_LIT + v) as u16; }
                    if nm != 0 { let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16; }
                    continue;
                }
                e += nl + nm;
                let mc = (tk >> 3) & 15;
                let cl = tk & 7;
                let cm = if mc == 0 { 0 } else { mc + MATCH_CODE_BIAS };
                *l.add(i) = (if nl != 0 { ESCAPE_BASE_LIT + vl } else { cl }) as u16;
                *m.add(i) = (if nm != 0 { ESCAPE_BASE_MATCH + vm } else { cm }) as u16;
            }
        } else if ESC == 6 {
            // variant 4 without the scalar stores (values folded into sum)
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let nl = ((lit_esc_mask >> i) & 1) as usize;
                let nm = ((m_esc_mask >> i) & 1) as usize;
                let vl = *ex.add(e) as usize;
                let vm = *ex.add(e + nl) as usize;
                e += nl + nm;
                sum += vl ^ vm;
            }
        } else if ESC == 7 {
            // variant 4 with stores but loads from a fixed extras address
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let tk = *tp.add(t + i) as usize;
                let nl = ((lit_esc_mask >> i) & 1) as usize;
                let nm = ((m_esc_mask >> i) & 1) as usize;
                let vl = *ex.add(i) as usize;
                let vm = *ex.add(i + nl) as usize;
                e += nl + nm;
                let mc = (tk >> 3) & 15;
                let cl = tk & 7;
                let cm = if mc == 0 { 0 } else { mc + MATCH_CODE_BIAS };
                *l.add(i) = (if nl != 0 { ESCAPE_BASE_LIT + vl } else { cl }) as u16;
                *m.add(i) = (if nm != 0 { ESCAPE_BASE_MATCH + vm } else { cm }) as u16;
            }
        } else if ESC == 8 {
            // only the mask walk: tz/blsr loop, nothing else
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                sum += i;
            }
        } else if ESC == 9 {
            // Loop-free expand. fields per lane (0..2) -> exclusive prefix
            // sum across 32 lanes -> tbl gather from the next 64 extras
            // bytes -> u16 blend. A 255 continuation in any consumed field
            // sends the chunk to the scalar loop.
            let zero = vdupq_n_u8(0);
            let f0 = vreinterpretq_u8_s8(vnegq_s8(vaddq_s8(vreinterpretq_s8_u8(lit_esc0), vreinterpretq_s8_u8(m_esc0))));
            let f1 = vreinterpretq_u8_s8(vnegq_s8(vaddq_s8(vreinterpretq_s8_u8(lit_esc1), vreinterpretq_s8_u8(m_esc1))));
            // inclusive prefix within each 16-lane half
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
            let total = vgetq_lane_u8(p1, 15) as usize;
            // exclusive
            let x0 = vsubq_u8(p0, f0);
            let x1 = vsubq_u8(p1, f1);
            // indices: lit field at x, match field at x + lit_esc; 0xFF where absent
            let il0 = vorrq_u8(x0, vmvnq_u8(lit_esc0));
            let il1 = vorrq_u8(x1, vmvnq_u8(lit_esc1));
            let im0 = vorrq_u8(vsubq_u8(x0, lit_esc0), vmvnq_u8(m_esc0)); // - (-1) = +1
            let im1 = vorrq_u8(vsubq_u8(x1, lit_esc1), vmvnq_u8(m_esc1));
            let tab = uint8x16x4_t(vld1q_u8(ex.add(e)), vld1q_u8(ex.add(e + 16)), vld1q_u8(ex.add(e + 32)), vld1q_u8(ex.add(e + 48)));
            let vl0 = vqtbl4q_u8(tab, il0);
            let vl1 = vqtbl4q_u8(tab, il1);
            let vm0 = vqtbl4q_u8(tab, im0);
            let vm1 = vqtbl4q_u8(tab, im1);
            let ff = vdupq_n_u8(255);
            let cont = vorrq_u8(vorrq_u8(vandq_u8(vceqq_u8(vl0, ff), lit_esc0), vandq_u8(vceqq_u8(vl1, ff), lit_esc1)),
                                vorrq_u8(vandq_u8(vceqq_u8(vm0, ff), m_esc0), vandq_u8(vceqq_u8(vm1, ff), m_esc1)));
            if vmaxvq_u8(cont) != 0 {
                // scalar path for this chunk (variant 1's loop)
                let mut bits = esc_mask;
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if (lit_esc_mask >> i) & 1 != 0 {
                        let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *l.add(i) = (ESCAPE_BASE_LIT + v) as u16;
                    }
                    if (m_esc_mask >> i) & 1 != 0 {
                        let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16;
                    }
                }
            } else {
                e += total;
                let bl = vdupq_n_u16(ESCAPE_BASE_LIT as u16);
                let bm = vdupq_n_u16(ESCAPE_BASE_MATCH as u16);
                // widen and blend
                let w = |esc: uint8x16_t, v: uint8x16_t, base: uint16x8_t, cur: uint8x16_t, out: *mut u16| {
                    let e_lo = vmovl_u8(vget_low_u8(esc)); let e_hi = vmovl_high_u8(esc);
                    let e_lo = vceqq_u16(e_lo, vdupq_n_u16(0xFF)); let e_hi = vceqq_u16(e_hi, vdupq_n_u16(0xFF));
                    let v_lo = vaddq_u16(vmovl_u8(vget_low_u8(v)), base); let v_hi = vaddq_u16(vmovl_high_u8(v), base);
                    let c_lo = vmovl_u8(vget_low_u8(cur)); let c_hi = vmovl_high_u8(cur);
                    vst1q_u16(out, vbslq_u16(e_lo, v_lo, c_lo));
                    vst1q_u16(out.add(8), vbslq_u16(e_hi, v_hi, c_hi));
                };
                w(lit_esc0, vl0, bl, lit0, l);
                w(lit_esc1, vl1, bl, lit1, l.add(16));
                w(m_esc0, vm0, bm, ml0, m);
                w(m_esc1, vm1, bm, ml1, m.add(16));
            }
        } else if ESC == 10 {
            // Loop-free expand. fields per lane (0..2) -> exclusive prefix
            // sum across 32 lanes -> tbl gather from the next 64 extras
            // bytes -> u16 blend. A 255 continuation in any consumed field
            // sends the chunk to the scalar loop.
            let zero = vdupq_n_u8(0);
            let f0 = vreinterpretq_u8_s8(vnegq_s8(vaddq_s8(vreinterpretq_s8_u8(lit_esc0), vreinterpretq_s8_u8(m_esc0))));
            let f1 = vreinterpretq_u8_s8(vnegq_s8(vaddq_s8(vreinterpretq_s8_u8(lit_esc1), vreinterpretq_s8_u8(m_esc1))));
            // inclusive prefix within each 16-lane half
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
            let total = vgetq_lane_u8(p1, 15) as usize;
            // exclusive
            let x0 = vsubq_u8(p0, f0);
            let x1 = vsubq_u8(p1, f1);
            // indices: lit field at x, match field at x + lit_esc; 0xFF where absent
            let il0 = vorrq_u8(x0, vmvnq_u8(lit_esc0));
            let il1 = vorrq_u8(x1, vmvnq_u8(lit_esc1));
            let im0 = vorrq_u8(vsubq_u8(x0, lit_esc0), vmvnq_u8(m_esc0)); // - (-1) = +1
            let im1 = vorrq_u8(vsubq_u8(x1, lit_esc1), vmvnq_u8(m_esc1));
            let tab = uint8x16x4_t(vld1q_u8(ex.add(e)), vld1q_u8(ex.add(e + 16)), vld1q_u8(ex.add(e + 32)), vld1q_u8(ex.add(e + 48)));
            let vl0 = vqtbl4q_u8(tab, il0);
            let vl1 = vqtbl4q_u8(tab, il1);
            let vm0 = vqtbl4q_u8(tab, im0);
            let vm1 = vqtbl4q_u8(tab, im1);
            let ff = vdupq_n_u8(255);
            let cont = vorrq_u8(vorrq_u8(vandq_u8(vceqq_u8(vl0, ff), lit_esc0), vandq_u8(vceqq_u8(vl1, ff), lit_esc1)),
                                vorrq_u8(vandq_u8(vceqq_u8(vm0, ff), m_esc0), vandq_u8(vceqq_u8(vm1, ff), m_esc1)));
            if ESC == 10 && false && vmaxvq_u8(cont) != 0 {
                // scalar path for this chunk (variant 1's loop)
                let mut bits = esc_mask;
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if (lit_esc_mask >> i) & 1 != 0 {
                        let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *l.add(i) = (ESCAPE_BASE_LIT + v) as u16;
                    }
                    if (m_esc_mask >> i) & 1 != 0 {
                        let mut v = *ex.add(e) as usize; e += 1;
                        if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                        *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16;
                    }
                }
            } else {
                e += total;
                let bl = vdupq_n_u16(ESCAPE_BASE_LIT as u16);
                let bm = vdupq_n_u16(ESCAPE_BASE_MATCH as u16);
                // widen and blend
                let w = |esc: uint8x16_t, v: uint8x16_t, base: uint16x8_t, cur: uint8x16_t, out: *mut u16| {
                    let e_lo = vmovl_u8(vget_low_u8(esc)); let e_hi = vmovl_high_u8(esc);
                    let e_lo = vceqq_u16(e_lo, vdupq_n_u16(0xFF)); let e_hi = vceqq_u16(e_hi, vdupq_n_u16(0xFF));
                    let v_lo = vaddq_u16(vmovl_u8(vget_low_u8(v)), base); let v_hi = vaddq_u16(vmovl_high_u8(v), base);
                    let c_lo = vmovl_u8(vget_low_u8(cur)); let c_hi = vmovl_high_u8(cur);
                    vst1q_u16(out, vbslq_u16(e_lo, v_lo, c_lo));
                    vst1q_u16(out.add(8), vbslq_u16(e_hi, v_hi, c_hi));
                };
                w(lit_esc0, vl0, bl, lit0, l);
                w(lit_esc1, vl1, bl, lit1, l.add(16));
                w(m_esc0, vm0, bm, ml0, m);
                w(m_esc1, vm1, bm, ml1, m.add(16));
            }
        } else if ESC == 3 {
            // scalar per-token walk of the 32 tokens, table-free: for every
            // token, cursor advances by escape bits; only escaped lanes store.
            // (Measures whether the mask/tz loop is the cost.)
            for i in 0..32 {
                let tk = *tp.add(t + i) as usize;
                let nl = ((tk & 7) == 7) as usize;
                let nm = ((tk >> 3) & 15 == 15) as usize;
                if nl | nm != 0 {
                    let vl = *ex.add(e) as usize;
                    let vm = *ex.add(e + nl) as usize;
                    if ((vl == 255) & (nl != 0)) | ((vm == 255) & (nm != 0)) {
                        if nl != 0 { let mut v = *ex.add(e) as usize; e += 1;
                            if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                            *l.add(i) = (ESCAPE_BASE_LIT + v) as u16; }
                        if nm != 0 { let mut v = *ex.add(e) as usize; e += 1;
                            if v == 255 { v += std::ptr::read_unaligned(ex.add(e) as *const u16) as usize; e += 2; }
                            *m.add(i) = (ESCAPE_BASE_MATCH + v) as u16; }
                        continue;
                    }
                    e += nl + nm;
                    if nl != 0 { *l.add(i) = (ESCAPE_BASE_LIT + vl) as u16; }
                    if nm != 0 { *m.add(i) = (ESCAPE_BASE_MATCH + vm) as u16; }
                }
            }
        }
        t += 32;
    }
    std::hint::black_box(sum);
    t
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut blocks: Vec<Block> = Vec::new();
    let mut raw: Vec<(Vec<u8>, usize)> = Vec::new(); // compressed bytes per file, base
    let mut total_out = 0usize;
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let c = simd_stream_codec::compress(&d);
        let file_base = total_out;
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            if h.flags & (FLAG_RAW_UNCOMPRESSED | FLAG_DENSE) == 0 {
                let tb = cur + HEADER_SIZE;
                let ob = tb + h.token_bytes as usize;
                let eb = ob + h.offset_bytes as usize;
                let lb = eb + h.extras_bytes as usize;
                let (lit, ml, off) = predecode(&c[tb..ob], &c[ob..eb], &c[eb..lb]);
                let mut literals = c[lb..lb + h.literal_len as usize].to_vec();
                literals.resize(literals.len() + 64, 0);
                let mut extras = c[eb..lb].to_vec(); extras.resize(extras.len() + 64, 0);
                blocks.push(Block { tokens: c[tb..ob].to_vec(), extras, lit, ml, off, literals, out_len: h.uncompressed_len as usize, file_base, dst_pos: total_out });
            }
            total_out += h.uncompressed_len as usize;
            cur += HEADER_SIZE + h.payload_len();
        }
        raw.push((c, file_base));
    }
    let ntok: usize = blocks.iter().map(|b| b.lit.len()).sum();
    let nmatch: usize = blocks.iter().map(|b| b.ml.iter().filter(|&&m| m != 0).count()).sum();
    let mut offhist = [0usize; 4]; // <8, <16, <32, >=32
    let mut long_lit = 0usize; let mut long_ml = 0usize;
    let (mut tail_bytes, mut tail_iters, mut near) = (0usize, 0usize, 0usize);
    let (mut esc_l, mut esc_m, mut esc_both, mut cont) = (0usize, 0usize, 0usize, 0usize);
    for b in &blocks {
        for i in 0..b.lit.len() {
            if b.ml[i] != 0 {
                let o = b.off[i];
                offhist[if o < 8 { 0 } else if o < 16 { 1 } else if o < 32 { 2 } else { 3 }] += 1;
                if b.ml[i] > 32 { long_ml += 1; }
            }
            if b.lit[i] > 32 { long_lit += 1; tail_bytes += b.lit[i] as usize - 32; tail_iters += (b.lit[i] as usize - 1) / 32; }
            if b.ml[i] > 32 { tail_bytes += b.ml[i] as usize - 32; tail_iters += (b.ml[i] as usize - 1) / 32; }
            if b.ml[i] != 0 && b.off[i] < 256 { near += 1; }
            let el = b.lit[i] as usize >= ESCAPE_BASE_LIT; let em = b.ml[i] as usize >= ESCAPE_BASE_MATCH;
            if el { esc_l += 1; } if em { esc_m += 1; } if el && em { esc_both += 1; }
            if b.lit[i] as usize >= ESCAPE_BASE_LIT + 255 || b.ml[i] as usize >= ESCAPE_BASE_MATCH + 255 { cont += 1; }
        }
    }
    println!("{} blocks, {} tokens ({} matches), {} MB output, {:.1} bytes/token",
        blocks.len(), ntok, nmatch, total_out >> 20, total_out as f64 / ntok as f64);
    println!("escapes: lit {:.1}%  match {:.1}%  both {:.1}%  any {:.1}%  cont {:.2}% of tokens",
        100.0*esc_l as f64/ntok as f64, 100.0*esc_m as f64/ntok as f64, 100.0*esc_both as f64/ntok as f64,
        100.0*(esc_l+esc_m-esc_both) as f64/ntok as f64, 100.0*cont as f64/ntok as f64);
    let mut mh = [0usize; 5]; let mut lh = [0usize; 5];
    for b in &blocks { for i in 0..b.lit.len() {
        let m = b.ml[i] as usize; if m > 32 { mh[if m <= 64 {0} else if m <= 96 {1} else if m <= 128 {2} else if m <= 256 {3} else {4}] += 1; }
        let l = b.lit[i] as usize; if l > 32 { lh[if l <= 64 {0} else if l <= 96 {1} else if l <= 128 {2} else if l <= 256 {3} else {4}] += 1; }
    } }
    println!("ml  33-64 {:.2}%  65-96 {:.2}%  97-128 {:.2}%  129-256 {:.2}%  >256 {:.2}%  (of tokens)", 100.0*mh[0] as f64/ntok as f64, 100.0*mh[1] as f64/ntok as f64, 100.0*mh[2] as f64/ntok as f64, 100.0*mh[3] as f64/ntok as f64, 100.0*mh[4] as f64/ntok as f64);
    println!("lit 33-64 {:.2}%  65-96 {:.2}%  97-128 {:.2}%  129-256 {:.2}%  >256 {:.2}%  (of tokens)", 100.0*lh[0] as f64/ntok as f64, 100.0*lh[1] as f64/ntok as f64, 100.0*lh[2] as f64/ntok as f64, 100.0*lh[3] as f64/ntok as f64, 100.0*lh[4] as f64/ntok as f64);
    println!("bytes beyond the first 32 of a run: {:.1}% of output, {:.2} extra copy32 per token; offsets < 256: {:.1}% of matches",
        100.0*tail_bytes as f64/total_out as f64, tail_iters as f64/ntok as f64, 100.0*near as f64/nmatch as f64);
    println!("offset <8: {:.1}%  8..15: {:.1}%  16..31: {:.1}%  >=32: {:.1}%   lit>32: {:.1}% of tokens  ml>32: {:.1}% of matches",
        100.0 * offhist[0] as f64 / nmatch as f64, 100.0 * offhist[1] as f64 / nmatch as f64,
        100.0 * offhist[2] as f64 / nmatch as f64, 100.0 * offhist[3] as f64 / nmatch as f64,
        100.0 * long_lit as f64 / ntok as f64, 100.0 * long_ml as f64 / nmatch as f64);

    let mut dst = vec![0u8; total_out + 4096];
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    let report = |name: &str, best: f64| {
        println!("{:<9} {:>8.2} ms  {:>6.2} ns/token  {:>6.1} GB/s  {:>5.1}% of memcpy", name, best * 1e3,
            best * 1e9 / ntok as f64, (total_out as f64 / gb) / best, 0.0);
    };
    let mut time = |f: &mut dyn FnMut()| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 { let t = Instant::now(); f(); let e = t.elapsed().as_secs_f64(); if e < best { best = e; } }
        best
    };
    let src = dst.clone();
    let t_memcpy = time(&mut || { dst.copy_from_slice(&src); std::hint::black_box(&dst); });
    let t_lib = time(&mut || { for (c, base) in &raw { let _ = simd_stream_codec::decompress_into_raw(c, &mut dst[*base..]); } });
    let base = dst.as_mut_ptr();
    let t_copy = time(&mut || { for b in &blocks { unsafe { copy_real(b, base.add(b.dst_pos)); } } });
    let t_v1 = time(&mut || { for b in &blocks { unsafe { copy_var::<1>(b, base.add(b.dst_pos)); } } });
    let t_v2 = time(&mut || { for b in &blocks { unsafe { copy_var::<2>(b, base.add(b.dst_pos)); } } });
    let t_v4 = time(&mut || { for b in &blocks { unsafe { copy_var::<4>(b, base.add(b.dst_pos)); } } });
    let t_v5 = time(&mut || { for b in &blocks { unsafe { copy_var::<5>(b, base.add(b.dst_pos)); } } });
    let t_i1 = time(&mut || { for b in &blocks { unsafe { copy_iso::<1>(b, base.add(b.dst_pos)); } } });
    let t_i2 = time(&mut || { for b in &blocks { unsafe { copy_iso::<2>(b, base.add(b.dst_pos)); } } });
    let t_i3 = time(&mut || { for b in &blocks { unsafe { copy_iso::<3>(b, base.add(b.dst_pos)); } } });
    let t_i4 = time(&mut || { for b in &blocks { unsafe { copy_iso::<4>(b, base.add(b.dst_pos)); } } });
    let t_i6 = time(&mut || { for b in &blocks { unsafe { copy_iso::<6>(b, base.add(b.dst_pos)); } } });
    let t_i0 = time(&mut || { for b in &blocks { unsafe { copy_iso::<0>(b, base.add(b.dst_pos)); } } });
    let t_dual = time(&mut || {
        let mut k = 0;
        while k + 1 < blocks.len() {
            let (a, b) = (&blocks[k], &blocks[k + 1]);
            unsafe { copy_dual(a, base.add(a.dst_pos), b, base.add(b.dst_pos)); }
            k += 2;
        }
        if k < blocks.len() { let b = &blocks[k]; unsafe { copy_var::<5>(b, base.add(b.dst_pos)); } }
    });
    let t_v3 = time(&mut || { for b in &blocks { unsafe { copy_var::<3>(b, base.add(b.dst_pos)); } } });
    // correctness of the variants against the real decoder on every block
    {
        let mut refbuf = vec![0u8; total_out + 4096];
        for (c, base) in &raw { simd_stream_codec::decompress_into_raw(c, &mut refbuf[*base..]).unwrap(); }
        dst.copy_from_slice(&refbuf);
        for v in 1..6u8 {
            for b in &blocks {
                let n = unsafe { match v { 1 => copy_var::<1>(b, base.add(b.dst_pos)), 2 => copy_var::<2>(b, base.add(b.dst_pos)), 4 => copy_var::<4>(b, base.add(b.dst_pos)), 5 => copy_var::<5>(b, base.add(b.dst_pos)), _ => copy_var::<3>(b, base.add(b.dst_pos)) } };
                let bad = (0..n).find(|&k| dst[b.dst_pos + k] != refbuf[b.dst_pos + k]);
                assert!(bad.is_none(), "copy_var {} mismatch at byte {:?} of {}", v, bad, n);
                dst[b.dst_pos..b.dst_pos + b.out_len].copy_from_slice(&refbuf[b.dst_pos..b.dst_pos + b.out_len]);
            }
        }
    }
    let t_copy32 = time(&mut || { for b in &blocks { unsafe { copy_flat::<false, true, false>(b, base.add(b.dst_pos)); } } });
    let t_far = time(&mut || { for b in &blocks { unsafe { copy_flat::<true, true, false>(b, base.add(b.dst_pos)); } } });
    let t_vfar = time(&mut || { for b in &blocks { unsafe { copy_flat::<true, true, true>(b, base.add(b.dst_pos)); } } });
    let t_noload = time(&mut || { for b in &blocks { unsafe { copy_flat::<false, false, false>(b, base.add(b.dst_pos)); } } });
    let t_chain = time(&mut || { for b in &blocks { unsafe { ptrchain(b, base.add(b.dst_pos)); } } });
    let _ = report;
    let pr = |name: &str, best: f64| {
        println!("{:<9} {:>8.2} ms  {:>6.2} ns/token  {:>6.1} GB/s  {:>5.1}% of memcpy", name, best * 1e3,
            best * 1e9 / ntok as f64, (total_out as f64 / gb) / best, 100.0 * t_memcpy / best);
    };
    #[cfg(target_arch = "aarch64")]
    {
        let mut la = vec![0u16; 262144]; let mut ma = vec![0u16; 262144];
        let t0 = time(&mut || { for b in &blocks { unsafe { pass1::<0>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t1 = time(&mut || { for b in &blocks { unsafe { pass1::<1>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t2 = time(&mut || { for b in &blocks { unsafe { pass1::<2>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t3 = time(&mut || { for b in &blocks { unsafe { pass1::<3>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t4 = time(&mut || { for b in &blocks { unsafe { pass1::<4>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t6 = time(&mut || { for b in &blocks { unsafe { pass1::<6>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t7 = time(&mut || { for b in &blocks { unsafe { pass1::<7>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t8 = time(&mut || { for b in &blocks { unsafe { pass1::<8>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t9 = time(&mut || { for b in &blocks { unsafe { pass1::<9>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t10 = time(&mut || { for b in &blocks { unsafe { pass1::<10>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        let t5 = time(&mut || { for b in &blocks { unsafe { pass1::<5>(&b.tokens, &b.extras, &mut la, &mut ma); } } });
        // verify variant 1..3 agree with predecode on the first block
        for b in &blocks {
            let n = unsafe { pass1::<9>(&b.tokens, &b.extras, &mut la, &mut ma) };
            for i in 0..n { assert_eq!((la[i] as u32, ma[i] as u32), (b.lit[i], b.ml[i]), "pass1 v9 lane {}", i); }
        }
        for v in 1..6u8 {
            let b = &blocks[0];
            let n = unsafe { match v { 1 => pass1::<1>(&b.tokens, &b.extras, &mut la, &mut ma), 2 => pass1::<2>(&b.tokens, &b.extras, &mut la, &mut ma), 4 => pass1::<4>(&b.tokens, &b.extras, &mut la, &mut ma), 5 => pass1::<5>(&b.tokens, &b.extras, &mut la, &mut ma), _ => pass1::<3>(&b.tokens, &b.extras, &mut la, &mut ma) } };
            for i in 0..n { assert_eq!((la[i] as u32, ma[i] as u32), (b.lit[i], b.ml[i]), "pass1 v{} lane {}", v, i); }
        }
        pr("p1_noesc", t0);
        pr("p1_cur", t1);
        pr("p1_sel", t2);
        pr("p1_scal", t3);
        pr("p1_seltk", t4);
        pr("p1_allsc", t5);
        pr("p1_nostore", t6);
        pr("p1_noechain", t7);
        pr("p1_maskonly", t8);
        pr("p1_expand", t9);
        pr("p1_nocont", t10);
    }
    pr("lib", t_lib);
    pr("copy", t_copy);
    pr("v1_merge", t_v1);
    pr("v2_m64", t_v2);
    pr("v3_lm64", t_v3);
    pr("iso_none", t_i0);
    pr("iso_lit", t_i1);
    pr("iso_off", t_i2);
    pr("iso_ml3", t_i3);
    pr("iso_mlloop", t_i4);
    pr("iso_mlsel", t_i6);
    pr("v4_m128", t_v4);
    pr("v5_lm128", t_v5);
    pr("dual_v5", t_dual);
    pr("copy32", t_copy32);
    pr("far", t_far);
    pr("vfar64k", t_vfar);
    pr("noload", t_noload);
    pr("ptrchain", t_chain);
    pr("memcpy", t_memcpy);
}
