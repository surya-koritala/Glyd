#[cfg(target_arch = "x86_64")]
mod x86 {
// Why is the SIMD token pre-pass 10 cycles/token when its instruction count
// says 2? Isolate it: the same steps as the decoder's pre-pass over the real
// token and extras streams, built up one piece at a time, timed per token.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use glyd::format::*;
use std::time::Instant;

#[cfg(target_os = "linux")]
fn pin(c: usize) { unsafe {
    let mut s: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut s); libc::CPU_SET(c, &mut s);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
}}
#[cfg(not(target_os = "linux"))]
fn pin(_c: usize) {}

struct Blk { tokens: Vec<u8>, extras: Vec<u8> }

/// STAGE 0: SIMD decode + masks only (result folded into a sink).
/// STAGE 1: + widen to u16 arrays (stores).
/// STAGE 2: + escape fixup from extras (branch-free).
/// STAGE 3: + read the arrays back (what the copy loop does), no copies.
#[target_feature(enable = "avx2")]
unsafe fn prepass<const STAGE: u8>(b: &Blk, sink: &mut u64) {
    let tokens = b.tokens.as_ptr();
    let n = b.tokens.len() / 32 * 32;
    let extras = b.extras.as_ptr();
    let seven = _mm256_set1_epi8(7);
    let fifteen = _mm256_set1_epi8(15);
    let zero = _mm256_setzero_si256();
    let v_bias = _mm256_set1_epi8(5);
    let mut litl = [0u16; 32];
    let mut mll = [0u16; 32];
    let mut e = 0usize;
    let mut acc = 0u64;
    let mut t = 0usize;
    while t < n {
        let v = _mm256_loadu_si256(tokens.add(t) as *const __m256i);
        let lit = _mm256_and_si256(v, seven);
        let mc = _mm256_and_si256(_mm256_srli_epi16(v, 3), fifteen);
        let lit_esc = _mm256_cmpeq_epi8(lit, seven);
        let m_esc = _mm256_cmpeq_epi8(mc, fifteen);
        let m_zero = _mm256_cmpeq_epi8(mc, zero);
        let ml = _mm256_andnot_si256(m_zero, _mm256_add_epi8(mc, v_bias));
        let esc_mask = _mm256_movemask_epi8(_mm256_or_si256(lit_esc, m_esc)) as u32;
        let lit_esc_mask = _mm256_movemask_epi8(lit_esc) as u32;
        let m_esc_mask = _mm256_movemask_epi8(m_esc) as u32;
        acc = acc.wrapping_add(esc_mask as u64 ^ (lit_esc_mask as u64) << 1 ^ (m_esc_mask as u64) << 2);
        if STAGE >= 1 {
            let lit_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lit));
            let lit_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lit, 1));
            let ml_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ml));
            let ml_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(ml, 1));
            _mm256_storeu_si256(litl.as_mut_ptr() as *mut __m256i, lit_lo);
            _mm256_storeu_si256(litl.as_mut_ptr().add(16) as *mut __m256i, lit_hi);
            _mm256_storeu_si256(mll.as_mut_ptr() as *mut __m256i, ml_lo);
            _mm256_storeu_si256(mll.as_mut_ptr().add(16) as *mut __m256i, ml_hi);
        }
        if STAGE >= 2 && esc_mask != 0 {
            let mut bits = esc_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let l = ((lit_esc_mask >> i) & 1) as usize;
                let m = ((m_esc_mask >> i) & 1) as usize;
                if e + 2 > b.extras.len() { e = 0; }
                let b1 = *extras.add(e) as usize;
                let b2 = *extras.add(e + l) as usize;
                let cur_l = *litl.get_unchecked(i) as usize;
                let lm = l.wrapping_neg();
                *litl.get_unchecked_mut(i) = ((cur_l & !lm) | ((7 + b1) & lm)) as u16;
                let cur = *mll.get_unchecked(i) as usize;
                let mm = m.wrapping_neg();
                *mll.get_unchecked_mut(i) = ((cur & !mm) | ((20 + b2) & mm)) as u16;
                e += l + m;
            }
        }
        if STAGE >= 3 {
            for i in 0..32 {
                acc = acc.wrapping_add(*litl.get_unchecked(i) as u64 + ((*mll.get_unchecked(i) as u64) << 8));
            }
        }
        t += 32;
    }
    *sink = sink.wrapping_add(acc);
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut blocks: Vec<Blk> = Vec::new();
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let c = glyd::compress(&d);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            if h.flags & (FLAG_RAW_UNCOMPRESSED | FLAG_DENSE) == 0 {
                let tb = cur + HEADER_SIZE;
                let ob = tb + h.token_bytes as usize;
                let eb = ob + h.offset_bytes as usize;
                let lb = eb + h.extras_bytes as usize;
                blocks.push(Blk { tokens: c[tb..ob].to_vec(), extras: c[eb..lb].to_vec() });
            }
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    let ntok: usize = blocks.iter().map(|b| b.tokens.len()).sum();
    println!("{} blocks, {} tokens", blocks.len(), ntok);
    pin(4);
    let mut sink = 0u64;
    macro_rules! stage {
        ($name:expr, $s:expr) => {{
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                for b in &blocks { unsafe { prepass::<$s>(b, &mut sink); } }
                let e = t.elapsed().as_secs_f64();
                if e < best { best = e; }
            }
            println!("{:<28} {:>7.2} ms  {:>5.2} ns/token  {:>5.1} cycles/token @4.8GHz", $name, best * 1e3, best * 1e9 / ntok as f64, best * 4.8e9 / ntok as f64);
        }};
    }
    stage!("0 simd decode+masks", 0);
    stage!("1 + widen/store u16", 1);
    stage!("2 + escape fixup", 2);
    stage!("3 + read arrays back", 3);
    std::hint::black_box(sink);
}

}
#[cfg(target_arch = "x86_64")]
fn main() { x86::main() }
#[cfg(not(target_arch = "x86_64"))]
fn main() {}
