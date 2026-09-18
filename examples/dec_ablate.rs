#[cfg(target_arch = "x86_64")]
mod x86 {
// Where do the decoder's cycles go? Ablation: the fast decode loop re-run over
// every Silesia block's real streams with pieces switched off. Output is wrong
// for the reduced variants; only the time matters. Pinned, best of 5.
//
//   full      the real fast-path loop (literal store, offset read, match copy)
//   nomatch   skip the match copy (still read the offset, still advance)
//   nolit     skip the literal store
//   nooff     skip the offset read (fixed offset 64)
//   walk      token walk only: read token, advance cursors, no memory traffic
//   -noesc    escapes ignored (escaped tokens use the direct field values), so
//             the escape branch and extras reads vanish; output is wrong
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use simd_stream_codec::format::*;
use std::time::Instant;

#[cfg(target_os = "linux")]
fn pin(c: usize) { unsafe {
    let mut s: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut s); libc::CPU_SET(c, &mut s);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
}}
#[cfg(not(target_os = "linux"))]
fn pin(_c: usize) {}

struct Block { tokens: Vec<u8>, offsets: Vec<u8>, extras: Vec<u8>, literals: Vec<u8>, out_len: usize, file_base: usize, dst_pos: usize }

/// Fast loop with switches. Mirrors x86_decompress's fast phase; the careful
/// tail is replaced by a plain stop, so the last 64 bytes are skipped.
#[target_feature(enable = "avx2")]
unsafe fn run<const LIT: bool, const OFF: bool, const MATCH: bool, const ESC: bool, const FIXW: bool, const ALU: bool, const NARROW: bool, const ONEGUARD: bool>(b: &Block, dst: &mut [u8], buffer_start: *const u8) -> usize {
    let tokens = b.tokens.as_ptr();
    let num_tokens = b.tokens.len();
    let offsets = b.offsets.as_ptr();
    let offsets_len = b.offsets.len();
    let extras = b.extras.as_ptr();
    let extras_len = b.extras.len();
    let mut lit_ptr = b.literals.as_ptr();
    let lit_limit = lit_ptr.add(b.literals.len());
    let mut dst_ptr = dst.as_mut_ptr();
    let block_end = dst_ptr.add(b.out_len);
    let safe_limit = if b.out_len >= 64 { block_end.sub(64) } else { dst_ptr };
    let (mut off_pos, mut extra_idx, mut token_idx) = (0usize, 0usize, 0usize);
    let table = &TOKEN_TABLE;
    let mut sink = 0usize; // keeps the walk alive when nothing is stored
    // ONEGUARD: only the token count is checked (measurement only; unsafe on
    // arbitrary input, fine on this valid corpus where the ablation stops at
    // a conservative token bound).
    let tok_end = if ONEGUARD { num_tokens.saturating_sub(64) } else { num_tokens };
    while token_idx < tok_end && (ONEGUARD || (dst_ptr <= safe_limit && lit_ptr.add(64) <= lit_limit && off_pos + OFFSET_BYTES <= offsets_len)) {
        let t = *tokens.add(token_idx) as usize;
        // ALU: fields by shift and mask, no table load on the recurrence.
        let tv = if ALU {
            let lc = t & 3; let mc = (t >> 2) & 15; let w = (t >> 6) + 1;
            let lit = if lc == 3 { TOKEN_LIT_ESCAPE } else { lc as u32 };
            let m = if mc == 15 { TOKEN_MATCH_ESCAPE } else if mc != 0 { ((mc + MATCH_CODE_BIAS) as u32) << 8 } else { 0 };
            lit | m | if mc != 0 { (w as u32) << TOKEN_OFF_SHIFT } else { 0 }
        } else {
            *table.get_unchecked(t)
        };
        let mut lit_len = (tv & 0xFF) as usize;
        let mut match_len = ((tv >> 8) & 0xFF) as usize;
        let off_hi = ((tv >> TOKEN_OFF_SHIFT) & 1) as usize;
        if ESC && tv & TOKEN_ESCAPE_MASK != 0 {
            let mut e = extra_idx;
            if tv & TOKEN_LIT_ESCAPE != 0 {
                lit_len = simd_stream_codec::fallback::read_escape(extras, extras_len, &mut e, ESCAPE_BASE_LIT).unwrap();
            }
            if tv & TOKEN_MATCH_ESCAPE != 0 {
                match_len = simd_stream_codec::fallback::read_escape(extras, extras_len, &mut e, ESCAPE_BASE_MATCH).unwrap();
            }
            let remaining = block_end.offset_from(dst_ptr) as usize;
            if lit_len + match_len + 64 > remaining || lit_ptr.add(lit_len + 64) > lit_limit { break; }
            extra_idx = e;
        }
        token_idx += 1;
        sink = sink.wrapping_add(lit_len ^ (match_len << 8));
        if LIT && NARROW {
            // 16-byte literal copy covers the direct field (<= 2) and most escapes.
            let v = _mm_loadu_si128(lit_ptr as *const __m128i);
            _mm_storeu_si128(dst_ptr as *mut __m128i, v);
            if lit_len > 16 {
                let mut n = 16usize;
                while n < lit_len {
                    let v = _mm256_loadu_si256(lit_ptr.add(n) as *const __m256i);
                    _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, v);
                    n += 32;
                }
            }
        } else if LIT {
            let v = _mm256_loadu_si256(lit_ptr as *const __m256i);
            _mm256_storeu_si256(dst_ptr as *mut __m256i, v);
            if lit_len > 32 {
                let mut n = 32usize;
                while n < lit_len {
                    let v = _mm256_loadu_si256(lit_ptr.add(n) as *const __m256i);
                    _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, v);
                    n += 32;
                }
            }
        }
        lit_ptr = lit_ptr.add(lit_len);
        dst_ptr = dst_ptr.add(lit_len);
        if match_len != 0 {
            // v6: constant-stride 2-byte offsets, so FIXW is the format now.
            let _ = FIXW;
            let offset = if OFF {
                let lo = std::ptr::read_unaligned(offsets.add(off_pos) as *const u16) as usize;
                off_pos += OFFSET_BYTES;
                lo | (off_hi << 16)
            } else {
                off_pos += OFFSET_BYTES;
                64
            };
            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available { return 0; }
            if MATCH {
                let match_src = dst_ptr.sub(offset);
                if NARROW && match_len <= 16 && offset >= 16 {
                    let m = _mm_loadu_si128(match_src as *const __m128i);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, m);
                } else if offset >= 32 {
                    let m = _mm256_loadu_si256(match_src as *const __m256i);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, m);
                    if match_len > 32 {
                        let mut n = 32usize;
                        while n < match_len {
                            let m = _mm256_loadu_si256(match_src.add(n) as *const __m256i);
                            _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, m);
                            n += 32;
                        }
                    }
                } else if offset >= 16 {
                    let match_end = dst_ptr.add(match_len);
                    let (mut s, mut d) = (match_src, dst_ptr);
                    while d < match_end { let m = _mm_loadu_si128(s as *const __m128i); _mm_storeu_si128(d as *mut __m128i, m); s = s.add(16); d = d.add(16); }
                } else if offset >= 8 {
                    let match_end = dst_ptr.add(match_len);
                    let (mut s, mut d) = (match_src, dst_ptr);
                    while d < match_end { std::ptr::copy_nonoverlapping(s, d, 8); s = s.add(8); d = d.add(8); }
                } else {
                    let match_end = dst_ptr.add(match_len);
                    let (mut s, mut d) = (match_src, dst_ptr);
                    while d < match_end { *d = *s; d = d.add(1); s = s.add(1); }
                }
            }
            dst_ptr = dst_ptr.add(match_len);
        }
    }
    std::hint::black_box(sink);
    dst_ptr.offset_from(dst.as_ptr()) as usize
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut blocks: Vec<Block> = Vec::new();
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
                blocks.push(Block {
                    tokens: c[tb..ob].to_vec(), offsets: c[ob..eb].to_vec(),
                    extras: c[eb..lb].to_vec(), literals: c[lb..lb + h.literal_len as usize].to_vec(),
                    out_len: h.uncompressed_len as usize, file_base, dst_pos: total_out,
                });
            }
            total_out += h.uncompressed_len as usize;
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    let ntok: usize = blocks.iter().map(|b| b.tokens.len()).sum();
    println!("{} blocks, {} tokens, {} MB output", blocks.len(), ntok, total_out >> 20);
    pin(4);
    // One output buffer per block region is unrealistic; use one big buffer
    // and decode each block at its own offset, like the real decoder.
    let mut dst = vec![0u8; total_out + 4096];
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    macro_rules! variant {
        ($name:expr, $l:expr, $o:expr, $m:expr, $e:expr, $f:expr, $a:expr, $n:expr, $g:expr) => {{
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let base = dst.as_ptr();
                for b in &blocks {
                    let bs = unsafe { base.add(b.file_base) };
                    let _ = unsafe { run::<$l, $o, $m, $e, $f, $a, $n, $g>(b, &mut dst[b.dst_pos..], bs) };
                }
                let e = t.elapsed().as_secs_f64();
                if e < best { best = e; }
            }
            println!("{:<8} {:>8.2} ms  {:>6.2} ns/token  {:>6.2} GB/s", $name, best * 1e3, best * 1e9 / ntok as f64, (total_out as f64 / gb) / best);
        }};
    }
    variant!("full", true, true, true, true, false, false, false, false);
    variant!("walk", false, false, false, true, false, false, false, false);
    // The real library decoder on the same blocks, same conditions.
    {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let base = dst.as_ptr();
            for b in &blocks {
                let bs = unsafe { base.add(b.file_base) };
                let out = &mut dst[b.dst_pos..];
                let _ = unsafe { simd_stream_codec::x86_decompress::decompress_avx2(
                    b.tokens.as_ptr(), b.tokens.len(), b.offsets.as_ptr(), b.offsets.len(),
                    b.extras.as_ptr(), b.extras.len(), &b.literals, out, bs, b.out_len,
                    &TOKEN_TABLE, ESCAPE_BASE_MATCH) }.expect("lib decode");
            }
            let e = t.elapsed().as_secs_f64();
            if e < best { best = e; }
        }
        println!("{:<8} {:>8.2} ms  {:>6.2} ns/token  {:>6.2} GB/s", "lib", best * 1e3, best * 1e9 / ntok as f64, (total_out as f64 / gb) / best);
    }
}

}
#[cfg(target_arch = "x86_64")]
fn main() { x86::main() }
#[cfg(not(target_arch = "x86_64"))]
fn main() {}
