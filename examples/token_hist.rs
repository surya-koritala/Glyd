// Histograms of literal-run and match lengths, to choose the v3 token split.
use simd_stream_codec::format::*;
use std::path::Path;

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb","reymont","samba","sao","webster","xml","x-ray"];
    let dir = Path::new("corpus");
    let mut lit_h = vec![0u64; 64];
    let mut mat_h = vec![0u64; 2100];
    let mut total_tokens = 0u64;
    let mut total_comp = 0usize;
    let mut total_orig = 0usize;
    let mut ext_lit = 0u64;

    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let data = std::fs::read(&p).unwrap();
        let comp = simd_stream_codec::compress(&data);
        total_comp += comp.len();
        total_orig += data.len();
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= comp.len() {
            let h = unsafe { std::ptr::read_unaligned(comp.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let (tc, oc, ll) = (h.token_count as usize, h.offset_count as usize, h.literal_len as usize);
            let payload = if raw { h.uncompressed_len as usize } else { tc*2 + oc*2 + ll };
            if !raw {
                let base = cur + HEADER_SIZE;
                for i in 0..tc {
                    let t = unsafe { std::ptr::read_unaligned(comp.as_ptr().add(base + i*2) as *const u16) };
                    let tok = Token(t);
                    if tok.is_extended_literal() { ext_lit += 1; continue; }
                    let l = tok.lit_len().min(63);
                    let m = tok.match_len().min(2099);
                    lit_h[l] += 1; mat_h[m] += 1; total_tokens += 1;
                }
            }
            cur += HEADER_SIZE + payload;
        }
    }

    let cum = |h: &Vec<u64>, upto: usize| -> f64 {
        let s: u64 = h[..=upto.min(h.len()-1)].iter().sum();
        100.0 * s as f64 / total_tokens as f64
    };
    println!("tokens={} extended_literal_tokens={}", total_tokens, ext_lit);
    println!("\nliteral-run length coverage:");
    for b in [3usize,6,7,14,15,30,31] {
        println!("  lit_len <= {:>2}: {:.2}%", b, cum(&lit_h,b));
    }
    println!("\nmatch length coverage (match_len==0 means literal-only token):");
    println!("  match_len == 0 : {:.2}%", cum(&mat_h,0));
    for b in [18usize,33,34,49,255,2047] {
        println!("  match_len <= {:>4}: {:.2}%", b, cum(&mat_h,b));
    }

    // Model v3: 1-byte token, 3-bit lit (0..6, 7=escape), 5-bit match code
    // (0=no match, 1..30 => len 4..33, 31=escape). Escapes cost 2 extra bytes.
    let mut esc_m: u64 = 0;
    for m in 0..2100usize {
        if m != 0 && (m < 4 || m > 33) { esc_m += mat_h[m]; }
    }
    let esc_l: u64 = lit_h[7..].iter().sum();
    let v3_token_bytes = total_tokens + 2*esc_l + 2*esc_m;
    let old_token_bytes = total_tokens * 2;
    let new_comp = total_comp as i64 - old_token_bytes as i64 + v3_token_bytes as i64;
    println!("\nescape rate: literals {:.2}%  matches {:.2}%", 100.0*esc_l as f64/total_tokens as f64, 100.0*esc_m as f64/total_tokens as f64);
    println!("token bytes: now {} -> v3 {} ", old_token_bytes, v3_token_bytes);
    println!("ratio {:.4} -> {:.4}   (liblz4 2.1015, G2 needs >= 2.1435)",
             total_orig as f64/total_comp as f64, total_orig as f64/new_comp as f64);
}
