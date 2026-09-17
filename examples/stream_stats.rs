// Counts tokens/offsets/literals per corpus file so the cost of the current
// framing (2 token bytes + 2 offset bytes per match) can be measured exactly.
use simd_stream_codec::format::*;
use std::path::Path;

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb","reymont","samba","sao","webster","xml","x-ray"];
    let dir = Path::new("corpus");
    println!("{:<9} {:>10} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9}",
             "file","orig_MB","tokens","offsets","lit_bytes","ratio","tok_B%","save1B%");
    let (mut t_o, mut t_t, mut t_f, mut t_l, mut t_c) = (0usize,0usize,0usize,0usize,0usize);
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let data = std::fs::read(&p).unwrap();
        let comp = simd_stream_codec::compress(&data);

        // Walk the container and sum the per-block stream counts.
        let (mut toks, mut offs, mut lits) = (0usize, 0usize, 0usize);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= comp.len() {
            let h = unsafe { std::ptr::read_unaligned(comp.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let (tc, oc, ll) = (h.token_count as usize, h.offset_count as usize, h.literal_len as usize);
            let ec = h.extras_count as usize;
            let payload = if raw { h.uncompressed_len as usize } else { tc + oc*2 + ec*2 + ll };
            if !raw { toks += tc; offs += oc; lits += ll; }
            cur += HEADER_SIZE + payload;
        }
        let mb = data.len() as f64 / 1048576.0;
        let ratio = data.len() as f64 / comp.len() as f64;
        let tok_bytes = toks;
        let tok_pct = 100.0 * tok_bytes as f64 / comp.len() as f64;
        // Saving one byte per token (a 1-byte token in the common case).
        let save_pct = 100.0 * toks as f64 / comp.len() as f64;
        println!("{:<9} {:>10.1} {:>10} {:>10} {:>10} {:>8.3} {:>8.1}% {:>8.1}%",
                 f, mb, toks, offs, lits, ratio, tok_pct, save_pct);
        t_o += data.len(); t_t += toks; t_f += offs; t_l += lits; t_c += comp.len();
    }
    let ratio = t_o as f64 / t_c as f64;
    let new_ratio = t_o as f64 / (t_c - t_t) as f64;
    println!("\nTOTAL orig={:.1} MB comp={:.1} MB ratio={:.4}", t_o as f64/1048576.0, t_c as f64/1048576.0, ratio);
    println!("tokens={} offsets={} lit_bytes={}", t_t, t_f, t_l);
    println!("token stream = {:.1}% of output, offset stream = {:.1}%, literals = {:.1}%",
             100.0*t_t as f64/t_c as f64, 100.0*(t_f*2) as f64/t_c as f64, 100.0*t_l as f64/t_c as f64);
    println!("If token shrank to 1 byte: ratio {:.4} -> {:.4} (liblz4 is 2.1015)", ratio, new_ratio);
}
