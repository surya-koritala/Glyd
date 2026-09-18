// How often does a match reuse the previous match's offset? Decides whether a
// repeat-offset code is worth a format change. Measures before building.
use simd_stream_codec::format::*;

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let (mut tot_m, mut tot_rep, mut tot_comp, mut tot_off) = (0u64, 0u64, 0usize, 0u64);
    let mut hist = [0u64; 5]; // <256, <1K, <4K, <16K, rest
    println!("{:<9} {:>10} {:>10} {:>8}", "file", "matches", "repeats", "repeat%");
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let data = std::fs::read(&p).unwrap();
        let comp = simd_stream_codec::compress(&data);
        tot_comp += comp.len();
        let (mut m, mut rep) = (0u64, 0u64);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= comp.len() {
            let h = unsafe { std::ptr::read_unaligned(comp.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let (tc, oc, ec, ll) = (h.token_count as usize, h.offset_count as usize,
                                    h.extras_count as usize, h.literal_len as usize);
            let payload = if raw { h.uncompressed_len as usize } else { payload_len(tc, oc, ec, ll) };
            if !raw {
                let obase = cur + HEADER_SIZE + tc;
                let mut last: i64 = -1;
                for i in 0..oc {
                    let o = unsafe { std::ptr::read_unaligned(comp.as_ptr().add(obase + i*2) as *const u16) } as i64;
                    m += 1;
                    if o == last { rep += 1; }
                    let b = if o < 256 { 0 } else if o < 1024 { 1 }
                            else if o < 4096 { 2 } else if o < 16384 { 3 } else { 4 };
                    hist[b] += 1;
                    last = o;
                }
                tot_off += oc as u64;
            }
            cur += HEADER_SIZE + payload;
        }
        println!("{:<9} {:>10} {:>10} {:>7.1}%", f, m, rep, 100.0 * rep as f64 / m.max(1) as f64);
        tot_m += m; tot_rep += rep;
    }
    let pct = 100.0 * tot_rep as f64 / tot_m.max(1) as f64;
    println!("\nTOTAL matches {} repeats {} = {:.1}%", tot_m, tot_rep, pct);
    println!("offset stream = {} bytes = {:.1}% of {} compressed bytes",
             tot_off * 2, 100.0 * (tot_off * 2) as f64 / tot_comp as f64, tot_comp);
    let saved = tot_rep * 2;
    println!("Removing the offset on every repeat saves {} bytes = {:.2}% of output",
             saved, 100.0 * saved as f64 / tot_comp as f64);
    println!("\noffset magnitude distribution:");
    let names = ["<256", "<1K", "<4K", "<16K", ">=16K"];
    let mut cum = 0u64;
    for i in 0..5 {
        cum += hist[i];
        println!("  {:<6} {:>10} {:>6.1}%   cumulative {:>6.1}%",
                 names[i], hist[i], 100.0*hist[i] as f64/tot_m.max(1) as f64,
                 100.0*cum as f64/tot_m.max(1) as f64);
    }
    let one_byte = hist[0];
    println!("\nIf offsets <256 took 1 byte instead of 2: saves {} bytes = {:.2}% of output",
             one_byte, 100.0 * one_byte as f64 / tot_comp as f64);
    println!("   ratio 2.1950 -> about {:.4}", 2.1950 / (1.0 - one_byte as f64 / tot_comp as f64));
}
