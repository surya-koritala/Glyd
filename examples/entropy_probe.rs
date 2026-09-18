// How many bytes would entropy coding actually save? Measures Shannon entropy
// of each stream on real Silesia output, before committing to an implementation.
use simd_stream_codec::format::*;

fn ent(h: &[u64], n: u64) -> f64 {
    if n == 0 { return 0.0; }
    let mut e = 0.0;
    for &c in h {
        if c > 0 { let p = c as f64 / n as f64; e -= p * p.log2(); }
    }
    e
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut tok = [0u64; 256];
    let mut ohi = [0u64; 256];
    let mut olo = [0u64; 256];
    let mut lit = [0u64; 256];
    let (mut ntok, mut noff, mut nlit) = (0u64, 0u64, 0u64);
    let (mut orig, mut comp_total) = (0usize, 0usize);

    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let c = simd_stream_codec::compress(&d);
        orig += d.len(); comp_total += c.len();
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let (tc, oc, ec, ll) = (h.token_count as usize, h.offset_count as usize,
                                    h.extras_count as usize, h.literal_len as usize);
            let pay = if raw { h.uncompressed_len as usize } else { payload_len(tc, oc, ec, ll) };
            if !raw {
                let tb = cur + HEADER_SIZE;
                for i in 0..tc { tok[unsafe { *c.as_ptr().add(tb + i) } as usize] += 1; }
                ntok += tc as u64;
                let ob = tb + tc;
                for i in 0..oc {
                    let v = unsafe { std::ptr::read_unaligned(c.as_ptr().add(ob + i*2) as *const u16) };
                    ohi[(v >> 8) as usize] += 1; olo[(v & 0xFF) as usize] += 1;
                }
                noff += oc as u64;
                let lb = ob + oc*2 + ec*2;
                for i in 0..ll { lit[unsafe { *c.as_ptr().add(lb + i) } as usize] += 1; }
                nlit += ll as u64;
            }
            cur += HEADER_SIZE + pay;
        }
    }

    let et = ent(&tok, ntok); let eh = ent(&ohi, noff);
    let el = ent(&olo, noff); let eli = ent(&lit, nlit);
    println!("stream        symbols        bytes now   entropy b/sym   ideal bytes    saving");
    let rows: [(&str, u64, f64, f64); 4] = [
        ("tokens", ntok, ntok as f64, et),
        ("offset hi", noff, noff as f64, eh),
        ("offset lo", noff, noff as f64, el),
        ("literals", nlit, nlit as f64, eli),
    ];
    let mut total_save = 0.0;
    for (name, n, now, e) in rows {
        let ideal = n as f64 * e / 8.0;
        let save = now - ideal;
        total_save += save;
        println!("{:<12} {:>10} {:>14.0} {:>15.3} {:>13.0} {:>9.0}", name, n, now, e, ideal, save);
    }
    println!("\ncompressed now      {} bytes, ratio {:.5}", comp_total, orig as f64/comp_total as f64);
    println!("ideal entropy saving {:.0} bytes = {:.2}% of output", total_save, 100.0*total_save/comp_total as f64);
    let newc = comp_total as f64 - total_save;
    println!("ratio would become  {:.5}   (T1.2 target 2.4500)", orig as f64 / newc);
    println!("\nTokens+offset-hi only (cheapest to code):");
    let partial = (ntok as f64 - ntok as f64*et/8.0) + (noff as f64 - noff as f64*eh/8.0);
    println!("  saving {:.0} bytes = {:.2}%, ratio -> {:.5}",
             partial, 100.0*partial/comp_total as f64, orig as f64/(comp_total as f64 - partial));
}
