// Exact Silesia compression ratio, no timing. Fast enough to sweep with.
// An optional argument restricts it to one corpus file. Per-file lines show
// how many blocks were stored raw or parsed by the dense retry.
use glyd::format::*;
fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let only = std::env::args().nth(1);
    let verbose = only.is_some() || std::env::args().any(|a| a == "-v");
    let dir = std::path::Path::new("corpus");
    let (mut o, mut c) = (0usize, 0usize);
    for f in &files {
        if let Some(ref w) = only { if w != f && w != "-v" { continue; } }
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let mut b = Vec::with_capacity(d.len());
        glyd::compress_into(&d, &mut b);
        let r = glyd::decompress(&b).expect("roundtrip");
        assert_eq!(r, d, "roundtrip mismatch on {}", f);
        if verbose {
            let (mut nb, mut raw, mut dense) = (0, 0, 0);
            let mut cur = 0;
            while cur + HEADER_SIZE <= b.len() {
                let h = unsafe { std::ptr::read_unaligned(b.as_ptr().add(cur) as *const BlockHeader) };
                nb += 1;
                if h.flags & FLAG_RAW_UNCOMPRESSED != 0 { raw += 1; }
                if h.flags & FLAG_DENSE != 0 { dense += 1; }
                cur += HEADER_SIZE + h.payload_len();
            }
            println!("{:<8} {:>9} -> {:>9}  ratio {:.5}  blocks {:>4} raw {:>3} dense {:>3}",
                     f, d.len(), b.len(), d.len() as f64 / b.len() as f64, nb, raw, dense);
        }
        o += d.len(); c += b.len();
    }
    println!("TOTAL {:.5}", o as f64 / c as f64);
}
