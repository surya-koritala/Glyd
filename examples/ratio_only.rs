// Exact Silesia compression ratio, no timing. Fast enough to sweep with.
fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let (mut o, mut c) = (0usize, 0usize);
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let mut b = Vec::with_capacity(d.len());
        simd_stream_codec::compress_into(&d, &mut b);
        let r = simd_stream_codec::decompress(&b).expect("roundtrip");
        assert_eq!(r, d, "roundtrip mismatch on {}", f);
        o += d.len(); c += b.len();
    }
    println!("TOTAL {:.5}", o as f64 / c as f64);
}
