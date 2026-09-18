fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb","reymont","samba","sao","webster","xml","x-ray"];
    let data: Vec<Vec<u8>> = files.iter().map(|f| std::fs::read(format!("corpus/{}", f)).unwrap()).collect();
    let mut out = Vec::new();
    let fast = std::env::args().any(|a| a == "--fast");
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() < 4.0 {
        for d in &data { out.clear(); if fast { simd_stream_codec::compress_into_fast(d, &mut out) } else { simd_stream_codec::compress_into(d, &mut out) } }
    }
    std::hint::black_box(out.len());
}
