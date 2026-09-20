// Scratch: one decode of a .glyd file, several ways, to be timed by
// /usr/bin/time (user+sys): fresh buffer + checksum (the CLI today),
// kept buffer, no checksum, sequential.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if let Some(t) = std::env::var("GLYD_THREADS").ok().and_then(|v| v.parse().ok()) { glyd::set_threads(t); }
    let c = std::fs::read(&a[1]).unwrap();
    let n = glyd::decompressed_len(&c).unwrap();
    match a[2].as_str() {
        "fresh" => { let d = glyd::decompress_parallel(&c).unwrap(); assert_eq!(d.len(), n); }
        "fresh-seq" => { let d = glyd::decompress(&c).unwrap(); assert_eq!(d.len(), n); }
        "kept" => {
            let mut d = vec![0u8; n + 64];
            for b in d.chunks_mut(4096) { b[0] = 1; } // touch every page first
            let t = std::time::Instant::now();
            glyd::decompress_parallel_into(&c, &mut d).unwrap();
            eprintln!("decode only: {:.3} s", t.elapsed().as_secs_f64());
        }
        "kept-raw" => {
            let mut d = vec![0u8; n + 64];
            for b in d.chunks_mut(4096) { b[0] = 1; }
            let t = std::time::Instant::now();
            glyd::decompress_parallel_into_raw(&c, &mut d).unwrap();
            eprintln!("decode only: {:.3} s", t.elapsed().as_secs_f64());
        }
        "kept-seq-raw" => {
            let mut d = vec![0u8; n + 64];
            for b in d.chunks_mut(4096) { b[0] = 1; }
            let t = std::time::Instant::now();
            glyd::decompress_into_raw(&c, &mut d).unwrap();
            eprintln!("decode only: {:.3} s", t.elapsed().as_secs_f64());
        }
        _ => panic!(),
    }
}
