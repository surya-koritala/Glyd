// Scratch: the context-mixing coder on a slice, size and speed, exact.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let n: usize = std::env::args().nth(2).map_or(16 << 20, |s| s.parse().unwrap());
    let d = std::fs::read(&f).unwrap();
    let d = &d[..d.len().min(n)];
    let t = std::time::Instant::now();
    let mut c = Vec::new();
    glyd::cm::encode(d, &mut c);
    let ct = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let mut back = vec![0u8; d.len()];
    glyd::cm::decode(&c, &mut back);
    let dt = t.elapsed().as_secs_f64();
    assert!(back == d, "round trip differs");
    let mut z = Vec::new();
    glyd::compress_into_ultra(d, &mut z);
    println!("{}: {} B -> cm {} B ({:.2}x) {:.2} MB/s enc, {:.2} MB/s dec; --ultra {} B ({:.2}x); cm/ultra {:.3}", f.rsplit('/').next().unwrap(), d.len(), c.len(), d.len() as f64 / c.len() as f64, d.len() as f64 / ct / 1e6, d.len() as f64 / dt / 1e6, z.len(), d.len() as f64 / z.len() as f64, c.len() as f64 / z.len() as f64);
}
