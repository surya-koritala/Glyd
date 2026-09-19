// Compression loop of the max level over the first 256 MB of a file.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let d = &d[..d.len().min(256 << 20)];
    let mut c = Vec::with_capacity(d.len());
    let t = std::time::Instant::now();
    let mut k = 0;
    while t.elapsed().as_secs_f64() < 5.0 { c.clear(); glyd::compress_into_max(d, &mut c); k += 1; }
    println!("glyd max: ratio {:.3} compress {:.0} MB/s", d.len() as f64 / c.len() as f64, d.len() as f64 * k as f64 / t.elapsed().as_secs_f64() / 1e6);
}
