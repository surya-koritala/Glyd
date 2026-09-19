// Compress loop over small objects with a dictionary, for profiling.
use std::io::Read;
fn main() {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let mut f = std::fs::File::open("corpus/ext/gharchive.json").unwrap();
    let mut d = vec![0u8; 256 << 20];
    let n = f.read(&mut d).unwrap(); d.truncate(n);
    let objects: Vec<&[u8]> = d.chunks(size).take(2000).collect();
    let samples: Vec<&[u8]> = d[128 << 20..].chunks(size).take(2000).collect();
    let dict = glyd::Dict::train(&samples, 110 * 1024);
    let mut out = Vec::with_capacity(size + 1024);
    let t = std::time::Instant::now();
    let mut k = 0u64;
    while t.elapsed().as_secs_f64() < 8.0 { for o in &objects { out.clear(); glyd::compress_with_dict(&dict, o, &mut out); } k += 1; }
    let bytes: usize = objects.iter().map(|o| o.len()).sum();
    println!("{} MB/s ({:.2} us per object)", (bytes as f64 * k as f64 / t.elapsed().as_secs_f64() / 1e6) as u64, t.elapsed().as_secs_f64() / (k as f64 * objects.len() as f64) * 1e6);
}
