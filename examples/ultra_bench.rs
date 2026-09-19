// The ultra level on Silesia next to the max level: ratio, compression
// MB/s and decode MB/s per file, round trip verified.
//
// Usage: cargo run --release --example ultra_bench [file-filter]
use std::time::Instant;

fn main() {
    let filter = std::env::args().nth(1);
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let (mut tin, mut tmax, mut tultra, mut tc, mut td, mut tdm) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    println!("{:12} {:>8} {:>8} {:>10} {:>10} {:>10}", "file", "max", "ultra", "comp MB/s", "dec max", "dec ultra");
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if filter.as_ref().map_or(false, |f| !name.contains(f)) { continue; }
        let d = std::fs::read(&p).unwrap();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        glyd::compress_into_max(&d, &mut a);
        let t = Instant::now();
        glyd::compress_into_ultra(&d, &mut b);
        let cs = t.elapsed().as_secs_f64();
        assert_eq!(glyd::decompress(&b).unwrap(), d, "ultra round trip failed on {name}");
        let mut dst = vec![0u8; d.len()];
        let dec = |c: &[u8], dst: &mut [u8]| {
            let t = Instant::now();
            let mut k = 0;
            while t.elapsed().as_secs_f64() < 0.4 { glyd::decompress_into_raw(c, dst).unwrap(); k += 1; }
            d.len() as f64 * k as f64 / t.elapsed().as_secs_f64() / 1e6
        };
        let (dm, du) = (dec(&a, &mut dst), dec(&b, &mut dst));
        println!("{name:12} {:8.3} {:8.3} {:10.1} {:10.0} {:10.0}", d.len() as f64 / a.len() as f64, d.len() as f64 / b.len() as f64, d.len() as f64 / cs / 1e6, dm, du);
        tin += d.len() as f64; tmax += a.len() as f64; tultra += b.len() as f64; tc += cs; td += d.len() as f64 / (du * 1e6); tdm += d.len() as f64 / (dm * 1e6);
    }
    println!("{:12} {:8.3} {:8.3} {:10.1} {:10.0} {:10.0}", "total", tin / tmax, tin / tultra, tin / tc / 1e6, tin / tdm / 1e6, tin / td / 1e6);
}
