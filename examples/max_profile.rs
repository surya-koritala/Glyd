// Decode loop for profiling the max level: compresses every Silesia file
// with `--max`, then decodes them round-robin for `secs` seconds and
// prints MB/s per file and in total. `perf record` this.
//
// Usage: cargo run --release --example max_profile [secs=5] [file-filter]
use std::time::Instant;

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let filter = std::env::args().nth(2);
    let mut files: Vec<_> = std::fs::read_dir("corpus").expect("corpus/ (run scripts/download_corpus.sh)")
        .filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()).collect();
    files.sort();
    let mut set = Vec::new();
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("enwik") || filter.as_ref().map_or(false, |f| !name.contains(f)) { continue; }
        let d = std::fs::read(&p).unwrap();
        let mut b = Vec::new();
        glyd::compress_into_max(&d, &mut b);
        assert_eq!(glyd::decompress(&b).unwrap(), d);
        set.push((name, d.len(), b, vec![0u8; d.len()]));
    }
    let per = secs / set.len() as f64;
    let (mut tot_b, mut tot_s) = (0f64, 0f64);
    for (name, n, b, dst) in set.iter_mut() {
        let t = Instant::now();
        let mut k = 0usize;
        while t.elapsed().as_secs_f64() < per { glyd::decompress_into_raw(b, dst).unwrap(); k += 1; }
        let s = t.elapsed().as_secs_f64();
        println!("{name:12} {:8.1} MB/s  (ratio {:.3})", (*n as f64 * k as f64) / s / 1e6, *n as f64 / b.len() as f64);
        tot_b += *n as f64 * k as f64; tot_s += s;
    }
    println!("total        {:8.1} MB/s", tot_b / tot_s / 1e6);
}
