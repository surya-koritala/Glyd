// Ratio and speed of the parallel paths (independent chunks) next to the
// sequential ones, per level, on Silesia.
use std::time::Instant;
fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let data: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let total: usize = data.iter().map(|d| d.len()).sum();
    let levels: [(&str, fn(&[u8], &mut Vec<u8>)); 6] = [
        ("default seq", glyd::compress_into), ("default par", glyd::compress_parallel_into),
        ("max seq", glyd::compress_into_max), ("max par", glyd::compress_parallel_into_max),
        ("ultra seq", glyd::compress_into_ultra), ("ultra par", glyd::compress_parallel_into_ultra),
    ];
    for (name, f) in levels {
        let t = Instant::now();
        let mut out = 0usize;
        for d in &data { let mut c = Vec::new(); f(d, &mut c); out += c.len(); }
        let s = t.elapsed().as_secs_f64();
        println!("{name:12} ratio {:.3}  {:7.1} MB/s", total as f64 / out as f64, total as f64 / s / 1e6);
    }
}
