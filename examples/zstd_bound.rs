// What zstd's higher levels reach on Silesia inside a 2 MB window: an
// upper-bound proxy for an optimal parse in the v7 format, whose codes
// and tables mirror zstd's sequence model. Prints ratio and MB/s.
//
// Usage: cargo run --release --example zstd_bound [levels...]
use std::io::Write;
use std::time::Instant;

fn main() {
    let levels: Vec<i32> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let levels = if levels.is_empty() { vec![3, 9, 12, 16, 19] } else { levels };
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let data: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let total: usize = data.iter().map(|d| d.len()).sum();
    for &lvl in &levels {
        for wlog in [21u32, 27] {
            let t = Instant::now();
            let mut out_total = 0usize;
            for d in &data {
                let mut enc = zstd::Encoder::new(Vec::new(), lvl).unwrap();
                enc.set_parameter(zstd::zstd_safe::CParameter::WindowLog(wlog)).unwrap();
                enc.write_all(d).unwrap();
                out_total += enc.finish().unwrap().len();
            }
            let s = t.elapsed().as_secs_f64();
            println!("zstd -{lvl:<2} wlog {wlog}: ratio {:.3}  {:6.1} MB/s", total as f64 / out_total as f64, total as f64 / s / 1e6);
        }
    }
}
