// The extended corpus (corpus/ext) at the ultra level against zstd -16 and
// -19: ratio and decode MB/s per file, one run.
use std::io::Write;
use std::time::Instant;
fn timed<F: FnMut()>(mut op: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 { let t = Instant::now(); op(); best = best.min(t.elapsed().as_secs_f64()); }
    best
}
fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus/ext").unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()).collect();
    files.sort();
    let filter = std::env::args().nth(1);
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if filter.as_ref().map_or(false, |f| !name.contains(f.as_str())) { continue; }
        let d = std::fs::read(&p).unwrap();
        let mut dst = vec![0u8; d.len() + 1024];
        let t = Instant::now();
        let mut g = Vec::new(); glyd::compress_into_ultra(&d, &mut g);
        let gc = t.elapsed().as_secs_f64();
        assert!(glyd::decompress(&g).unwrap() == d);
        let gd = timed(|| { let _ = glyd::decompress_into_raw(&g, &mut dst); });
        let mut gm = Vec::new(); glyd::compress_into_max(&d, &mut gm);
        print!("{name:24} {:>6.0} MB  Glyd --ultra {:.3} ({:.1} MB/s, decode {:.0} MB/s)  --max {:.3}", d.len() as f64 / 1e6, d.len() as f64 / g.len() as f64, d.len() as f64 / gc / 1e6, d.len() as f64 / gd / 1e6, d.len() as f64 / gm.len() as f64);
        for lvl in [16, 19] {
            let t = Instant::now();
            let mut enc = zstd::Encoder::new(Vec::new(), lvl).unwrap(); enc.write_all(&d).unwrap(); let z = enc.finish().unwrap();
            let zc = t.elapsed().as_secs_f64();
            let mut dec = zstd::bulk::Decompressor::new().unwrap();
            dec.set_parameter(zstd::zstd_safe::DParameter::WindowLogMax(27)).unwrap();
            let zd = timed(|| { let _ = dec.decompress_to_buffer(&z, &mut dst[..]); });
            print!("  zstd -{lvl} {:.3} ({:.1} MB/s, decode {:.0} MB/s)", d.len() as f64 / z.len() as f64, d.len() as f64 / zc / 1e6, d.len() as f64 / zd / 1e6);
        }
        println!();
    }
}
