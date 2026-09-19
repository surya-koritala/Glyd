// Small objects: compress and decompress throughput with a dictionary,
// Glyd --max against zstd -3 (prepared decoder dictionaries on both
// sides), 2,000 objects per size.
use std::io::Read;
use std::time::Instant;
fn main() {
    let file = std::env::args().nth(1).unwrap_or("corpus/ext/gharchive.json".into());
    let mut f = std::fs::File::open(&file).unwrap();
    let mut d = vec![0u8; 64 << 20];
    let n = f.read(&mut d).unwrap();
    d.truncate(n);
    for &size in &[1024usize, 4096, 16384] {
        let objects: Vec<&[u8]> = d.chunks(size).take(2000).collect();
        let samples: Vec<&[u8]> = d[32 << 20..].chunks(size).take(2000).collect();
        let zdict = zstd::dict::from_samples(&samples, 110 * 1024).unwrap();
        let gdict = glyd::Dict::train(&samples, 110 * 1024);
        let raw: usize = objects.iter().map(|o| o.len()).sum();
        // Glyd
        let t = Instant::now();
        let gc: Vec<Vec<u8>> = objects.iter().map(|o| { let mut b = Vec::new(); glyd::compress_with_dict(&gdict, o, &mut b); b }).collect();
        let gcs = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut k = 0;
        while t.elapsed().as_secs_f64() < 1.0 { for c in &gc { let _ = glyd::decompress_with_dict(&gdict, c).unwrap(); } k += 1; }
        let gds = t.elapsed().as_secs_f64() / k as f64;
        // zstd
        let mut zc = zstd::bulk::Compressor::with_dictionary(3, &zdict).unwrap();
        let t = Instant::now();
        let zcv: Vec<Vec<u8>> = objects.iter().map(|o| zc.compress(o).unwrap()).collect();
        let zcs = t.elapsed().as_secs_f64();
        let mut zd = zstd::bulk::Decompressor::with_dictionary(&zdict).unwrap();
        let t = Instant::now();
        let mut k = 0;
        while t.elapsed().as_secs_f64() < 1.0 { for c in &zcv { let _ = zd.decompress(c, size + 64).unwrap(); } k += 1; }
        let zds = t.elapsed().as_secs_f64() / k as f64;
        let gout: usize = gc.iter().map(|c| c.len()).sum();
        let zout: usize = zcv.iter().map(|c| c.len()).sum();
        println!("{size:6} B: Glyd --max+dict ratio {:.2} comp {:.0} MB/s decode {:.0} MB/s | zstd -3+dict ratio {:.2} comp {:.0} MB/s decode {:.0} MB/s",
            raw as f64 / gout as f64, raw as f64 / gcs / 1e6, raw as f64 / gds / 1e6, raw as f64 / zout as f64, raw as f64 / zcs / 1e6, raw as f64 / zds / 1e6);
    }
}
