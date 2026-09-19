// Small objects with a prepared dictionary: ratio, compression and
// decompression throughput, Glyd --max against zstd -3, each with its own
// 110 KB dictionary trained on 2,000 objects disjoint from the 2,000
// measured. Both loops run at least a second and take the best of three
// passes. Glyd's output carries a 4-byte checksum per object; zstd's
// (its default) none.
use std::io::Read;
use std::time::Instant;

fn best_of_three(mut f: impl FnMut() -> usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let mut bytes = 0usize;
        let mut k = 0;
        while k < 3 || t.elapsed().as_secs_f64() < 1.0 {
            bytes += f();
            k += 1;
        }
        best = best.min(t.elapsed().as_secs_f64() / bytes as f64);
    }
    1.0 / best / 1e6
}

fn main() {
    let file = std::env::args().nth(1).unwrap_or("corpus/ext/gharchive.json".into());
    let mut f = std::fs::File::open(&file).unwrap();
    let mut d = vec![0u8; 256 << 20];
    let n = f.read(&mut d).unwrap();
    d.truncate(n);
    println!("{file}: 2,000 objects per size, dictionaries of 110 KB trained on 2,000 others");
    for &size in &[1024usize, 4096, 16384] {
        let objects: Vec<&[u8]> = d.chunks(size).take(2000).collect();
        let samples: Vec<&[u8]> = d[128 << 20..].chunks(size).take(2000).collect();
        let zdict = zstd::dict::from_samples(&samples, 110 * 1024).unwrap();
        let gdict = glyd::Dict::train(&samples, 110 * 1024);
        let raw: usize = objects.iter().map(|o| o.len()).sum();
        let mut buf = Vec::with_capacity(size + 1024);
        let mut out = vec![0u8; size + 64];
        // Glyd
        let gc: Vec<Vec<u8>> = objects.iter().map(|o| { let mut b = Vec::new(); glyd::compress_with_dict(&gdict, o, &mut b); b }).collect();
        let g_comp = best_of_three(|| { for o in &objects { buf.clear(); glyd::compress_with_dict(&gdict, o, &mut buf); } raw });
        let g_dec = best_of_three(|| { for c in &gc { glyd::decompress_with_dict_into(&gdict, c, &mut out).unwrap(); } raw });
        // zstd
        let mut zc = zstd::bulk::Compressor::with_dictionary(3, &zdict).unwrap();
        let zcv: Vec<Vec<u8>> = objects.iter().map(|o| zc.compress(o).unwrap()).collect();
        let z_comp = best_of_three(|| { for o in &objects { buf.clear(); zc.compress_to_buffer(o, &mut buf).unwrap(); } raw });
        let mut zd = zstd::bulk::Decompressor::with_dictionary(&zdict).unwrap();
        let z_dec = best_of_three(|| { for c in &zcv { zd.decompress_to_buffer(c, &mut out[..]).unwrap(); } raw });
        let gout: usize = gc.iter().map(|c| c.len()).sum();
        let zout: usize = zcv.iter().map(|c| c.len()).sum();
        println!("{size:6} B: Glyd --max+dict ratio {:.2} comp {:.0} MB/s decode {:.0} MB/s | zstd -3+dict ratio {:.2} comp {:.0} MB/s decode {:.0} MB/s",
            raw as f64 / gout as f64, g_comp, g_dec, raw as f64 / zout as f64, z_comp, z_dec);
    }
}
