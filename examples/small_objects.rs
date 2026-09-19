// Small objects (records cut from a corpus file) with and without a
// dictionary: total compressed bytes for zstd -3 (trained dictionary) and
// Glyd --max (the same bytes as a window dictionary), per object size.
use std::io::Read;
fn main() {
    let file = std::env::args().nth(1).unwrap_or("corpus/ext/gharchive.json".into());
    let take: usize = 64 << 20;
    let mut f = std::fs::File::open(&file).unwrap();
    let mut d = vec![0u8; take];
    let n = f.read(&mut d).unwrap();
    d.truncate(n);
    for &size in &[1024usize, 4096, 16384, 65536] {
        let objects: Vec<&[u8]> = d.chunks(size).take(2000).collect();
        // Dictionary trained on other objects from the same source.
        let samples: Vec<&[u8]> = d[32 << 20..].chunks(size).take(2000).collect();
        let zdict = zstd::dict::from_samples(&samples, 110 * 1024).unwrap();
        // Glyd's own trainer on the same samples, the same content budget.
        let gdict = glyd::Dict::train(&samples, 110 * 1024);
        let gdict = &gdict;
        let raw: usize = objects.iter().map(|o| o.len()).sum();
        let mut z3 = 0; let mut z3d = 0; let mut gm = 0; let mut gmd = 0; let mut gu = 0; let mut gud = 0;
        let mut zc = zstd::bulk::Compressor::new(3).unwrap();
        let mut zcd = zstd::bulk::Compressor::with_dictionary(3, &zdict).unwrap();
        for o in &objects {
            z3 += zc.compress(o).unwrap().len();
            z3d += zcd.compress(o).unwrap().len();
            let mut b = Vec::new(); glyd::compress_into_max(o, &mut b); gm += b.len();
            let mut b = Vec::new(); glyd::compress_with_dict(gdict, o, &mut b); gmd += b.len();
            let mut b = Vec::new(); glyd::compress_into_ultra(o, &mut b); gu += b.len();
            let mut b = Vec::new(); glyd::compress_with_dict_ultra(gdict, o, &mut b); gud += b.len();
        }
        println!("{size:6} B objects: zstd -3 {:.2} | +dict {:.2} || Glyd --max {:.2} | +dict {:.2} || --ultra {:.2} | +dict {:.2}  (dict {} B content)",
            raw as f64 / z3 as f64, raw as f64 / z3d as f64, raw as f64 / gm as f64, raw as f64 / gmd as f64, raw as f64 / gu as f64, raw as f64 / gud as f64, gdict.content().len());
    }
}
