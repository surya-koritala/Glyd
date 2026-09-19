// Small objects (records cut from a corpus file) with and without a
// dictionary: ratio per object size for zstd -3 and -19 (zstd's trained
// dictionary) and Glyd --max and --ultra (Glyd's trained Dict), each
// dictionary 110 KB from 2,000 objects disjoint from the 2,000 measured.
use std::io::Read;
fn main() {
    let file = std::env::args().nth(1).unwrap_or("corpus/ext/gharchive.json".into());
    let take: usize = 256 << 20;
    let mut f = std::fs::File::open(&file).unwrap();
    let mut d = vec![0u8; take];
    let n = f.read(&mut d).unwrap();
    d.truncate(n);
    for &size in &[1024usize, 4096, 16384, 65536] {
        let objects: Vec<&[u8]> = d.chunks(size).take(2000).collect();
        // Dictionary trained on other objects from the same source: the
        // measured objects come from the first 128 MB, the samples from the
        // second (2,000 x 64 KB = 125 MB, so neither half runs into the other).
        let samples: Vec<&[u8]> = d[128 << 20..].chunks(size).take(2000).collect();
        let zdict = zstd::dict::from_samples(&samples, 110 * 1024).unwrap();
        // Glyd's own trainer on the same samples, the same content budget.
        let gdict = glyd::Dict::train(&samples, 110 * 1024);
        let gdict = &gdict;
        let raw: usize = objects.iter().map(|o| o.len()).sum();
        let mut z3 = 0; let mut z3d = 0; let mut z19d = 0; let mut gm = 0; let mut gmd = 0; let mut gu = 0; let mut gud = 0;
        let mut zc = zstd::bulk::Compressor::new(3).unwrap();
        let mut zcd = zstd::bulk::Compressor::with_dictionary(3, &zdict).unwrap();
        let mut zcd19 = zstd::bulk::Compressor::with_dictionary(19, &zdict).unwrap();
        for o in &objects {
            z3 += zc.compress(o).unwrap().len();
            z3d += zcd.compress(o).unwrap().len();
            z19d += zcd19.compress(o).unwrap().len();
            let mut b = Vec::new(); glyd::compress_into_max(o, &mut b); gm += b.len();
            let mut b = Vec::new(); glyd::compress_with_dict(gdict, o, &mut b); gmd += b.len();
            let mut b = Vec::new(); glyd::compress_into_ultra(o, &mut b); gu += b.len();
            let mut b = Vec::new(); glyd::compress_with_dict_ultra(gdict, o, &mut b); gud += b.len();
        }
        println!("{size:6} B objects: zstd -3 {:.2} | +dict {:.2} | zstd -19 +dict {:.2} || Glyd --max {:.2} | +Dict {:.2} || --ultra {:.2} | +Dict {:.2}  (Dict content {} B)",
            raw as f64 / z3 as f64, raw as f64 / z3d as f64, raw as f64 / z19d as f64, raw as f64 / gm as f64, raw as f64 / gmd as f64, raw as f64 / gu as f64, raw as f64 / gud as f64, gdict.content().len());
    }
}
