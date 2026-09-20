// Scratch: small objects with and without a trained dictionary (Glyd --max, zstd -3).
fn objects(path: &str, size: usize, n: usize, skip: usize) -> Vec<Vec<u8>> {
    let d = std::fs::read(path).unwrap();
    let mut v = Vec::new();
    let mut at = skip;
    while v.len() < n && at + size <= d.len() {
        // Whole lines, at least `size` bytes.
        let end = at + size;
        let end = d[end..].iter().position(|&b| b == b'\n').map_or(d.len(), |p| end + p + 1);
        v.push(d[at..end].to_vec());
        at = end;
    }
    v
}
fn main() {
    for (name, path, train_path, size) in [("nasa log records", "corpus/bench/nasa-access-jul95.log", "corpus/bench/train/nasa-access-aug95.log", 1024usize), ("github events", "corpus/bench/gharchive-2024-01-15-12.json", "corpus/bench/train/gharchive-2024-01-14-12.json", 4096)] {
        let objs = objects(path, size, 5000, 0);
        let train = objects(train_path, size, 20000, 0);
        let samples: Vec<&[u8]> = train.iter().map(|v| v.as_slice()).collect();
        let dict = glyd::Dict::train(&samples, 110 << 10);
        let zd = zstd::dict::from_samples(&samples, 110 << 10).unwrap();
        let raw: usize = objs.iter().map(|o| o.len()).sum();
        let (mut g0, mut g1, mut z0, mut z1) = (0usize, 0usize, 0usize, 0usize);
        let mut zc = zstd::bulk::Compressor::with_dictionary(3, &zd).unwrap();
        for o in &objs {
            let mut c = Vec::new(); glyd::compress_into_max(o, &mut c); g0 += c.len();
            let mut c = Vec::new(); glyd::compress_with_dict(&dict, o, &mut c); g1 += c.len();
            assert!(glyd::decompress_with_dict(&dict, &c).unwrap() == *o);
            z0 += zstd::bulk::compress(o, 3).unwrap().len();
            z1 += zc.compress(o).unwrap().len();
        }
        println!("{name}, {} objects of ~{size} B ({raw} B): Glyd --max alone {:.2}x, +Dict {:.2}x | zstd -3 alone {:.2}x, +dict {:.2}x", objs.len(), raw as f64 / g0 as f64, raw as f64 / g1 as f64, raw as f64 / z0 as f64, raw as f64 / z1 as f64);
    }
}
