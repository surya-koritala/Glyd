// Scratch: what a shape dictionary could give small objects. For objects
// of ~S bytes cut from a file: zstd -3 + trained dict, Glyd --max + Dict,
// and the ideal shape-dictionary cost = what the object adds to a 4 MB
// record-mode stream of the same data (plus 24 bytes of framing).
fn objects(d: &[u8], size: usize, n: usize, skip: usize) -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    let mut at = skip;
    while v.len() < n && at + size <= d.len() {
        let end = at + size;
        let end = d[end..].iter().position(|&b| b == b'\n').map_or(d.len(), |p| end + p + 1);
        v.push(d[at..end].to_vec());
        at = end;
    }
    v
}
fn rec_max(d: &[u8]) -> usize {
    let mut c = Vec::new();
    match glyd::record::transform(d) {
        Some(img) => glyd::compress_into_max(&img, &mut c),
        None => glyd::compress_into_max(d, &mut c),
    }
    c.len()
}
fn main() {
    let n_obj = 400usize;
    for (name, path, size) in [
        ("nasa log", "corpus/bench/nasa-access-jul95.log", 1024usize),
        ("nasa log", "corpus/bench/nasa-access-jul95.log", 4096),
        ("alibaba csv", "corpus/ext2/alibaba_machine_usage.csv", 1024),
        ("alibaba csv", "corpus/ext2/alibaba_machine_usage.csv", 4096),
        ("alibaba json lines", "corpus/ext2/alibaba_machine_usage.jsonl", 1024),
        ("alibaba json lines", "corpus/ext2/alibaba_machine_usage.jsonl", 4096),
        ("taxi csv", "corpus/ext2/yellow_tripdata_2024-02.csv", 1024),
        ("HDFS log", "corpus/ext2/HDFS.log", 1024),
        ("HDFS log", "corpus/ext2/HDFS.log", 4096),
    ] {
        let d = std::fs::read(path).unwrap();
        let d = &d[..d.len().min(64 << 20)];
        // Training data: the first 8 MB (dictionaries) / 4 MB (the record stream); objects from the second half.
        let train = objects(&d[..8 << 20], size, 8000, 0);
        let samples: Vec<&[u8]> = train.iter().map(|v| v.as_slice()).collect();
        let dict = glyd::Dict::train(&samples, 110 << 10);
        let zd = zstd::dict::from_samples(&samples, 110 << 10).unwrap();
        let base = &d[..4 << 20];
        let base = &base[..base.iter().rposition(|&b| b == b'\n').map_or(base.len(), |p| p + 1)];
        let base_cost = rec_max(base);
        let objs = objects(d, size, n_obj, d.len() / 2);
        let raw: usize = objs.iter().map(|o| o.len()).sum();
        let (mut g1, mut z1, mut sh) = (0usize, 0usize, 0usize);
        let mut zc = zstd::bulk::Compressor::with_dictionary(3, &zd).unwrap();
        let mut joined = base.to_vec();
        let mut batch = Vec::new();
        let t = std::time::Instant::now();
        let shape = glyd::ShapeDict::train(&d[..4 << 20]).expect("record-shaped");
        let train_s = t.elapsed().as_secs_f64();
        let shape = glyd::ShapeDict::from_bytes(&shape.to_bytes()).unwrap();
        let (mut enc_s, mut dec_s) = (0.0f64, 0.0f64);
        for o in &objs {
            let mut c = Vec::new(); glyd::compress_with_dict(&dict, o, &mut c); g1 += c.len();
            z1 += zc.compress(o).unwrap().len();
            let t = std::time::Instant::now();
            let mut c = Vec::new(); shape.compress(o, &mut c); sh += c.len();
            enc_s += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            assert!(shape.decompress(&c).unwrap() == *o, "shape round trip");
            dec_s += t.elapsed().as_secs_f64();
            joined.extend_from_slice(o);
            batch.extend_from_slice(o);
        }
        println!("{name:>18} ~{size:>4} B: ShapeDict {:.2}x ({:.2}x over zstd+dict; dict {} KB, trained in {:.1} s; {:.0} MB/s enc, {:.0} MB/s dec)", raw as f64 / sh as f64, z1 as f64 / sh as f64, shape.to_bytes().len() >> 10, train_s, raw as f64 / enc_s / 1e6, raw as f64 / dec_s / 1e6);
        // All the objects added to the record stream at once: what the
        // shape dictionary would let each cost, plus 24 B of framing each.
        let ideal = rec_max(&joined) as i64 - base_cost as i64 + 24 * objs.len() as i64;
        let batched = rec_max(&batch);
        // The objects packed: 1 MB packs at --max, one object read back from each.
        let refs: Vec<&[u8]> = objs.iter().map(|v| v.as_slice()).collect();
        let per_pack = (1 << 20) / size;
        let (mut packed, mut read_s) = (0usize, 0.0f64);
        let t = std::time::Instant::now();
        let packs: Vec<Vec<u8>> = refs.chunks(per_pack).map(|group| { let mut p = Vec::new(); glyd::compress_pack(group, &mut p, glyd::compress_into_max); p }).collect();
        let pack_s = t.elapsed().as_secs_f64();
        for (k, p) in packs.iter().enumerate() {
            packed += p.len();
            let t = std::time::Instant::now();
            let o = glyd::decompress_pack_object(p, 0).unwrap();
            read_s += t.elapsed().as_secs_f64();
            assert!(o == objs[k * per_pack], "pack object");
            let all = glyd::decompress_pack(p).unwrap();
            assert!(all.iter().zip(&objs[k * per_pack..]).all(|(a, b)| a == b), "pack contents");
        }
        println!("{name:>18} ~{size:>4} B: packed ({} per 1 MB pack, --max) {:.2}x ({:.2}x over zstd+dict); {:.0} MB/s to pack, one object read in {:.2} ms", per_pack, raw as f64 / packed as f64, z1 as f64 / packed as f64, raw as f64 / pack_s / 1e6, read_s / packs.len() as f64 * 1e3);
        println!("{name:>18} ~{size:>4} B x {}: zstd -3 + dict {:.2}x | Glyd --max + Dict {:.2}x | shape dict (ideal) {:.2}x  -> {:.2}x over zstd+dict | the objects as one record stream {:.2}x", objs.len(), raw as f64 / z1 as f64, raw as f64 / g1 as f64, raw as f64 / ideal.max(1) as f64, z1 as f64 / ideal.max(1) as f64, raw as f64 / batched as f64);
    }
}
