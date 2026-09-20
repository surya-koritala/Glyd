// Scratch: record transform on a slice, image size through --max vs plain.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let d = &d[..d.len().min(64 << 20)];
    let d = &d[..d.iter().rposition(|&b| b == b'\n').map_or(d.len(), |p| p + 1)];
    let t = std::time::Instant::now();
    let img = glyd::record::transform(d);
    let ts = t.elapsed().as_secs_f64();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    glyd::compress_into_max(d, &mut a);
    match img {
        Some(img) => {
            glyd::compress_into_max(&img, &mut b);
            let t = std::time::Instant::now();
            let back = glyd::record::inverse(&img).unwrap();
            let is = t.elapsed().as_secs_f64();
            assert!(back == d);
            if std::env::args().nth(2).is_some() { anatomy(&img); }
            println!("{}: shape {:?}; plain {} vs record {} ({:+.1}%); transform {:.0} MB/s, inverse {:.0} MB/s", f, glyd::record::detect(d), a.len(), b.len(), 100.0 * (b.len() as f64 - a.len() as f64) / a.len() as f64, d.len() as f64 / ts / 1e6, d.len() as f64 / is / 1e6);
        }
        None => println!("{}: not record-shaped (plain {})", f, a.len()),
    }
}

/// Per-stream anatomy of a record image: type, raw and --max size.
pub fn anatomy(img: &[u8]) {
    let mut pos = 10usize;
    let mut varint = |pos: &mut usize| { let (mut v, mut sh) = (0u64, 0); loop { let b = img[*pos]; *pos += 1; v |= ((b & 127) as u64) << sh; if b < 128 { return v; } sh += 7; } };
    let mode = img[8];
    let fields = varint(&mut pos) as usize;
    let _lines = varint(&mut pos);
    pos += 1;
    let types = img[pos..pos + fields].to_vec();
    pos += fields;
    let n = varint(&mut pos) as usize;
    let mut lens = Vec::new();
    for _ in 0..n { lens.push(varint(&mut pos) as usize); }
    let names = ["text", "int", "dict", "time", "dict8", "dec"];
    let per_type = [1usize, 1, 3, 3, 2, 3];
    let mut streams = Vec::new();
    for &l in &lens { streams.push(&img[pos..pos + l]); pos += l; }
    let fixed = match mode { 1 => 2, 2 => 1, 3 => 3, _ => 0 };
    println!("   mode {} types {:?} stream lengths {:?}", mode, types, lens);
    let cost = |s: &[u8]| { let mut c = Vec::new(); glyd::compress_into_max(s, &mut c); c.len() };
    let mut si = 0;
    for k in 0..fixed { println!("   frame/order stream {}: {:>10} -> {:>9}", k, streams[si].len(), cost(streams[si])); si += 1; }
    for (i, &t) in types.iter().enumerate() {
        let k = per_type[t as usize];
        let raw: usize = streams[si..si + k].iter().map(|s| s.len()).sum();
        let c: usize = streams[si..si + k].iter().map(|s| cost(s)).sum();
        println!("   column {:>3} {:5}: {:>10} -> {:>9}", i, names[t as usize], raw, c);
        si += k;
    }
}
