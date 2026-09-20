fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = std::fs::read(&a[1]).unwrap();
    let size: usize = a.get(2).map_or(1024, |s| s.parse().unwrap());
    let shape = glyd::ShapeDict::train(&d[..4 << 20]).expect("record-shaped");
    println!("{}", shape.describe());
    let mut at = d.len() / 2;
    let (mut raw, mut img, mut comp, mut n) = (0, 0, 0, 0);
    let mut cols: Vec<usize> = Vec::new();
    while n < 400 {
        let end = at + size;
        let end = d[end..].iter().position(|&b| b == b'\n').map_or(d.len(), |p| end + p + 1);
        let o = &d[at..end];
        let i = shape.image_bytes(o);
        let mut c = Vec::new(); shape.compress(o, &mut c);
        raw += o.len(); img += i.len(); comp += c.len(); n += 1;
        let cb = shape.column_bytes(o);
        if cols.is_empty() { cols = vec![0; cb.len()]; }
        for (k, v) in cb.iter().enumerate() { cols[k] += v; }
        at = end;
    }
    println!("{} objects: raw {} -> image {} ({:.2}x) -> compressed {} ({:.2}x)", n, raw, img, raw as f64 / img as f64, comp, raw as f64 / comp as f64);
    println!("image bytes by column: {:?} (rows/frames: {})", cols, img - cols.iter().sum::<usize>());
}
