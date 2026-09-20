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
            println!("{}: shape {:?}; plain {} vs record {} ({:+.1}%); transform {:.0} MB/s, inverse {:.0} MB/s", f, glyd::record::detect(d), a.len(), b.len(), 100.0 * (b.len() as f64 - a.len() as f64) / a.len() as f64, d.len() as f64 / ts / 1e6, d.len() as f64 / is / 1e6);
        }
        None => println!("{}: not record-shaped (plain {})", f, a.len()),
    }
}
