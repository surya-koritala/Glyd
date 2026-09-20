// Scratch: loop the record rebuild of a file for sampling.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let d = &d[..d.len().min(64 << 20)];
    let d = &d[..d.iter().rposition(|&b| b == b'\n').map_or(d.len(), |p| p + 1)];
    let img = glyd::record::transform(d).unwrap();
    let mut out = Vec::new();
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() < 12.0 { glyd::record::inverse_into(&img, &mut out).unwrap(); }
}
