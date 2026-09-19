fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let t = std::time::Instant::now();
    let m = glyd::ldm::Matches::find(&d, std::env::args().nth(2).is_some());
    let s = t.elapsed().as_secs_f64();
    let covered: usize = m.list.iter().map(|x| x.len as usize).sum();
    println!("{}: {} matches covering {:.1}% in {:.2} s ({:.0} MB/s)", f, m.list.len(), 100.0 * covered as f64 / d.len() as f64, s, d.len() as f64 / s / 1e6);
}
