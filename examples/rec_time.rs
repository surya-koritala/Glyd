// Times the record transform and its inverse on a file.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let t = std::time::Instant::now();
    let img = glyd::record::transform(&d).expect("record-shaped");
    let tt = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let back = glyd::record::inverse(&img).unwrap();
    let ti = t.elapsed().as_secs_f64();
    assert!(back == d);
    println!("{}: transform {:.0} MB/s ({:.2} s), inverse {:.0} MB/s ({:.2} s), image {} of {}", f, d.len() as f64 / tt / 1e6, tt, d.len() as f64 / ti / 1e6, ti, img.len(), d.len());
}
