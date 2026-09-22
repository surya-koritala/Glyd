// Scratch: a version against its base, and alone: bytes and times.
fn main() {
    let a = std::fs::read(std::env::args().nth(1).unwrap()).unwrap();
    let b = std::fs::read(std::env::args().nth(2).unwrap()).unwrap();
    let t = std::time::Instant::now();
    let mut d = Vec::new();
    glyd::compress_with_base(&a, &b, &mut d, false);
    let w = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let back = glyd::decompress_with_base(&a, &d).unwrap();
    let r = t.elapsed().as_secs_f64();
    assert!(back == b);
    println!("delta {} B of {} B: {:.2} s in ({:.0} MB/s), {:.2} s out ({:.0} MB/s), exact", d.len(), b.len(), w, b.len() as f64 / w / 1e6, r, b.len() as f64 / r / 1e6);
    let t = std::time::Instant::now();
    let mut c = Vec::new();
    glyd::compress_parallel_into_max(&b, &mut c);
    let w = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let back = glyd::decompress_parallel(&c).unwrap();
    let r = t.elapsed().as_secs_f64();
    assert!(back == b);
    println!("alone --max {} B: {:.2} s in ({:.0} MB/s), {:.2} s out ({:.0} MB/s), exact", c.len(), w, b.len() as f64 / w / 1e6, r, b.len() as f64 / r / 1e6);
}
