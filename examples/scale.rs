// Scratch: --max throughput by thread count.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    for t in [1usize, 2, 4, 8, 10] {
        glyd::set_threads(t);
        let mut c = Vec::new();
        let t0 = std::time::Instant::now();
        glyd::compress_parallel_into_max(&d, &mut c);
        let s = t0.elapsed().as_secs_f64();
        println!("{} threads: {:.0} MB/s ({:.0} per thread), {} B", t, d.len() as f64 / s / 1e6, d.len() as f64 / s / 1e6 / t as f64, c.len());
    }
}
