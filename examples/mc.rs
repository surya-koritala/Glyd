// Multi-core decode on Silesia: parallel-compressed (independent 256 KB
// blocks), decompress_parallel_into_raw, best of 7.
use std::time::Instant;
fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb","reymont","samba","sao","webster","xml","x-ray"];
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    let (mut tb, mut tt, mut tt1) = (0usize, 0.0f64, 0.0f64);
    println!("{:<8} {:>6} {:>10} {:>10}", "file", "ratio", "1C GB/s", "MC GB/s");
    for f in &files {
        let d = std::fs::read(format!("corpus/{}", f)).unwrap();
        let c = glyd::compress_parallel(&d);
        let mut o = vec![0u8; d.len() + 4096];
        let best = |f: &mut dyn FnMut()| { let mut b = f64::MAX; for _ in 0..7 { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } b };
        let t1 = best(&mut || { glyd::decompress_into_raw(&c, &mut o).unwrap(); });
        let tm = best(&mut || { glyd::decompress_parallel_into_raw(&c, &mut o).unwrap(); });
        assert_eq!(&o[..d.len()], &d[..]);
        println!("{:<8} {:>6.3} {:>10.2} {:>10.2}", f, d.len() as f64 / c.len() as f64, d.len() as f64 / gb / t1, d.len() as f64 / gb / tm);
        tb += d.len(); tt += tm; tt1 += t1;
    }
    println!("{:<8} {:>6} {:>10.2} {:>10.2}", "TOTAL", "", tb as f64 / gb / tt1, tb as f64 / gb / tt);
    println!("threads: {}", rayon::current_num_threads());
}
