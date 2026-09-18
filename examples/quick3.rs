// Fast three-axis check on Silesia: ratio, compression GB/s, decompression GB/s.
// Verifies round-trip. Appends durably to quick3_log.csv (this host crashes).
use std::io::Write;
use std::time::Instant;
const GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[cfg(target_os = "linux")]
fn pin(c: usize) { unsafe {
    let mut s: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut s); libc::CPU_SET(c, &mut s);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
}}
#[cfg(not(target_os = "linux"))]
fn pin(_c: usize) {}

fn timed<F: FnMut()>(runs: usize, min_s: f64, mut op: F) -> f64 {
    let mut v = Vec::new();
    for _ in 0..runs {
        op();
        let t = Instant::now();
        let mut n = 0;
        loop { op(); n += 1;
            let e = t.elapsed().as_secs_f64();
            if e >= min_s && n >= 2 { v.push(e / n as f64); break; } }
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let label = std::env::args().nth(1).unwrap_or_else(|| "unlabelled".into());
    let runs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let min_s: f64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let verbose = std::env::args().any(|a| a == "-v");
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let (mut o, mut c, mut ct, mut dt) = (0usize, 0usize, 0.0f64, 0.0f64);
    pin(4);
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let mut b = Vec::with_capacity(d.len());
        simd_stream_codec::compress_into(&d, &mut b);
        let r = simd_stream_codec::decompress(&b).expect("roundtrip failed");
        assert_eq!(r, d, "roundtrip mismatch on {}", f);
        let mut dst = vec![0u8; d.len() + 1024];
        let fc = timed(runs, min_s, || { let mut x = Vec::with_capacity(d.len());
                                          simd_stream_codec::compress_into(&d, &mut x); });
        let fd = timed(runs, min_s, || { let _ = simd_stream_codec::decompress_into_raw(&b, &mut dst); });
        ct += fc; dt += fd;
        if verbose {
            println!("  {:<8} ratio {:.4}  comp {:.3} GB/s ({:.1} ms)  decomp {:.3} GB/s ({:.1} ms)", f,
                     d.len() as f64 / b.len() as f64, (d.len() as f64 / GB) / fc, fc * 1e3, (d.len() as f64 / GB) / fd, fd * 1e3);
        }
        o += d.len(); c += b.len();
    }
    let ratio = o as f64 / c as f64;
    let cg = (o as f64 / GB) / ct;
    let dg = (o as f64 / GB) / dt;
    // LZAV targets from Phase 0: ratio 2.4500, comp 0.489, decomp 3.130
    println!("{:<22} ratio {:.5} ({:.1}% of T1.2)  comp {:.4} ({:.1}% of T1.3)  decomp {:.4} ({})",
             label, ratio, 100.0*ratio/2.45, cg, 100.0*cg/0.489, dg,
             if dg >= 3.130 { "T1.4 OK" } else { "T1.4 FAIL" });
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open("quick3_log.csv") {
        let _ = writeln!(fh, "{},{:.5},{:.5},{:.5}", label, ratio, cg, dg);
        let _ = fh.flush();
    }
}
