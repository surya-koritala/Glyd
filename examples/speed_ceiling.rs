// How fast can a decompressor physically go on this machine, single core?
//
// A decoder must write every output byte, so memcpy of the output is a hard
// ceiling no codec can beat; memset is the pure store ceiling. Measured next
// to liblz4 (the industry bar) and our decoder, on Silesia and on
// cache-resident buffers, so "how far can this go" is a number.
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

/// Median of `runs` measurements, each averaged over at least `min_s` seconds.
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
    pin(4);
    let runs = 5;
    let min_s = 0.5;

    // 1. Cache-level store ceilings: memset/memcpy of buffers that fit L1, L2, L3, DRAM.
    println!("Store ceilings by working set (single core, pinned):");
    println!("{:>10}  {:>12}  {:>12}", "size", "memset GB/s", "memcpy GB/s");
    for &sz in &[16usize << 10, 512 << 10, 8 << 20, 64 << 20, 212 << 20] {
        let src = vec![0x5Au8; sz];
        let mut dst = vec![0u8; sz];
        let ms = timed(runs, min_s, || { dst.iter_mut().for_each(|b| *b = 0x3C); std::hint::black_box(&dst); });
        let mc = timed(runs, min_s, || { dst.copy_from_slice(&src); std::hint::black_box(&dst); });
        let label = if sz >= 1 << 20 { format!("{} MB", sz >> 20) } else { format!("{} KB", sz >> 10) };
        println!("{:>10}  {:>12.1}  {:>12.1}", label, (sz as f64 / GB) / ms, (sz as f64 / GB) / mc);
    }

    // 2. Silesia: memcpy ceiling vs liblz4 vs us, per file and total.
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    println!();
    println!("Silesia decode, single core: memcpy ceiling vs liblz4 vs Alatirok");
    println!("{:<8} {:>12} {:>12} {:>12}   {:>10} {:>10}", "file", "memcpy GB/s", "lz4 GB/s", "ours GB/s", "lz4 ratio", "our ratio");
    let (mut tot_o, mut t_mc, mut t_lz, mut t_us) = (0usize, 0.0, 0.0, 0.0);
    let (mut c_lz, mut c_us) = (0usize, 0usize);
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let len = d.len();
        let mut dst = vec![0u8; len + 1024];

        let mc = timed(runs, min_s, || { dst[..len].copy_from_slice(&d); std::hint::black_box(&dst); });

        let bound = lz4::block::compress_bound(len).unwrap_or(len * 2 + 64);
        let mut lb = vec![0u8; bound];
        let n = lz4::block::compress_to_buffer(&d, None, false, &mut lb).unwrap();
        let lz4c = lb[..n].to_vec();
        let lz = timed(runs, min_s, || { let _ = lz4::block::decompress_to_buffer(&lz4c, Some(len as i32), &mut dst); });

        let mut ours = Vec::with_capacity(len);
        simd_stream_codec::compress_into(&d, &mut ours);
        let us = timed(runs, min_s, || { let _ = simd_stream_codec::decompress_into_raw(&ours, &mut dst); });

        println!("{:<8} {:>12.2} {:>12.2} {:>12.2}   {:>10.3} {:>10.3}",
                 f, (len as f64 / GB) / mc, (len as f64 / GB) / lz, (len as f64 / GB) / us,
                 len as f64 / lz4c.len() as f64, len as f64 / ours.len() as f64);
        tot_o += len; t_mc += mc; t_lz += lz; t_us += us; c_lz += lz4c.len(); c_us += ours.len();
    }
    let o = tot_o as f64 / GB;
    println!("{:<8} {:>12.2} {:>12.2} {:>12.2}   {:>10.3} {:>10.3}",
             "TOTAL", o / t_mc, o / t_lz, o / t_us, tot_o as f64 / c_lz as f64, tot_o as f64 / c_us as f64);
    println!();
    println!("memcpy is the ceiling for any decoder that writes its output: {:.2} GB/s here.", o / t_mc);
    println!("liblz4 reaches {:.0}% of it; Alatirok {:.0}%.", 100.0 * t_mc / t_lz, 100.0 * t_mc / t_us);
}
