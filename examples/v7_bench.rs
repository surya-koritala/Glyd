// Format v7 (max level) on Silesia: ratio, compression GB/s, decode GB/s,
// with zstd -3 and zstd -1 measured in the same run under the same
// protocol (quick3's `timed`: median over `runs` of a >= `min_s` loop).
// Verifies the v7 round trip on every file first.
//
// Usage: cargo run --release --example v7_bench [runs=3] [min_s=0.3]
use std::time::Instant;
const GB: f64 = 1024.0 * 1024.0 * 1024.0;

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

/// One codec's running totals: input bytes, output bytes, compress and
/// decompress seconds.
#[derive(Default, Clone, Copy)]
struct Tot { o: usize, c: usize, ct: f64, dt: f64 }

impl Tot {
    fn add(&mut self, o: usize, c: usize, ct: f64, dt: f64) { self.o += o; self.c += c; self.ct += ct; self.dt += dt; }
    fn line(&self) -> String {
        format!("ratio {:.4} comp {:.3} GB/s decomp {:.3} GB/s",
                self.o as f64 / self.c as f64, (self.o as f64 / GB) / self.ct, (self.o as f64 / GB) / self.dt)
    }
}

fn main() {
    let runs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let min_s: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let (mut v7, mut z3, mut z1) = (Tot::default(), Tot::default(), Tot::default());
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let mut dst = vec![0u8; d.len() + 1024];

        let mut b = Vec::with_capacity(d.len());
        simd_stream_codec::compress_into_max(&d, &mut b);
        assert_eq!(simd_stream_codec::decompress(&b).expect("v7 decode failed"), d, "v7 roundtrip mismatch on {}", f);
        let fc = timed(runs, min_s, || { b.clear(); simd_stream_codec::compress_into_max(&d, &mut b); });
        let fd = timed(runs, min_s, || { let _ = simd_stream_codec::decompress_into_raw(&b, &mut dst); });
        v7.add(d.len(), b.len(), fc, fd);
        print!("  {:<8} v7 {:<50}", f, Tot { o: d.len(), c: b.len(), ct: fc, dt: fd }.line());

        for (level, tot) in [(3, &mut z3), (1, &mut z1)] {
            let mut comp = zstd::bulk::Compressor::new(level).unwrap();
            let mut dec = zstd::bulk::Decompressor::new().unwrap();
            let mut zb = vec![0u8; zstd::zstd_safe::compress_bound(d.len())];
            let n = comp.compress_to_buffer(&d, &mut zb[..]).unwrap();
            let z = zb[..n].to_vec();
            let zc = timed(runs, min_s, || { let _ = comp.compress_to_buffer(&d, &mut zb[..]); });
            let zd = timed(runs, min_s, || { let _ = dec.decompress_to_buffer(&z, &mut dst[..]); });
            tot.add(d.len(), z.len(), zc, zd);
            print!(" | zstd-{} {:<50}", level, Tot { o: d.len(), c: z.len(), ct: zc, dt: zd }.line());
        }
        println!();
    }
    println!("v7 total: {} | zstd-3 {} | zstd-1 {}", v7.line(), z3.line(), z1.line());
}
