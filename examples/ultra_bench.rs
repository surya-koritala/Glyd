// The ultra level on Silesia next to the max level and zstd's high
// levels, measured in one run under one protocol: decode as quick3's
// `timed` (median over `runs` of a >= `min_s` loop); compression timed
// once per file (these levels run at single-digit MB/s). zstd -16 and
// -19 are run at their default window and inside Glyd's 2 MB one
// (wlog 21). Every round trip is verified.
//
// Usage: cargo run --release --example ultra_bench [runs=3] [min_s=0.3] [file-filter]
use std::io::Write;
use std::time::Instant;

fn timed<F: FnMut()>(runs: usize, min_s: f64, mut op: F) -> f64 {
    let mut v = Vec::new();
    for _ in 0..runs {
        op();
        let t = Instant::now();
        let mut n = 0;
        loop {
            op();
            n += 1;
            let e = t.elapsed().as_secs_f64();
            if e >= min_s && n >= 2 {
                v.push(e / n as f64);
                break;
            }
        }
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Default, Clone, Copy)]
struct Tot {
    o: usize,
    c: usize,
    ct: f64,
    dt: f64,
}

impl Tot {
    fn add(&mut self, o: usize, c: usize, ct: f64, dt: f64) {
        self.o += o;
        self.c += c;
        self.ct += ct;
        self.dt += dt;
    }
    fn line(&self) -> String {
        format!("ratio {:.3} comp {:6.1} MB/s decode {:5.0} MB/s", self.o as f64 / self.c as f64, self.o as f64 / self.ct / 1e6, self.o as f64 / self.dt / 1e6)
    }
}

fn zstd_compress(d: &[u8], level: i32, wlog: Option<u32>) -> Vec<u8> {
    let mut enc = zstd::Encoder::new(Vec::with_capacity(d.len()), level).unwrap();
    if let Some(w) = wlog {
        enc.set_parameter(zstd::zstd_safe::CParameter::WindowLog(w)).unwrap();
    }
    enc.write_all(d).unwrap();
    enc.finish().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let runs: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(3);
    let min_s: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let filter = args.get(2).cloned();
    let files = ["dickens", "mozilla", "mr", "nci", "ooffice", "osdb", "reymont", "samba", "sao", "webster", "xml", "x-ray"];
    let cols = ["Glyd --max", "Glyd --ultra", "zstd -16", "zstd -16 w21", "zstd -19", "zstd -19 w21"];
    let mut tot = [Tot::default(); 6];
    for f in files {
        if filter.as_ref().map_or(false, |x| !f.contains(x.as_str())) {
            continue;
        }
        let p = std::path::Path::new("corpus").join(f);
        if !p.exists() {
            continue;
        }
        let d = std::fs::read(&p).unwrap();
        let mut dst = vec![0u8; d.len() + 1024];
        println!("{f}:");
        for (i, name) in cols.iter().enumerate() {
            let (c, ct) = match i {
                0 | 1 => {
                    let mut b = Vec::with_capacity(d.len());
                    let t = Instant::now();
                    if i == 0 {
                        glyd::compress_into_max(&d, &mut b);
                    } else {
                        glyd::compress_into_ultra(&d, &mut b);
                    }
                    let ct = t.elapsed().as_secs_f64();
                    assert!(glyd::decompress(&b).expect("decode failed") == d, "{name} round trip mismatch on {f}");
                    (b, ct)
                }
                _ => {
                    let (level, wlog) = match i {
                        2 => (16, None),
                        3 => (16, Some(21)),
                        4 => (19, None),
                        _ => (19, Some(21)),
                    };
                    let t = Instant::now();
                    let z = zstd_compress(&d, level, wlog);
                    let ct = t.elapsed().as_secs_f64();
                    assert!(zstd::bulk::decompress(&z, d.len()).unwrap() == d);
                    (z, ct)
                }
            };
            let dt = if i < 2 {
                timed(runs, min_s, || {
                    let _ = glyd::decompress_into_raw(&c, &mut dst);
                })
            } else {
                let mut dec = zstd::bulk::Decompressor::new().unwrap();
                dec.set_parameter(zstd::zstd_safe::DParameter::WindowLogMax(27)).unwrap();
                timed(runs, min_s, || {
                    let _ = dec.decompress_to_buffer(&c, &mut dst[..]);
                })
            };
            tot[i].add(d.len(), c.len(), ct, dt);
            println!("  {name:<13} {}", Tot { o: d.len(), c: c.len(), ct, dt }.line());
        }
    }
    println!("total:");
    for (i, name) in cols.iter().enumerate() {
        println!("  {name:<13} {}", tot[i].line());
    }
}
