// Phase 0: measure the competitive field on this machine, so a target can be
// chosen from a real Pareto frontier instead of invented numbers.
//
// The question this answers: does ANY codec dominate liblz4 on all three of
// ratio, compression speed and decompression speed at once? If one does, it is
// a safe target. If none does, "beat LZ4 on all three" is a research bet.
use std::path::Path;
use std::time::Instant;

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[cfg(target_os = "linux")]
fn pin_to_core(core_id: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core_id, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}
#[cfg(not(target_os = "linux"))]
fn pin_to_core(_c: usize) {}

/// Median of `runs` measurements, each at least `min_secs` and `min_iters`.
fn timed<F: FnMut()>(runs: usize, min_secs: f64, min_iters: usize, mut op: F) -> f64 {
    let mut ts = Vec::with_capacity(runs);
    for _ in 0..runs {
        op(); // warm-up
        let start = Instant::now();
        let mut n = 0usize;
        loop {
            op();
            n += 1;
            let e = start.elapsed().as_secs_f64();
            if e >= min_secs && n >= min_iters {
                ts.push(e / n as f64);
                break;
            }
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[ts.len() / 2]
}

#[derive(Default, Clone)]
struct Totals {
    name: String,
    orig: u64,
    comp: u64,
    comp_time: f64,
    decomp_time: f64,
}

impl Totals {
    fn ratio(&self) -> f64 { self.orig as f64 / self.comp as f64 }
    fn comp_gb(&self) -> f64 { (self.orig as f64 / GB) / self.comp_time }
    fn decomp_gb(&self) -> f64 { (self.orig as f64 / GB) / self.decomp_time }
}

fn main() {
    let runs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let min_secs: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);

    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = Path::new("corpus");

    let names = ["Glyd", "liblz4", "lz4_flex", "LZAV", "LZAV-hi",
                 "zstd--5", "zstd--3", "zstd--1", "zstd-1", "zstd-3", "snappy", "Glyd-fast", "Glyd-turbo", "Glyd-max"];
    let mut tot: Vec<Totals> = names.iter().map(|n| Totals { name: n.to_string(), ..Default::default() }).collect();

    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let data = std::fs::read(&p).unwrap();
        let len = data.len();
        eprintln!("measuring {} ({:.1} MB)...", f, len as f64 / 1048576.0);

        let mut dst = vec![0u8; len + 1024];

        // ---- Glyd ----
        pin_to_core(4);
        let mut c0 = Vec::with_capacity(len);
        glyd::compress_into(&data, &mut c0);
        let t = timed(runs, min_secs, 3, || { let mut b = Vec::with_capacity(len); glyd::compress_into(&data, &mut b); });
        let d = timed(runs, min_secs, 3, || { let _ = glyd::decompress_into_raw(&c0, &mut dst); });
        acc(&mut tot[0], len, c0.len(), t, d); log_row(f, "Glyd", len, c0.len(), t, d);

        // ---- Glyd fast level ----
        let mut c1 = Vec::with_capacity(len);
        glyd::compress_into_fast(&data, &mut c1);
        let t = timed(runs, min_secs, 3, || { let mut b = Vec::with_capacity(len); glyd::compress_into_fast(&data, &mut b); });
        let d = timed(runs, min_secs, 3, || { let _ = glyd::decompress_into_raw(&c1, &mut dst); });
        acc(&mut tot[11], len, c1.len(), t, d); log_row(f, "Glyd-fast", len, c1.len(), t, d);

        // ---- Glyd turbo level ----
        let mut c2 = Vec::with_capacity(len);
        glyd::compress_into_turbo(&data, &mut c2);
        let t = timed(runs, min_secs, 3, || { let mut b = Vec::with_capacity(len); glyd::compress_into_turbo(&data, &mut b); });
        let d = timed(runs, min_secs, 3, || { let _ = glyd::decompress_into_raw(&c2, &mut dst); });
        acc(&mut tot[12], len, c2.len(), t, d); log_row(f, "Glyd-turbo", len, c2.len(), t, d);

        // ---- Glyd max level (format v7, entropy coded) ----
        let mut c3 = Vec::with_capacity(len);
        glyd::compress_into_max(&data, &mut c3);
        let t = timed(runs, min_secs, 3, || { let mut b = Vec::with_capacity(len); glyd::compress_into_max(&data, &mut b); });
        let d = timed(runs, min_secs, 3, || { let _ = glyd::decompress_into_raw(&c3, &mut dst); });
        acc(&mut tot[13], len, c3.len(), t, d); log_row(f, "Glyd-max", len, c3.len(), t, d);

        // ---- liblz4 (reference C) ----
        let bound = lz4::block::compress_bound(len).unwrap_or(len * 2 + 64);
        let mut lb = vec![0u8; bound];
        let n = lz4::block::compress_to_buffer(&data, None, false, &mut lb).unwrap();
        let lz4c = lb[..n].to_vec();
        let t = timed(runs, min_secs, 3, || { let _ = lz4::block::compress_to_buffer(&data, None, false, &mut lb); });
        let d = timed(runs, min_secs, 3, || { let _ = lz4::block::decompress_to_buffer(&lz4c, Some(len as i32), &mut dst); });
        acc(&mut tot[1], len, lz4c.len(), t, d); log_row(f, "liblz4", len, lz4c.len(), t, d);

        // ---- lz4_flex ----
        let fc = lz4_flex::compress(&data);
        let t = timed(runs, min_secs, 3, || { let _ = lz4_flex::compress(&data); });
        let d = timed(runs, min_secs, 3, || { let _ = lz4_flex::decompress_into(&fc, &mut dst); });
        acc(&mut tot[2], len, fc.len(), t, d); log_row(f, "lz4_flex", len, fc.len(), t, d);

        // ---- LZAV default and hi ----
        for (idx, hi) in [(3usize, false), (4usize, true)] {
            unsafe {
                let b = if hi { lzav::compress_bound_hi(len as i32) } else { lzav::compress_bound(len as i32) } as usize;
                let mut buf = vec![0u8; b];
                let n = if hi {
                    lzav::compress_hi(data.as_ptr() as *const _, buf.as_mut_ptr() as *mut _, len as i32, b as i32)
                } else {
                    lzav::compress_default(data.as_ptr() as *const _, buf.as_mut_ptr() as *mut _, len as i32, b as i32)
                };
                assert!(n > 0, "lzav compress failed");
                let cbuf = buf[..n as usize].to_vec();
                let t = timed(runs, min_secs, 3, || {
                    if hi { lzav::compress_hi(data.as_ptr() as *const _, buf.as_mut_ptr() as *mut _, len as i32, b as i32); }
                    else { lzav::compress_default(data.as_ptr() as *const _, buf.as_mut_ptr() as *mut _, len as i32, b as i32); }
                });
                let d = timed(runs, min_secs, 3, || {
                    lzav::decompress(cbuf.as_ptr() as *const _, dst.as_mut_ptr() as *mut _, cbuf.len() as i32, len as i32);
                });
                // verify correctness before trusting the timing
                let r = lzav::decompress(cbuf.as_ptr() as *const _, dst.as_mut_ptr() as *mut _, cbuf.len() as i32, len as i32);
                assert_eq!(r, len as i32, "lzav roundtrip length mismatch");
                assert_eq!(&dst[..len], &data[..], "lzav roundtrip mismatch");
                acc(&mut tot[idx], len, cbuf.len(), t, d); log_row(f, if hi {"LZAV-hi"} else {"LZAV"}, len, cbuf.len(), t, d);
            }
        }

        // ---- zstd levels ----
        for (idx, lvl) in [(5usize, -5i32), (6, -3), (7, -1), (8, 1), (9, 3)] {
            let zc = zstd::bulk::compress(&data, lvl).unwrap();
            let t = timed(runs, min_secs, 2, || { let _ = zstd::bulk::compress(&data, lvl); });
            let d = timed(runs, min_secs, 2, || { let _ = zstd::bulk::decompress_to_buffer(&zc, &mut dst); });
            acc(&mut tot[idx], len, zc.len(), t, d); log_row(f, &format!("zstd{}", lvl), len, zc.len(), t, d);
        }

        // ---- snappy ----
        let mut enc = snap::raw::Encoder::new();
        let sc = enc.compress_vec(&data).unwrap();
        let mut sbuf = vec![0u8; snap::raw::max_compress_len(len)];
        let mut dec = snap::raw::Decoder::new();
        let t = timed(runs, min_secs, 3, || { let _ = enc.compress(&data, &mut sbuf[..]); });
        let d = timed(runs, min_secs, 3, || { let _ = dec.decompress(&sc, &mut dst); });
        acc(&mut tot[10], len, sc.len(), t, d); log_row(f, "snappy", len, sc.len(), t, d);
    }

    report(&tot);
}

fn log_row(file: &str, codec: &str, orig: usize, comp: usize, ct: f64, dt: f64) {
    use std::io::Write;
    let path = "field_survey_partial.csv";
    let new = !std::path::Path::new(path).exists();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        if new {
            let _ = writeln!(f, "file,codec,orig_bytes,comp_bytes,comp_secs,decomp_secs");
        }
        let _ = writeln!(f, "{},{},{},{},{:.9},{:.9}", file, codec, orig, comp, ct, dt);
        let _ = f.flush();
    }
}

fn acc(t: &mut Totals, orig: usize, comp: usize, ct: f64, dt: f64) {
    t.orig += orig as u64;
    t.comp += comp as u64;
    t.comp_time += ct;
    t.decomp_time += dt;
}

fn report(tot: &[Totals]) {
    let lz4 = tot.iter().find(|t| t.name == "liblz4").unwrap().clone();
    let (br, bc, bd) = (lz4.ratio(), lz4.comp_gb(), lz4.decomp_gb());

    println!("\n=================================================================================");
    println!("  PHASE 0 FIELD SURVEY - Silesia, single core, this machine");
    println!("  Question: does any codec dominate liblz4 on ratio AND comp AND decomp?");
    println!("=================================================================================");
    println!("{:<10} | {:>7} {:>7} | {:>9} {:>7} | {:>9} {:>7} | {}",
             "codec", "ratio", "vs lz4", "comp GB/s", "vs lz4", "dec GB/s", "vs lz4", "dominates lz4?");
    println!("{}", "-".repeat(97));

    let mut rows: Vec<&Totals> = tot.iter().filter(|t| t.orig > 0).collect();
    rows.sort_by(|a, b| b.ratio().partial_cmp(&a.ratio()).unwrap());

    let mut dominators = Vec::new();
    for t in &rows {
        let (r, c, d) = (t.ratio(), t.comp_gb(), t.decomp_gb());
        let dom = r > br && c > bc && d > bd;
        if dom && t.name != "liblz4" { dominators.push(t.name.clone()); }
        let mark = if t.name == "liblz4" { "(baseline)".to_string() }
                   else if dom { "YES - ALL THREE".to_string() }
                   else {
                       let mut w = Vec::new();
                       if r > br { w.push("ratio"); }
                       if c > bc { w.push("comp"); }
                       if d > bd { w.push("dec"); }
                       if w.is_empty() { "no".to_string() } else { format!("wins: {}", w.join("+")) }
                   };
        println!("{:<10} | {:>7.4} {:>6.3}x | {:>9.3} {:>6.3}x | {:>9.3} {:>6.3}x | {}",
                 t.name, r, r/br, c, c/bc, d, d/bd, mark);
    }
    println!("{}", "-".repeat(97));
    println!();
    if dominators.is_empty() {
        println!("ANSWER: NO codec in this field dominates liblz4 on all three axes.");
        println!("        'Beat LZ4 on ratio + compression + decompression' has no existence proof.");
        println!("        It is a research bet, not an engineering target.");
    } else {
        println!("ANSWER: these dominate liblz4 on all three: {}", dominators.join(", "));
        println!("        Target the weakest of them. Achievable by construction.");
    }
}
