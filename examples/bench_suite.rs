// The benchmark suite behind docs/benchmarks: Glyd against zstd -3, zstd
// -19 and LZ4 on the corpus of scripts/download_bench_corpus.sh (logs,
// JSON, database exports, Parquet), large files and small objects, at a
// stated thread count, repeated, every decode checked byte for byte
// against the input. Rows go to a JSON-lines file; a summary prints at
// the end.
//
//   bench_suite [--threads N] [--repeats R] [--slow-repeats R] [--out FILE]
//               [--large] [--small] [--codecs a,b,c] [--files f1,f2] [--max-bytes N]
//
// `--slow-repeats` (default 1) is the repeat count for glyd-ultra and
// zstd-19, whose single-thread passes over a gigabyte take minutes.
// Peak memory is measured in a child process per (file, codec) on the
// file's first 64 MB: the suite re-runs itself with `--child` and reads
// the child's ru_maxrss (input, compressed and decompressed buffers
// included, so 64 MB + compressed + 64 MB is the floor).
//
// What is compared, and how:
// - Large files: whole-file compress and decompress in memory. Glyd's
//   parallel paths and zstd's `-T N` (zstdmt) use N threads to compress;
//   zstd and LZ4 decompress on one thread (their formats offer no more;
//   zstd's CLI does the same), Glyd's parallel decode uses N.
// - Small objects: JSON lines are one object each; logs are cut into
//   records of a fixed size. Every codec gets a 110 KB dictionary trained
//   on a file from corpus/bench/train/ (another day of the same source;
//   never on the measured objects): zstd's trainer for zstd, Glyd's for
//   Glyd, and for LZ4 (lz4_flex's block API with an external
//   dictionary) zstd's trained content. Compressed bytes are reported
//   with and without the dictionary's own size. Latencies are per
//   object, one thread, with the dictionaries prepared once.
// - "Compressed bytes" always include every codec's own framing (magic,
//   headers, checksums: Glyd writes a checksum per block; zstd's default
//   frame has none; LZ4 blocks here carry none).
use std::io::{Read, Write};
use std::time::Instant;

#[derive(Clone)]
struct Opts {
    threads: usize,
    repeats: usize,
    slow_repeats: usize,
    out: String,
    large: bool,
    small: bool,
    codecs: Vec<String>,
    files: Vec<String>,
    max_bytes: usize,
    dir: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--child") {
        // --child <file> <codec> <threads> <max_bytes>
        child(&args[2], &args[3], args[4].parse().unwrap(), args[5].parse().unwrap());
        return;
    }
    let mut o = Opts { threads: 1, repeats: 3, slow_repeats: 1, out: "bench_suite.jsonl".into(), large: false, small: false, codecs: Vec::new(), files: Vec::new(), max_bytes: usize::MAX, dir: "corpus/bench".into() };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => { o.threads = args[i + 1].parse().unwrap(); i += 1; }
            "--repeats" => { o.repeats = args[i + 1].parse().unwrap(); i += 1; }
            "--slow-repeats" => { o.slow_repeats = args[i + 1].parse().unwrap(); i += 1; }
            "--out" => { o.out = args[i + 1].clone(); i += 1; }
            "--codecs" => { o.codecs = args[i + 1].split(',').map(String::from).collect(); i += 1; }
            "--files" => { o.files = args[i + 1].split(',').map(String::from).collect(); i += 1; }
            "--max-bytes" => { o.max_bytes = args[i + 1].parse().unwrap(); i += 1; }
            "--dir" => { o.dir = args[i + 1].clone(); i += 1; }
            "--large" => o.large = true,
            "--small" => o.small = true,
            a => panic!("unknown argument {a}"),
        }
        i += 1;
    }
    if !o.large && !o.small {
        o.large = true;
        o.small = true;
    }
    rayon::ThreadPoolBuilder::new().num_threads(o.threads).build_global().unwrap();
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&o.out).unwrap();
    let machine = machine();
    writeln!(out, "{{\"kind\":\"machine\",{machine},\"threads\":{},\"repeats\":{}}}", o.threads, o.repeats).unwrap();
    eprintln!("machine: {machine}");
    if o.large {
        large(&o, &mut out);
    }
    if o.small {
        small(&o, &mut out);
    }
}

fn machine() -> String {
    let cpu = if cfg!(target_os = "macos") {
        std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    } else {
        std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()))
    }
    .unwrap_or_else(|| "unknown".into());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    format!("\"cpu\":\"{}\",\"cores\":{},\"arch\":\"{}\",\"os\":\"{}\",\"date\":\"{}\"", cpu.replace('"', ""), cores, std::env::consts::ARCH, std::env::consts::OS, date())
}

fn date() -> String {
    String::from_utf8_lossy(&std::process::Command::new("date").args(["-u", "+%FT%TZ"]).output().map(|o| o.stdout).unwrap_or_default()).trim().to_string()
}

// ---------------------------------------------------------------- codecs

/// One codec at one level: whole-buffer compress and decompress.
struct Codec {
    name: &'static str,
    compress: fn(&[u8], usize, &mut Vec<u8>),
    decompress: fn(&[u8], usize, &mut Vec<u8>),
}

fn glyd_default(input: &[u8], threads: usize, out: &mut Vec<u8>) {
    if threads > 1 { glyd::compress_parallel_into(input, out) } else { glyd::compress_into(input, out) }
}
fn glyd_max(input: &[u8], threads: usize, out: &mut Vec<u8>) {
    if threads > 1 { glyd::compress_parallel_into_max(input, out) } else { glyd::compress_into_max(input, out) }
}
fn glyd_ultra(input: &[u8], threads: usize, out: &mut Vec<u8>) {
    if threads > 1 { glyd::compress_parallel_into_ultra(input, out) } else { glyd::compress_into_ultra(input, out) }
}
fn glyd_decompress(input: &[u8], threads: usize, out: &mut Vec<u8>) {
    // Into the caller's buffer, as every codec here: memory that is
    // already mapped after the first repeat (a fresh allocation per call
    // cost Glyd 25% on a 170 MB file: page faults, not decoding).
    let len = DECOMPRESSED_LEN.load(std::sync::atomic::Ordering::Relaxed);
    out.resize(len, 0);
    let n = if threads > 1 { glyd::decompress_parallel_into(input, out) } else { glyd::decompress_into(input, out) }.expect("glyd decode");
    out.truncate(n);
}
fn zstd_level(input: &[u8], threads: usize, level: i32, out: &mut Vec<u8>) {
    out.clear();
    if threads <= 1 {
        // The one-shot API: zstd's fastest single-thread path.
        let mut c = zstd::bulk::Compressor::new(level).unwrap();
        out.reserve(zstd::zstd_safe::compress_bound(input.len()));
        c.compress_to_buffer(input, out).unwrap();
        return;
    }
    let mut enc = zstd::stream::Encoder::new(std::mem::take(out), level).unwrap();
    enc.multithread(threads as u32).unwrap();
    enc.write_all(input).unwrap();
    *out = enc.finish().unwrap();
}
fn zstd3(input: &[u8], threads: usize, out: &mut Vec<u8>) {
    zstd_level(input, threads, 3, out)
}
fn zstd19(input: &[u8], threads: usize, out: &mut Vec<u8>) {
    zstd_level(input, threads, 19, out)
}
fn zstd_decompress(input: &[u8], _threads: usize, out: &mut Vec<u8>) {
    // The one-shot API into the caller's buffer (zstd's fastest path);
    // the frame's content size is not in the header (streaming
    // compression), so the buffer is sized by the upper bound the
    // harness keeps for every codec (`DECOMPRESSED_LEN`).
    let len = DECOMPRESSED_LEN.load(std::sync::atomic::Ordering::Relaxed);
    out.resize(len, 0);
    let n = zstd::bulk::decompress_to_buffer(input, &mut out[..]).unwrap();
    out.truncate(n);
}

/// The size every codec's decode buffer is set to: the original input's
/// length, known to the harness (a store keeps object sizes too).
static DECOMPRESSED_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn lz4_frame(input: &[u8], _threads: usize, out: &mut Vec<u8>) {
    out.clear();
    let mut enc = lz4::EncoderBuilder::new().level(1).build(std::mem::take(out)).unwrap();
    enc.write_all(input).unwrap();
    let (w, r) = enc.finish();
    r.unwrap();
    *out = w;
}
fn lz4_decompress(input: &[u8], _threads: usize, out: &mut Vec<u8>) {
    // The frame reader into the caller's buffer (its capacity kept).
    let len = DECOMPRESSED_LEN.load(std::sync::atomic::Ordering::Relaxed);
    out.clear();
    out.reserve(len);
    let mut dec = lz4::Decoder::new(input).unwrap();
    dec.read_to_end(out).unwrap();
}

fn codecs() -> Vec<Codec> {
    vec![
        Codec { name: "glyd-default", compress: glyd_default, decompress: glyd_decompress },
        Codec { name: "glyd-max", compress: glyd_max, decompress: glyd_decompress },
        Codec { name: "glyd-ultra", compress: glyd_ultra, decompress: glyd_decompress },
        Codec { name: "zstd-3", compress: zstd3, decompress: zstd_decompress },
        Codec { name: "zstd-19", compress: zstd19, decompress: zstd_decompress },
        Codec { name: "lz4", compress: lz4_frame, decompress: lz4_decompress },
    ]
}

// ------------------------------------------------------------ large files

fn read_file(path: &str, max_bytes: usize) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let len = f.metadata().unwrap().len() as usize;
    let mut d = vec![0u8; len.min(max_bytes)];
    f.read_exact(&mut d).unwrap();
    d
}

fn bench_files(o: &Opts) -> Vec<String> {
    if !o.files.is_empty() {
        return o.files.clone();
    }
    let mut v: Vec<String> = std::fs::read_dir(&o.dir).unwrap_or_else(|e| panic!("{}: {e} (run scripts/download_bench_corpus.sh)", o.dir)).filter_map(|e| e.ok()).filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false)).map(|e| e.path().to_string_lossy().to_string()).filter(|p| !p.ends_with(".part")).collect();
    v.sort();
    v
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn large(o: &Opts, out: &mut std::fs::File) {
    let codecs: Vec<Codec> = codecs().into_iter().filter(|c| o.codecs.is_empty() || o.codecs.iter().any(|n| n == c.name)).collect();
    let files = bench_files(o);
    let mut totals: Vec<(String, usize, usize, f64, f64)> = codecs.iter().map(|c| (c.name.to_string(), 0usize, 0usize, 0f64, 0f64)).collect();
    for path in &files {
        let input = read_file(path, o.max_bytes);
        DECOMPRESSED_LEN.store(input.len(), std::sync::atomic::Ordering::Relaxed);
        let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
        eprintln!("{name}: {} bytes", input.len());
        for (ci, c) in codecs.iter().enumerate() {
            let repeats = if c.name == "glyd-ultra" || c.name == "zstd-19" { o.slow_repeats } else { o.repeats };
            let mut comp = Vec::with_capacity(input.len() + input.len() / 8 + (1 << 16));
            let mut ct = Vec::new();
            for _ in 0..repeats {
                comp.clear(); // Glyd's compress_into appends
                let t = Instant::now();
                (c.compress)(&input, o.threads, &mut comp);
                ct.push(t.elapsed().as_secs_f64());
            }
            let mut back = Vec::with_capacity(input.len());
            let mut dt = Vec::new();
            for _ in 0..o.repeats {
                let t = Instant::now();
                (c.decompress)(&comp, o.threads, &mut back);
                dt.push(t.elapsed().as_secs_f64());
                assert!(back == input, "{name} / {}: decoded bytes differ from the input", c.name);
            }
            let (cmed, dmed) = (median(&mut ct), median(&mut dt));
            let rss = child_rss(path, c.name, o.threads, o.max_bytes.min(64 << 20));
            let row = format!("{{\"kind\":\"large\",\"file\":\"{name}\",\"bytes\":{},\"codec\":\"{}\",\"threads\":{},\"compressed\":{},\"ratio\":{:.4},\"compress_s\":{:.4},\"compress_min_s\":{:.4},\"decompress_s\":{:.4},\"decompress_min_s\":{:.4},\"compress_mb_s\":{:.1},\"decompress_mb_s\":{:.1},\"peak_rss_bytes_64mb\":{},\"repeats\":{},\"verified\":true}}",
                input.len(), c.name, o.threads, comp.len(), input.len() as f64 / comp.len() as f64, cmed, ct[0], dmed, dt[0], input.len() as f64 / cmed / 1e6, input.len() as f64 / dmed / 1e6, rss, repeats);
            writeln!(out, "{row}").unwrap();
            eprintln!("  {:<13} ratio {:>7.3}  comp {:>8.1} MB/s  decomp {:>8.1} MB/s  peak rss {:>6} MB", c.name, input.len() as f64 / comp.len() as f64, input.len() as f64 / cmed / 1e6, input.len() as f64 / dmed / 1e6, rss >> 20);
            let t = &mut totals[ci];
            t.1 += input.len();
            t.2 += comp.len();
            t.3 += cmed;
            t.4 += dmed;
        }
    }
    eprintln!("\nlarge files, {} threads, {} bytes in:", o.threads, totals.first().map(|t| t.1).unwrap_or(0));
    for (name, raw, comp, cs, ds) in &totals {
        if *raw > 0 {
            eprintln!("  {:<13} ratio {:>7.3}  compressed {:>12} B  compress {:>8.1} MB/s  decompress {:>8.1} MB/s", name, *raw as f64 / *comp as f64, comp, *raw as f64 / cs / 1e6, *raw as f64 / ds / 1e6);
            writeln!(out, "{{\"kind\":\"large-total\",\"codec\":\"{name}\",\"threads\":{},\"bytes\":{raw},\"compressed\":{comp},\"compress_s\":{cs:.4},\"decompress_s\":{ds:.4}}}", o.threads).unwrap();
        }
    }
}

/// Peak resident set of one compress + decompress of `path` with `codec`,
/// in a fresh process.
fn child_rss(path: &str, codec: &str, threads: usize, max_bytes: usize) -> u64 {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe).args(["--child", path, codec, &threads.to_string(), &max_bytes.to_string()]).output().expect("child");
    assert!(out.status.success(), "child failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
}

fn child(path: &str, codec: &str, threads: usize, max_bytes: usize) {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let c = codecs().into_iter().find(|c| c.name == codec).unwrap();
    let input = read_file(path, max_bytes);
    DECOMPRESSED_LEN.store(input.len(), std::sync::atomic::Ordering::Relaxed);
    let mut comp = Vec::new();
    (c.compress)(&input, threads, &mut comp);
    let mut back = Vec::new();
    (c.decompress)(&comp, threads, &mut back);
    assert!(back == input);
    println!("{}", max_rss());
}

fn max_rss() -> u64 {
    // Linux: the address space's high-water mark (getrusage's ru_maxrss
    // reported the parent's peak for a spawned child on Graviton3).
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(line) = status.lines().find(|l| l.starts_with("VmHWM:")) {
            if let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()) {
                return kb * 1024;
            }
        }
    }
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    // ru_maxrss is bytes on macOS, kilobytes elsewhere.
    if cfg!(target_os = "macos") { ru.ru_maxrss as u64 } else { ru.ru_maxrss as u64 * 1024 }
}

// ----------------------------------------------------------- small objects

/// A small-object dataset: the objects measured and the samples the
/// dictionaries are trained on, from different files.
struct SmallSet {
    name: String,
    objects: Vec<Vec<u8>>,
    samples: Vec<Vec<u8>>,
}

fn small_sets(o: &Opts) -> Vec<SmallSet> {
    let dir = &o.dir;
    let mut sets = Vec::new();
    let take = 20_000usize;
    // JSON events: one line each.
    let lines = |path: &str, n: usize| -> Vec<Vec<u8>> {
        let d = read_file(path, 256 << 20);
        d.split(|&b| b == b'\n').filter(|l| !l.is_empty()).take(n).map(|l| l.to_vec()).collect()
    };
    let json = format!("{dir}/gharchive-2024-01-16-12.json");
    let json_train = format!("{dir}/train/gharchive-2024-01-14-12.json");
    if std::path::Path::new(&json).exists() && std::path::Path::new(&json_train).exists() {
        sets.push(SmallSet { name: "gharchive events (JSON lines)".into(), objects: lines(&json, take), samples: lines(&json_train, take) });
    }
    // Logs: records of 1 KB and 4 KB (whole lines up to the size).
    let records = |path: &str, size: usize, n: usize| -> Vec<Vec<u8>> {
        let d = read_file(path, 256 << 20);
        let mut v = Vec::new();
        let mut cur = Vec::new();
        for l in d.split(|&b| b == b'\n') {
            if cur.len() + l.len() + 1 > size && !cur.is_empty() {
                v.push(std::mem::take(&mut cur));
                if v.len() == n {
                    break;
                }
            }
            cur.extend_from_slice(l);
            cur.push(b'\n');
        }
        v
    };
    for (label, file, train) in [("nasa access log", "nasa-access-jul95.log", "train/nasa-access-aug95.log"), ("wikipedia pageviews", "pageviews-20240115-10.log", "train/pageviews-20240114-10.log")] {
        let (f, t) = (format!("{dir}/{file}"), format!("{dir}/{train}"));
        if std::path::Path::new(&f).exists() && std::path::Path::new(&t).exists() {
            for size in [1024usize, 4096] {
                sets.push(SmallSet { name: format!("{label}, {} B records", size), objects: records(&f, size, take), samples: records(&t, size, take) });
            }
        }
    }
    sets
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

/// Per-object latencies of `f` over `objects`, `repeats` passes, best
/// pass per object; returns (p50, p90, p99, mean) in microseconds and
/// the total bytes produced in the last pass.
fn latencies(objects: &[Vec<u8>], repeats: usize, mut f: impl FnMut(&[u8]) -> usize) -> ([f64; 4], usize) {
    let mut best = vec![f64::MAX; objects.len()];
    let mut total = 0;
    for _ in 0..repeats {
        total = 0;
        for (i, obj) in objects.iter().enumerate() {
            let t = Instant::now();
            total += f(obj);
            let dt = t.elapsed().as_secs_f64() * 1e6;
            best[i] = best[i].min(dt);
        }
    }
    let mean = best.iter().sum::<f64>() / best.len() as f64;
    ([percentile(&mut best, 0.5), percentile(&mut best, 0.9), percentile(&mut best, 0.99), mean], total)
}

fn small(o: &Opts, out: &mut std::fs::File) {
    const DICT_BYTES: usize = 110 * 1024;
    for set in small_sets(o) {
        let raw: usize = set.objects.iter().map(|x| x.len()).sum();
        let n = set.objects.len();
        eprintln!("\n{}: {} objects, {} bytes (mean {} B); dictionaries from {} samples", set.name, n, raw, raw / n.max(1), set.samples.len());
        let samples: Vec<&[u8]> = set.samples.iter().map(|s| s.as_slice()).collect();
        let zdict = zstd::dict::from_samples(&samples, DICT_BYTES).unwrap();
        let gdict = glyd::Dict::train(&samples, DICT_BYTES);
        let gdict_bytes = gdict.to_bytes().len();
        // LZ4's dictionary: zstd's trained content (the dictionary's tail).
        let lz4_dict: Vec<u8> = zdict[zdict.len().saturating_sub(100 * 1024)..].to_vec();
        let want = |name: &str| o.codecs.is_empty() || o.codecs.iter().any(|c| c == name);
        let mut rows: Vec<(String, usize, usize, [f64; 4], [f64; 4])> = Vec::new();
        let mut buf = Vec::with_capacity(1 << 20);
        let max_len = set.objects.iter().map(|x| x.len()).max().unwrap_or(0);
        let mut dst = vec![0u8; max_len + 1024];
        let objects = &set.objects;
        let set_name = set.name.clone();
        let check = |name: &str, comp: &[Vec<u8>], dec: &mut dyn FnMut(&[u8], &mut [u8]) -> usize| {
            let mut d = vec![0u8; max_len + 1024];
            for (c, obj) in comp.iter().zip(objects) {
                let n = dec(c, &mut d);
                assert!(&d[..n] == obj.as_slice(), "{set_name}: {name} decoded bytes differ");
            }
        };
        // Glyd --max + Dict
        if want("glyd-max") {
            let (cl, comp_total) = latencies(&set.objects, o.repeats, |x| { buf.clear(); glyd::compress_with_dict(&gdict, x, &mut buf); buf.len() });
            let comp: Vec<Vec<u8>> = set.objects.iter().map(|x| { let mut b = Vec::new(); glyd::compress_with_dict(&gdict, x, &mut b); b }).collect();
            check("glyd-max", &comp, &mut |c, d| glyd::decompress_with_dict_into(&gdict, c, d).unwrap());
            let (dl, _) = latencies(&comp, o.repeats, |c| glyd::decompress_with_dict_into(&gdict, c, &mut dst).unwrap());
            rows.push(("glyd-max+dict".into(), comp_total, gdict_bytes, cl, dl));
        }
        if want("glyd-ultra") {
            let (cl, comp_total) = latencies(&set.objects, 1, |x| { buf.clear(); glyd::compress_with_dict_ultra(&gdict, x, &mut buf); buf.len() });
            let comp: Vec<Vec<u8>> = set.objects.iter().map(|x| { let mut b = Vec::new(); glyd::compress_with_dict_ultra(&gdict, x, &mut b); b }).collect();
            check("glyd-ultra", &comp, &mut |c, d| glyd::decompress_with_dict_into(&gdict, c, d).unwrap());
            let (dl, _) = latencies(&comp, o.repeats, |c| glyd::decompress_with_dict_into(&gdict, c, &mut dst).unwrap());
            rows.push(("glyd-ultra+dict".into(), comp_total, gdict_bytes, cl, dl));
        }
        for (name, level) in [("zstd-3", 3), ("zstd-19", 19)] {
            if !want(name) {
                continue;
            }
            let mut zc = zstd::bulk::Compressor::with_dictionary(level, &zdict).unwrap();
            let mut zd = zstd::bulk::Decompressor::with_dictionary(&zdict).unwrap();
            let reps = if level == 19 { 1 } else { o.repeats };
            let (cl, comp_total) = latencies(&set.objects, reps, |x| { buf.clear(); zc.compress_to_buffer(x, &mut buf).unwrap() });
            let comp: Vec<Vec<u8>> = set.objects.iter().map(|x| zc.compress(x).unwrap()).collect();
            check(name, &comp, &mut |c, d| zd.decompress_to_buffer(c, d).unwrap());
            let (dl, _) = latencies(&comp, o.repeats, |c| zd.decompress_to_buffer(c, &mut dst[..]).unwrap());
            rows.push((format!("{name}+dict"), comp_total, zdict.len(), cl, dl));
        }
        if want("lz4") {
            let (cl, comp_total) = latencies(&set.objects, o.repeats, |x| { buf.clear(); buf.resize(lz4_flex::block::get_maximum_output_size(x.len()), 0); lz4_flex::block::compress_into_with_dict(x, &mut buf, &lz4_dict).unwrap() });
            let comp: Vec<Vec<u8>> = set.objects.iter().map(|x| lz4_flex::block::compress_with_dict(x, &lz4_dict)).collect();
            check("lz4", &comp, &mut |c, d| lz4_flex::block::decompress_into_with_dict(c, d, &lz4_dict).unwrap());
            let (dl, _) = latencies(&comp, o.repeats, |c| lz4_flex::block::decompress_into_with_dict(c, &mut dst, &lz4_dict).unwrap());
            // lz4_flex blocks carry no size: a real store keeps the object
            // length beside them, counted here as 4 bytes per object.
            rows.push(("lz4+dict".into(), comp_total + 4 * n, lz4_dict.len(), cl, dl));
        }
        eprintln!("  {:<16} {:>8} {:>10} {:>9}  {:>26}  {:>26}", "codec", "ratio", "bytes", "+dict", "compress us p50/p90/p99", "decompress us p50/p90/p99");
        for (name, comp, dict, cl, dl) in &rows {
            eprintln!("  {:<16} {:>8.3} {:>10} {:>9}  {:>8.2}/{:>7.2}/{:>7.2}  {:>8.2}/{:>7.2}/{:>7.2}", name, raw as f64 / *comp as f64, comp, comp + dict, cl[0], cl[1], cl[2], dl[0], dl[1], dl[2]);
            writeln!(out, "{{\"kind\":\"small\",\"set\":\"{}\",\"objects\":{n},\"bytes\":{raw},\"codec\":\"{name}\",\"compressed\":{comp},\"dict_bytes\":{dict},\"ratio\":{:.4},\"compress_us_p50\":{:.3},\"compress_us_p90\":{:.3},\"compress_us_p99\":{:.3},\"compress_us_mean\":{:.3},\"decompress_us_p50\":{:.3},\"decompress_us_p90\":{:.3},\"decompress_us_p99\":{:.3},\"decompress_us_mean\":{:.3},\"verified\":true}}",
                set.name, raw as f64 / *comp as f64, cl[0], cl[1], cl[2], cl[3], dl[0], dl[1], dl[2], dl[3]).unwrap();
        }
    }
}
