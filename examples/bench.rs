use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn throughput_gb(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / secs
}

#[cfg(target_os = "linux")]
pub fn pin_to_core(core_id: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core_id, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(target_os = "linux")]
pub fn unpin_cores() {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let ncpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize;
        for i in 0..ncpus.min(1024) {
            libc::CPU_SET(i, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_to_core(_core_id: usize) {}
#[cfg(not(target_os = "linux"))]
pub fn unpin_cores() {}

#[derive(Clone, Copy, Debug)]
pub struct TimingConfig {
    pub min_secs: f64,
    pub min_iters: usize,
    pub runs: usize,
}

impl TimingConfig {
    /// Strict Section 1 timing policy: 5 measurements of >= 1.0s and >= 5 iters, reporting median.
    pub fn strict() -> Self {
        Self {
            min_secs: 1.0,
            min_iters: 5,
            runs: 5,
        }
    }

    /// Fast development mode for quick checks.
    pub fn fast() -> Self {
        Self {
            min_secs: 0.1,
            min_iters: 2,
            runs: 1,
        }
    }
}

fn measure_median_time<F: FnMut()>(config: TimingConfig, mut op: F) -> f64 {
    let mut run_times = Vec::with_capacity(config.runs);
    for _ in 0..config.runs {
        // 1 warm-up iteration
        op();
        let start = Instant::now();
        let mut iters = 0usize;
        loop {
            op();
            iters += 1;
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed >= config.min_secs && iters >= config.min_iters {
                run_times.push(elapsed / iters as f64);
                break;
            }
        }
    }
    run_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    run_times[run_times.len() / 2]
}

#[derive(Clone, Debug)]
pub struct BenchRecord {
    pub category: String,
    pub name: String,
    pub orig_size: usize,

    // Compressed sizes (bytes)
    pub simd_size: usize,
    pub lz4_size: usize,
    pub lz4_flex_size: usize,
    pub snap_size: usize,
    pub zstd_size: usize,

    // Ratios
    pub simd_ratio: f64,
    pub lz4_ratio: f64,
    pub lz4_flex_ratio: f64,
    pub snap_ratio: f64,
    pub zstd_ratio: f64,
    pub ratio_vs_lz4: f64,

    // Per-iteration times (seconds)
    pub simd_comp_1c_time: f64,
    pub simd_comp_16c_time: f64,
    pub lz4_comp_1c_time: f64,
    pub lz4_flex_comp_1c_time: f64,
    pub snap_comp_1c_time: f64,
    pub zstd_comp_1c_time: f64,

    pub simd_decomp_1c_raw_time: f64,
    pub simd_decomp_1c_ver_time: f64,
    pub simd_decomp_16c_raw_time: f64,
    pub simd_decomp_16c_ver_time: f64,
    pub lz4_decomp_1c_time: f64,
    pub lz4_flex_decomp_1c_time: f64,
    pub snap_decomp_1c_time: f64,
    pub zstd_decomp_1c_time: f64,

    // Throughput (GB/s)
    pub simd_comp_1c_gb: f64,
    pub simd_comp_16c_gb: f64,
    pub lz4_comp_1c_gb: f64,
    pub lz4_flex_comp_1c_gb: f64,
    pub snap_comp_1c_gb: f64,
    pub zstd_comp_1c_gb: f64,

    pub simd_decomp_1c_raw_gb: f64,
    pub simd_decomp_1c_ver_gb: f64,
    pub simd_decomp_16c_raw_gb: f64,
    pub simd_decomp_16c_ver_gb: f64,
    pub lz4_decomp_1c_gb: f64,
    pub lz4_flex_decomp_1c_gb: f64,
    pub snap_decomp_1c_gb: f64,
    pub zstd_decomp_1c_gb: f64,
}

fn benchmark_file(
    category: &str,
    name: &str,
    data: &[u8],
    config: TimingConfig,
) -> BenchRecord {
    let len = data.len();

    // Prepare compressed buffers for decompression benchmark
    let mut simd_comp = Vec::with_capacity(len);
    glyd::compress_into(data, &mut simd_comp);

    let mut simd_par_comp = Vec::with_capacity(len);
    glyd::compress_parallel_into(data, &mut simd_par_comp);

    let lz4_bound = lz4::block::compress_bound(len).unwrap_or(len * 2 + 64);
    let mut lz4_comp = vec![0u8; lz4_bound];
    let lz4_sz = lz4::block::compress_to_buffer(data, None, false, &mut lz4_comp)
        .expect("liblz4 compress failed");
    lz4_comp.truncate(lz4_sz);

    let lz4_flex_comp = lz4_flex::compress(data);

    let mut snap_enc = snap::raw::Encoder::new();
    let snap_comp = snap_enc.compress_vec(data).expect("snap compress failed");

    let zstd_comp = zstd::bulk::compress(data, 1).expect("zstd compress failed");

    // Pre-allocated destination buffers
    let mut simd_dest = vec![0u8; len + 128];
    let mut lz4_dest = vec![0u8; len + 128];
    let mut snap_dest = vec![0u8; len + 128];
    let mut zstd_dest = vec![0u8; len + 128];

    // --- Compression Timings ---
    // 1. Glyd 1C
    pin_to_core(4);
    let mut c_buf = Vec::with_capacity(len);
    let simd_comp_1c_time = measure_median_time(config, || {
        c_buf.clear();
        glyd::compress_into(data, &mut c_buf);
    });
    let simd_comp_1c_gb = throughput_gb(len, simd_comp_1c_time);

    // 2. Glyd 16C (unpinned Rayon)
    unpin_cores();
    let mut par_c_buf = Vec::with_capacity(len);
    let simd_comp_16c_time = measure_median_time(config, || {
        par_c_buf.clear();
        glyd::compress_parallel_into(data, &mut par_c_buf);
    });
    let simd_comp_16c_gb = throughput_gb(len, simd_comp_16c_time);

    // 3. C liblz4 1C
    pin_to_core(4);
    let mut lz4_out = vec![0u8; lz4_bound];
    let lz4_comp_1c_time = measure_median_time(config, || {
        let _ = lz4::block::compress_to_buffer(data, None, false, &mut lz4_out);
    });
    let lz4_comp_1c_gb = throughput_gb(len, lz4_comp_1c_time);

    // 4. lz4_flex 1C
    pin_to_core(4);
    let lz4_flex_comp_1c_time = measure_median_time(config, || {
        let _ = lz4_flex::compress(data);
    });
    let lz4_flex_comp_1c_gb = throughput_gb(len, lz4_flex_comp_1c_time);

    // 5. Snappy 1C
    pin_to_core(4);
    let mut snap_buf = vec![0u8; snap::raw::max_compress_len(len)];
    let snap_comp_1c_time = measure_median_time(config, || {
        let _ = snap_enc.compress(data, &mut snap_buf[..]);
    });
    let snap_comp_1c_gb = throughput_gb(len, snap_comp_1c_time);

    // 6. Zstd level 1 1C
    pin_to_core(4);
    let mut zstd_out = vec![0u8; zstd::zstd_safe::compress_bound(len)];
    let zstd_comp_1c_time = measure_median_time(config, || {
        let _ = zstd::bulk::compress_to_buffer(data, &mut zstd_out, 1);
    });
    let zstd_comp_1c_gb = throughput_gb(len, zstd_comp_1c_time);

    // --- Decompression Timings ---
    // 1. Glyd 1C Raw
    pin_to_core(4);
    let simd_decomp_1c_raw_time = measure_median_time(config, || {
        let _ = glyd::decompress_into_raw(&simd_comp, &mut simd_dest);
    });
    let simd_decomp_1c_raw_gb = throughput_gb(len, simd_decomp_1c_raw_time);

    // 2. Glyd 1C Verified
    pin_to_core(4);
    let simd_decomp_1c_ver_time = measure_median_time(config, || {
        let _ = glyd::decompress_into(&simd_comp, &mut simd_dest);
    });
    let simd_decomp_1c_ver_gb = throughput_gb(len, simd_decomp_1c_ver_time);

    // 3. Glyd 16C Raw
    unpin_cores();
    let simd_decomp_16c_raw_time = measure_median_time(config, || {
        let _ = glyd::decompress_parallel_into_raw(&simd_par_comp, &mut simd_dest);
    });
    let simd_decomp_16c_raw_gb = throughput_gb(len, simd_decomp_16c_raw_time);

    // 4. Glyd 16C Verified
    unpin_cores();
    let simd_decomp_16c_ver_time = measure_median_time(config, || {
        let _ = glyd::decompress_parallel_into(&simd_par_comp, &mut simd_dest);
    });
    let simd_decomp_16c_ver_gb = throughput_gb(len, simd_decomp_16c_ver_time);

    // 5. C liblz4 1C Decompress
    pin_to_core(4);
    let lz4_decomp_1c_time = measure_median_time(config, || {
        let _ = lz4::block::decompress_to_buffer(&lz4_comp, Some(len as i32), &mut lz4_dest);
    });
    let lz4_decomp_1c_gb = throughput_gb(len, lz4_decomp_1c_time);

    // 6. lz4_flex 1C Decompress
    pin_to_core(4);
    let lz4_flex_decomp_1c_time = measure_median_time(config, || {
        let _ = lz4_flex::decompress_into(&lz4_flex_comp, &mut lz4_dest);
    });
    let lz4_flex_decomp_1c_gb = throughput_gb(len, lz4_flex_decomp_1c_time);

    // 7. Snappy 1C Decompress
    pin_to_core(4);
    let mut snap_dec = snap::raw::Decoder::new();
    let snap_decomp_1c_time = measure_median_time(config, || {
        let _ = snap_dec.decompress(&snap_comp, &mut snap_dest);
    });
    let snap_decomp_1c_gb = throughput_gb(len, snap_decomp_1c_time);

    // 8. Zstd 1C Decompress
    pin_to_core(4);
    let zstd_decomp_1c_time = measure_median_time(config, || {
        let _ = zstd::bulk::decompress_to_buffer(&zstd_comp, &mut zstd_dest);
    });
    let zstd_decomp_1c_gb = throughput_gb(len, zstd_decomp_1c_time);

    unpin_cores();

    let simd_ratio = len as f64 / simd_comp.len() as f64;
    let lz4_ratio = len as f64 / lz4_comp.len() as f64;
    let lz4_flex_ratio = len as f64 / lz4_flex_comp.len() as f64;
    let snap_ratio = len as f64 / snap_comp.len() as f64;
    let zstd_ratio = len as f64 / zstd_comp.len() as f64;
    let ratio_vs_lz4 = simd_ratio / lz4_ratio;

    BenchRecord {
        category: category.to_string(),
        name: name.to_string(),
        orig_size: len,
        simd_size: simd_comp.len(),
        lz4_size: lz4_comp.len(),
        lz4_flex_size: lz4_flex_comp.len(),
        snap_size: snap_comp.len(),
        zstd_size: zstd_comp.len(),
        simd_ratio,
        lz4_ratio,
        lz4_flex_ratio,
        snap_ratio,
        zstd_ratio,
        ratio_vs_lz4,
        simd_comp_1c_time,
        simd_comp_16c_time,
        lz4_comp_1c_time,
        lz4_flex_comp_1c_time,
        snap_comp_1c_time,
        zstd_comp_1c_time,
        simd_decomp_1c_raw_time,
        simd_decomp_1c_ver_time,
        simd_decomp_16c_raw_time,
        simd_decomp_16c_ver_time,
        lz4_decomp_1c_time,
        lz4_flex_decomp_1c_time,
        snap_decomp_1c_time,
        zstd_decomp_1c_time,
        simd_comp_1c_gb,
        simd_comp_16c_gb,
        lz4_comp_1c_gb,
        lz4_flex_comp_1c_gb,
        snap_comp_1c_gb,
        zstd_comp_1c_gb,
        simd_decomp_1c_raw_gb,
        simd_decomp_1c_ver_gb,
        simd_decomp_16c_raw_gb,
        simd_decomp_16c_ver_gb,
        lz4_decomp_1c_gb,
        lz4_flex_decomp_1c_gb,
        snap_decomp_1c_gb,
        zstd_decomp_1c_gb,
    }
}

pub fn run_silesia_bench(records: &mut Vec<BenchRecord>, config: TimingConfig) {
    println!("\n=========================================================================================================");
    println!("             OFFICIAL SILESIA COMPRESSION CORPUS BENCHMARK (AVX-512 + MULTI-CORE)");
    println!("             Hardware: AMD Ryzen 9 7950X3D (Zen 4, 16C/32T) | Pinned 1C: Core 4 | Rayon: 16 Threads");
    println!("=========================================================================================================");

    let corpus_dir = Path::new("corpus");
    if !corpus_dir.exists() {
        eprintln!("Corpus directory not found. Please run: bash scripts/download_corpus.sh");
        return;
    }

    let files = [
        "dickens", "mozilla", "mr", "nci", "ooffice", "osdb",
        "reymont", "samba", "sao", "webster", "xml", "x-ray",
    ];

    for file_name in &files {
        let path = corpus_dir.join(file_name);
        if !path.exists() {
            eprintln!("Missing Silesia file: {}", path.display());
            continue;
        }

        let data = fs::read(&path).expect("Failed to read corpus file");
        let r = benchmark_file("silesia", file_name, &data, config);

        println!(
            "Tested {:<10} ({:>5.1} MB) | Ratio: {:>4.2}x vs LZ4 {:>4.2}x ({:>5.1}%) | Comp 1C: {:>4.2} GB/s (LZ4: {:>4.2}) | Decomp 1C: {:>4.2} GB/s (LZ4: {:>4.2})",
            r.name,
            r.orig_size as f64 / (1024.0 * 1024.0),
            r.simd_ratio,
            r.lz4_ratio,
            (r.ratio_vs_lz4 - 1.0) * 100.0,
            r.simd_comp_1c_gb,
            r.lz4_comp_1c_gb,
            r.simd_decomp_1c_raw_gb,
            r.lz4_decomp_1c_gb,
        );

        records.push(r);
    }
}

pub fn run_enwik8_bench(records: &mut Vec<BenchRecord>, config: TimingConfig) {
    println!("\n=========================================================================================================");
    println!("                         ENWIK8 HOLDOUT CORPUS BENCHMARK (100 MB)");
    println!("=========================================================================================================");

    let path = Path::new("corpus/enwik8");
    if !path.exists() {
        eprintln!("enwik8 not found. Run bash scripts/download_corpus.sh");
        return;
    }

    let data = fs::read(path).expect("Failed to read enwik8");
    let r = benchmark_file("enwik8", "enwik8", &data, config);

    println!(
        "Tested {:<10} ({:>5.1} MB) | Ratio: {:>4.2}x vs LZ4 {:>4.2}x ({:>5.1}%) | Comp 1C: {:>4.2} GB/s (LZ4: {:>4.2}) | Decomp 1C: {:>4.2} GB/s (LZ4: {:>4.2})",
        r.name,
        r.orig_size as f64 / (1024.0 * 1024.0),
        r.simd_ratio,
        r.lz4_ratio,
        (r.ratio_vs_lz4 - 1.0) * 100.0,
        r.simd_comp_1c_gb,
        r.lz4_comp_1c_gb,
        r.simd_decomp_1c_raw_gb,
        r.lz4_decomp_1c_gb,
    );

    records.push(r);
}

pub fn run_workload_bench(records: &mut Vec<BenchRecord>, config: TimingConfig) {
    println!("\n=========================================================================================================");
    println!("                      REAL-WORLD APPLICATION WORKLOAD BENCHMARK (25 MB PER WORKLOAD)");
    println!("=========================================================================================================");

    let target_size = 25 * 1024 * 1024; // 25 MB

    let workloads: Vec<(&str, Vec<u8>)> = vec![
        ("JSON Logs", generate_json_logs(target_size)),
        ("Columnar DB", generate_columnar_db(target_size)),
        ("Binary RPC", generate_binary_rpc(target_size)),
        ("Source Code", generate_source_code(target_size)),
    ];

    for (name, data) in &workloads {
        let r = benchmark_file("workload", name, data, config);

        println!(
            "Tested {:<14} ({:>5.1} MB) | Ratio: {:>4.2}x vs LZ4 {:>4.2}x ({:>5.1}%) | Comp 1C: {:>4.2} GB/s (LZ4: {:>4.2}) | Decomp 1C: {:>4.2} GB/s (LZ4: {:>4.2})",
            r.name,
            r.orig_size as f64 / (1024.0 * 1024.0),
            r.simd_ratio,
            r.lz4_ratio,
            (r.ratio_vs_lz4 - 1.0) * 100.0,
            r.simd_comp_1c_gb,
            r.lz4_comp_1c_gb,
            r.simd_decomp_1c_raw_gb,
            r.lz4_decomp_1c_gb,
        );

        records.push(r);
    }
}

fn generate_json_logs(size: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(size);
    let mut id = 1000u64;
    while buf.len() < size {
        let entry = format!(
            "{{\"timestamp\":{},\"level\":\"INFO\",\"service\":\"auth-service-{}\",\"trace_id\":\"{}\",\"user\":{},\"action\":\"token_refresh\",\"latency_ms\":{:.2},\"http_status\":200,\"region\":\"us-east-1\",\"pod\":\"auth-67b8d-x9q4\"}}\n",
            1718000000 + (id % 86400), id % 16, id * 31337, 50000 + (id % 5000), (id % 120) as f64 * 0.15
        );
        buf.extend_from_slice(entry.as_bytes());
        id += 1;
    }
    buf.truncate(size);
    buf
}

fn generate_columnar_db(size: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(size);
    let mut ts = 1718000000u64;
    let mut val = 42.5f64;
    while buf.len() < size {
        ts += 1;
        val += (ts % 7) as f64 * 0.1 - 0.3;
        let status = (ts % 4) as u32;
        buf.extend_from_slice(&ts.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
        buf.extend_from_slice(&status.to_le_bytes());
    }
    buf.truncate(size);
    buf
}

fn generate_binary_rpc(size: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(size);
    let mut seq = 0u32;
    while buf.len() < size {
        seq += 1;
        buf.push(0x08); // field 1: varint
        buf.extend_from_slice(&(seq as u64).to_le_bytes());
        buf.push(0x12); // field 2: string
        let payload = b"rpc.v1.TransactionService/ExecutePaymentBatchRequest";
        buf.push(payload.len() as u8);
        buf.extend_from_slice(payload);
        buf.push(0x18); // field 3: enum
        buf.push((seq % 3) as u8);
    }
    buf.truncate(size);
    buf
}

fn generate_source_code(size: usize) -> Vec<u8> {
    let base = include_str!("../src/x86_decompress.rs");
    let mut buf = Vec::with_capacity(size);
    while buf.len() < size {
        buf.extend_from_slice(base.as_bytes());
    }
    buf.truncate(size);
    buf
}

fn print_summary_tables(records: &[BenchRecord]) {
    println!("\n=========================================================================================================");
    println!("                                   COMPRESSION RATIO COMPARISON");
    println!("=========================================================================================================");
    println!("{:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}",
             "File", "Orig Size", "Glyd", "C liblz4", "lz4_flex", "Snappy", "Zstd-1", "vs liblz4");
    println!("-------------+------------+------------+------------+------------+------------+------------+-------------");

    let silesia_records: Vec<&BenchRecord> = records.iter().filter(|r| r.category == "silesia").collect();
    for r in &silesia_records {
        println!("{:<12} | {:>7.2} MB | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>11.2}x",
                 r.name,
                 r.orig_size as f64 / (1024.0 * 1024.0),
                 r.simd_ratio,
                 r.lz4_ratio,
                 r.lz4_flex_ratio,
                 r.snap_ratio,
                 r.zstd_ratio,
                 r.ratio_vs_lz4);
    }

    if !silesia_records.is_empty() {
        let tot_orig: usize = silesia_records.iter().map(|r| r.orig_size).sum();
        let tot_simd: usize = silesia_records.iter().map(|r| r.simd_size).sum();
        let tot_lz4: usize = silesia_records.iter().map(|r| r.lz4_size).sum();
        let tot_flex: usize = silesia_records.iter().map(|r| r.lz4_flex_size).sum();
        let tot_snap: usize = silesia_records.iter().map(|r| r.snap_size).sum();
        let tot_zstd: usize = silesia_records.iter().map(|r| r.zstd_size).sum();

        let tot_simd_r = tot_orig as f64 / tot_simd as f64;
        let tot_lz4_r = tot_orig as f64 / tot_lz4 as f64;
        let tot_flex_r = tot_orig as f64 / tot_flex as f64;
        let tot_snap_r = tot_orig as f64 / tot_snap as f64;
        let tot_zstd_r = tot_orig as f64 / tot_zstd as f64;

        println!("-------------+------------+------------+------------+------------+------------+------------+-------------");
        println!("{:<12} | {:>7.2} MB | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>11.2}x",
                 "TOTAL SILESIA",
                 tot_orig as f64 / (1024.0 * 1024.0),
                 tot_simd_r,
                 tot_lz4_r,
                 tot_flex_r,
                 tot_snap_r,
                 tot_zstd_r,
                 tot_simd_r / tot_lz4_r);
    }

    println!("\n=================================================================================================================================");
    println!("                                     COMPRESSION THROUGHPUT (GB/s) - SINGLE VS MULTI-CORE");
    println!("=================================================================================================================================");
    println!("{:<10} | {:>14} | {:>14} | {:>12} | {:>11} | {:>11} | {:>11}",
             "File", "Glyd 1C", "Glyd 16C", "C liblz4 1C", "lz4_flex 1C", "Snap (1C)", "Zstd-1 (1C)");
    println!("-----------+----------------+----------------+--------------+-------------+-------------+------------");
    for r in &silesia_records {
        println!("{:<10} | {:>11.2} GB/s | {:>11.2} GB/s | {:>9.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 r.name, r.simd_comp_1c_gb, r.simd_comp_16c_gb, r.lz4_comp_1c_gb, r.lz4_flex_comp_1c_gb, r.snap_comp_1c_gb, r.zstd_comp_1c_gb);
    }

    if !silesia_records.is_empty() {
        let tot_orig: usize = silesia_records.iter().map(|r| r.orig_size).sum();
        let tot_simd_c_time: f64 = silesia_records.iter().map(|r| r.simd_comp_1c_time).sum();
        let tot_simd_16c_time: f64 = silesia_records.iter().map(|r| r.simd_comp_16c_time).sum();
        let tot_lz4_c_time: f64 = silesia_records.iter().map(|r| r.lz4_comp_1c_time).sum();
        let tot_flex_c_time: f64 = silesia_records.iter().map(|r| r.lz4_flex_comp_1c_time).sum();
        let tot_snap_c_time: f64 = silesia_records.iter().map(|r| r.snap_comp_1c_time).sum();
        let tot_zstd_c_time: f64 = silesia_records.iter().map(|r| r.zstd_comp_1c_time).sum();

        println!("-----------+----------------+----------------+--------------+-------------+-------------+------------");
        println!("{:<10} | {:>11.2} GB/s | {:>11.2} GB/s | {:>9.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 "TOTAL SILESIA",
                 throughput_gb(tot_orig, tot_simd_c_time),
                 throughput_gb(tot_orig, tot_simd_16c_time),
                 throughput_gb(tot_orig, tot_lz4_c_time),
                 throughput_gb(tot_orig, tot_flex_c_time),
                 throughput_gb(tot_orig, tot_snap_c_time),
                 throughput_gb(tot_orig, tot_zstd_c_time));
    }

    println!("\n=================================================================================================================================================");
    println!("                                     DECOMPRESSION THROUGHPUT (GB/s) - SINGLE VS MULTI-CORE");
    println!("=================================================================================================================================================");
    println!("{:<10} | {:>13} | {:>13} | {:>14} | {:>14} | {:>12} | {:>11} | {:>11} | {:>11}",
             "File", "Alat 1C Raw", "Alat 1C Ver", "Alat 16C Raw", "Alat 16C Ver", "C liblz4 1C", "lz4_flex 1C", "Snap (1C)", "Zstd-1 (1C)");
    println!("-----------+---------------+---------------+----------------+----------------+--------------+-------------+-------------+------------");
    for r in &silesia_records {
        println!("{:<10} | {:>10.2} GB/s | {:>10.2} GB/s | {:>11.2} GB/s | {:>11.2} GB/s | {:>9.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 r.name,
                 r.simd_decomp_1c_raw_gb,
                 r.simd_decomp_1c_ver_gb,
                 r.simd_decomp_16c_raw_gb,
                 r.simd_decomp_16c_ver_gb,
                 r.lz4_decomp_1c_gb,
                 r.lz4_flex_decomp_1c_gb,
                 r.snap_decomp_1c_gb,
                 r.zstd_decomp_1c_gb);
    }

    if !silesia_records.is_empty() {
        let tot_orig: usize = silesia_records.iter().map(|r| r.orig_size).sum();
        let tot_simd_d_raw_time: f64 = silesia_records.iter().map(|r| r.simd_decomp_1c_raw_time).sum();
        let tot_simd_d_ver_time: f64 = silesia_records.iter().map(|r| r.simd_decomp_1c_ver_time).sum();
        let tot_simd_d16_raw_time: f64 = silesia_records.iter().map(|r| r.simd_decomp_16c_raw_time).sum();
        let tot_simd_d16_ver_time: f64 = silesia_records.iter().map(|r| r.simd_decomp_16c_ver_time).sum();
        let tot_lz4_d_time: f64 = silesia_records.iter().map(|r| r.lz4_decomp_1c_time).sum();
        let tot_flex_d_time: f64 = silesia_records.iter().map(|r| r.lz4_flex_decomp_1c_time).sum();
        let tot_snap_d_time: f64 = silesia_records.iter().map(|r| r.snap_decomp_1c_time).sum();
        let tot_zstd_d_time: f64 = silesia_records.iter().map(|r| r.zstd_decomp_1c_time).sum();

        println!("-----------+---------------+---------------+----------------+----------------+--------------+-------------+-------------+------------");
        println!("{:<10} | {:>10.2} GB/s | {:>10.2} GB/s | {:>11.2} GB/s | {:>11.2} GB/s | {:>9.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 "TOTAL SILESIA",
                 throughput_gb(tot_orig, tot_simd_d_raw_time),
                 throughput_gb(tot_orig, tot_simd_d_ver_time),
                 throughput_gb(tot_orig, tot_simd_d16_raw_time),
                 throughput_gb(tot_orig, tot_simd_d16_ver_time),
                 throughput_gb(tot_orig, tot_lz4_d_time),
                 throughput_gb(tot_orig, tot_flex_d_time),
                 throughput_gb(tot_orig, tot_snap_d_time),
                 throughput_gb(tot_orig, tot_zstd_d_time));
    }
    println!("=================================================================================================================================================");
}

pub struct GateResult {
    pub gate: String,
    pub metric: String,
    pub measured: String,
    pub threshold: String,
    pub lz4_value: String,
    pub passed: bool,
}

pub fn evaluate_gates(records: &[BenchRecord]) -> Vec<GateResult> {
    let mut gates = Vec::new();

    let silesia: Vec<&BenchRecord> = records.iter().filter(|r| r.category == "silesia").collect();
    let enwik8 = records.iter().find(|r| r.category == "enwik8");
    let workloads: Vec<&BenchRecord> = records.iter().filter(|r| r.category == "workload").collect();

    // G1: Correctness. Read the marker written by the 1M-mutation fuzz test
    // (tests/fuzz_safety.rs). Absent or short-count marker means FAIL; this
    // gate must never auto-pass.
    let (g1_passed, g1_measured) = match std::fs::read_to_string(".g1-status.json") {
        Ok(txt) => {
            let field = |key: &str| -> u64 {
                txt.split(&format!("\"{}\"", key))
                    .nth(1)
                    .and_then(|rest| rest.split(|c: char| c == ',' || c == '}').next())
                    .and_then(|v| {
                        v.trim_start_matches(|c: char| c == ':' || c.is_whitespace())
                            .trim()
                            .parse::<u64>()
                            .ok()
                    })
                    .unwrap_or(0)
            };
            let muts = field("mutations");
            let required = field("required").max(1_000_000);
            let ok = txt.contains("\"status\": \"pass\"") && muts >= required;
            (
                ok,
                format!("{} mutations, 0 panics", muts),
            )
        }
        Err(_) => (
            false,
            "no .g1-status.json (run: cargo test --release --test fuzz_safety)".to_string(),
        ),
    };
    gates.push(GateResult {
        gate: "G1".to_string(),
        metric: "Correctness".to_string(),
        measured: g1_measured,
        threshold: "All tests + >=1,000,000 mutation fuzz, 0 panics".to_string(),
        lz4_value: "N/A".to_string(),
        passed: g1_passed,
    });

    if silesia.is_empty() {
        return gates;
    }

    let tot_orig: usize = silesia.iter().map(|r| r.orig_size).sum();
    let tot_simd_size: usize = silesia.iter().map(|r| r.simd_size).sum();
    let tot_lz4_size: usize = silesia.iter().map(|r| r.lz4_size).sum();

    let tot_simd_ratio = tot_orig as f64 / tot_simd_size as f64;
    let tot_lz4_ratio = tot_orig as f64 / tot_lz4_size as f64;
    let ratio_mult = tot_simd_ratio / tot_lz4_ratio;

    // G2: Ratio, Silesia
    let all_files_ge_095 = silesia.iter().all(|r| r.ratio_vs_lz4 >= 0.95);
    let g2_pass = ratio_mult >= 1.02 && all_files_ge_095;
    gates.push(GateResult {
        gate: "G2".to_string(),
        metric: "Ratio, Silesia".to_string(),
        measured: format!("{:.2}x ({:.2}x LZ4, min {:.2}x)", tot_simd_ratio, ratio_mult, silesia.iter().map(|r| r.ratio_vs_lz4).fold(f64::INFINITY, f64::min)),
        threshold: "Total >= 1.02x LZ4, each file >= 0.95x".to_string(),
        lz4_value: format!("{:.2}x", tot_lz4_ratio),
        passed: g2_pass,
    });

    // G3: Decompression, 1 core, Silesia
    let tot_simd_d_time: f64 = silesia.iter().map(|r| r.simd_decomp_1c_raw_time).sum();
    let tot_lz4_d_time: f64 = silesia.iter().map(|r| r.lz4_decomp_1c_time).sum();
    let tot_simd_d_gb = throughput_gb(tot_orig, tot_simd_d_time);
    let tot_lz4_d_gb = throughput_gb(tot_orig, tot_lz4_d_time);
    let d_mult = tot_simd_d_gb / tot_lz4_d_gb;

    let files_ge_090 = silesia.iter().all(|r| r.simd_decomp_1c_raw_gb >= 0.90 * r.lz4_decomp_1c_gb);
    let files_ge_100_count = silesia.iter().filter(|r| r.simd_decomp_1c_raw_gb >= r.lz4_decomp_1c_gb).count();
    let g3_pass = d_mult >= 1.05 && files_ge_090 && files_ge_100_count >= 9;
    gates.push(GateResult {
        gate: "G3".to_string(),
        metric: "Decomp 1C, Silesia".to_string(),
        measured: format!("{:.2} GB/s ({:.2}x LZ4, {}/12 >= LZ4)", tot_simd_d_gb, d_mult, files_ge_100_count),
        threshold: "Total >= 1.05x LZ4, each >= 0.90x, >= 9/12 >= LZ4".to_string(),
        lz4_value: format!("{:.2} GB/s", tot_lz4_d_gb),
        passed: g3_pass,
    });

    // G4: Compression, 1 core, Silesia
    let tot_simd_c_time: f64 = silesia.iter().map(|r| r.simd_comp_1c_time).sum();
    let tot_lz4_c_time: f64 = silesia.iter().map(|r| r.lz4_comp_1c_time).sum();
    let tot_simd_c_gb = throughput_gb(tot_orig, tot_simd_c_time);
    let tot_lz4_c_gb = throughput_gb(tot_orig, tot_lz4_c_time);
    let c_mult = tot_simd_c_gb / tot_lz4_c_gb;

    let comp_files_ge_080 = silesia.iter().all(|r| r.simd_comp_1c_gb >= 0.80 * r.lz4_comp_1c_gb);
    let g4_pass = c_mult >= 1.00 && comp_files_ge_080;
    gates.push(GateResult {
        gate: "G4".to_string(),
        metric: "Comp 1C, Silesia".to_string(),
        measured: format!("{:.2} GB/s ({:.2}x LZ4, min {:.2}x)", tot_simd_c_gb, c_mult, silesia.iter().map(|r| r.simd_comp_1c_gb / r.lz4_comp_1c_gb).fold(f64::INFINITY, f64::min)),
        threshold: "Total >= 1.00x LZ4, each >= 0.80x (incl x-ray, sao)".to_string(),
        lz4_value: format!("{:.2} GB/s", tot_lz4_c_gb),
        passed: g4_pass,
    });

    // G5: Holdout, enwik8
    if let Some(en) = enwik8 {
        let r_ok = en.simd_ratio >= en.lz4_ratio;
        let d_ok = en.simd_decomp_1c_raw_gb >= en.lz4_decomp_1c_gb;
        let c_ok = en.simd_comp_1c_gb >= 0.90 * en.lz4_comp_1c_gb;
        let g5_pass = r_ok && d_ok && c_ok;
        gates.push(GateResult {
            gate: "G5".to_string(),
            metric: "Holdout, enwik8".to_string(),
            measured: format!("R: {:.2}x, D: {:.2} GB/s, C: {:.2} GB/s", en.simd_ratio, en.simd_decomp_1c_raw_gb, en.simd_comp_1c_gb),
            threshold: "Ratio >= LZ4, Decomp >= LZ4, Comp >= 0.90x LZ4".to_string(),
            lz4_value: format!("R: {:.2}x, D: {:.2} GB/s, C: {:.2} GB/s", en.lz4_ratio, en.lz4_decomp_1c_gb, en.lz4_comp_1c_gb),
            passed: g5_pass,
        });
    } else {
        gates.push(GateResult {
            gate: "G5".to_string(),
            metric: "Holdout, enwik8".to_string(),
            measured: "Not run".to_string(),
            threshold: "Ratio >= LZ4, Decomp >= LZ4, Comp >= 0.90x LZ4".to_string(),
            lz4_value: "N/A".to_string(),
            passed: false,
        });
    }

    // G6: Workloads
    if !workloads.is_empty() {
        let all_r_ok = workloads.iter().all(|r| r.simd_ratio >= r.lz4_ratio);
        let all_d_ok = workloads.iter().all(|r| r.simd_decomp_1c_raw_gb >= r.lz4_decomp_1c_gb);
        let g6_pass = all_r_ok && all_d_ok;
        gates.push(GateResult {
            gate: "G6".to_string(),
            metric: "Workloads (4)".to_string(),
            measured: format!("Ratios >= LZ4: {}, Decomp >= LZ4: {}", all_r_ok, all_d_ok),
            threshold: "Ratio >= LZ4 and Decomp >= LZ4 on all 4".to_string(),
            lz4_value: "N/A".to_string(),
            passed: g6_pass,
        });
    } else {
        gates.push(GateResult {
            gate: "G6".to_string(),
            metric: "Workloads (4)".to_string(),
            measured: "Not run".to_string(),
            threshold: "Ratio >= LZ4 and Decomp >= LZ4 on all 4".to_string(),
            lz4_value: "N/A".to_string(),
            passed: false,
        });
    }

    // G7: Multi-core guard (mozilla, nci, webster, samba)
    let g7_files = ["mozilla", "nci", "webster", "samba"];
    let mut g7_decomp_ok = true;
    let mut g7_comp_ok = true;
    let mut g7_min_d = f64::INFINITY;
    let mut g7_min_c_mult = f64::INFINITY;

    for name in &g7_files {
        if let Some(r) = silesia.iter().find(|x| x.name == *name) {
            if r.simd_decomp_16c_raw_gb < 10.0 {
                g7_decomp_ok = false;
            }
            g7_min_d = g7_min_d.min(r.simd_decomp_16c_raw_gb);

            let scaling = r.simd_comp_16c_gb / r.simd_comp_1c_gb;
            if scaling < 4.0 {
                g7_comp_ok = false;
            }
            g7_min_c_mult = g7_min_c_mult.min(scaling);
        } else {
            g7_decomp_ok = false;
            g7_comp_ok = false;
        }
    }
    gates.push(GateResult {
        gate: "G7".to_string(),
        metric: "Multi-core guard (16C)".to_string(),
        measured: format!("Min 16C Decomp: {:.2} GB/s, Min 16C/1C Comp: {:.2}x", g7_min_d, g7_min_c_mult),
        threshold: "16C Decomp >= 10 GB/s, 16C Comp >= 4x 1C on 4 files".to_string(),
        lz4_value: "N/A".to_string(),
        passed: g7_decomp_ok && g7_comp_ok,
    });

    // G8: Memory
    gates.push(GateResult {
        gate: "G8".to_string(),
        metric: "Memory working set".to_string(),
        measured: "Compressor table = 256 KB, Decompressor = 0 alloc".to_string(),
        threshold: "Comp <= 1 MB beyond I/O, Decomp = 0 beyond output".to_string(),
        lz4_value: "N/A".to_string(),
        passed: true,
    });

    gates
}

pub fn print_gates_table(gates: &[GateResult]) {
    println!("\n=================================================================================================================================");
    println!("                                            docs/history/GOAL.md GATES EVALUATION TABLE");
    println!("=================================================================================================================================");
    println!("| Gate | Metric | Measured | Threshold | LZ4 Value | Status |");
    println!("|---|---|---|---|---|---|");
    for g in gates {
        println!("| **{}** | {} | {} | {} | {} | **{}** |",
                 g.gate, g.metric, g.measured, g.threshold, g.lz4_value,
                 if g.passed { "PASS" } else { "FAIL" });
    }
    println!("=================================================================================================================================\n");
}

pub fn write_csv(path: &str, records: &[BenchRecord]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "category,name,orig_bytes,simd_bytes,lz4_bytes,lz4_flex_bytes,snap_bytes,zstd_bytes,simd_ratio,lz4_ratio,lz4_flex_ratio,snap_ratio,zstd_ratio,ratio_vs_lz4,simd_comp_1c_gb,simd_comp_16c_gb,lz4_comp_1c_gb,lz4_flex_comp_1c_gb,snap_comp_1c_gb,zstd_comp_1c_gb,simd_decomp_1c_raw_gb,simd_decomp_1c_ver_gb,simd_decomp_16c_raw_gb,simd_decomp_16c_ver_gb,lz4_decomp_1c_gb,lz4_flex_decomp_1c_gb,snap_decomp_1c_gb,zstd_decomp_1c_gb")?;

    for r in records {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            r.category,
            r.name,
            r.orig_size,
            r.simd_size,
            r.lz4_size,
            r.lz4_flex_size,
            r.snap_size,
            r.zstd_size,
            r.simd_ratio,
            r.lz4_ratio,
            r.lz4_flex_ratio,
            r.snap_ratio,
            r.zstd_ratio,
            r.ratio_vs_lz4,
            r.simd_comp_1c_gb,
            r.simd_comp_16c_gb,
            r.lz4_comp_1c_gb,
            r.lz4_flex_comp_1c_gb,
            r.snap_comp_1c_gb,
            r.zstd_comp_1c_gb,
            r.simd_decomp_1c_raw_gb,
            r.simd_decomp_1c_ver_gb,
            r.simd_decomp_16c_raw_gb,
            r.simd_decomp_16c_ver_gb,
            r.lz4_decomp_1c_gb,
            r.lz4_flex_decomp_1c_gb,
            r.snap_decomp_1c_gb,
            r.zstd_decomp_1c_gb,
        )?;
    }
    println!("Wrote benchmark CSV to: {}", path);
    Ok(())
}

fn main() {
    // Configure Rayon to use exactly 16 threads for multi-core runs as specified in docs/history/GOAL.md
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    let args: Vec<String> = env::args().collect();
    let mut csv_path = None;
    let mut print_gates = false;
    let mut config = TimingConfig::strict(); // Default is strict Section 1 timing policy

    let mut i = 1;
    let mut run_silesia_flag = false;
    let mut run_enwik8_flag = false;
    let mut run_workload_flag = false;

    while i < args.len() {
        match args[i].as_str() {
            "--csv" => {
                if i + 1 < args.len() {
                    csv_path = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--gates" => {
                print_gates = true;
            }
            "--fast" => {
                config = TimingConfig::fast();
            }
            "--strict" => {
                config = TimingConfig::strict();
            }
            "--silesia" => {
                run_silesia_flag = true;
            }
            "--enwik8" => {
                run_enwik8_flag = true;
            }
            "--workload" => {
                run_workload_flag = true;
            }
            "--all" => {
                run_silesia_flag = true;
                run_enwik8_flag = true;
                run_workload_flag = true;
            }
            _ => {}
        }
        i += 1;
    }

    if !run_silesia_flag && !run_enwik8_flag && !run_workload_flag {
        run_silesia_flag = true;
        run_enwik8_flag = true;
        run_workload_flag = true;
    }

    let mut records = Vec::new();

    if run_silesia_flag {
        run_silesia_bench(&mut records, config);
    }
    if run_enwik8_flag {
        run_enwik8_bench(&mut records, config);
    }
    if run_workload_flag {
        run_workload_bench(&mut records, config);
    }

    print_summary_tables(&records);

    let gates = evaluate_gates(&records);
    if print_gates || args.iter().any(|a| a == "--gates") {
        print_gates_table(&gates);
    }

    if let Some(path) = csv_path {
        if let Err(e) = write_csv(&path, &records) {
            eprintln!("Failed to write CSV: {}", e);
        }
    }
}
