use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn throughput_gb(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / secs
}

struct SilesiaResult {
    file_name: String,
    orig_size: usize,
    simd_size: usize,
    simd_decomp_1core_raw_gb: f64,
    simd_decomp_1core_ver_gb: f64,
    simd_decomp_par_raw_gb: f64,
    simd_decomp_par_ver_gb: f64,
    lz4_size: usize,
    lz4_decomp_gb: f64,
    snap_size: usize,
    snap_decomp_gb: f64,
    zstd_size: usize,
    zstd_decomp_gb: f64,
}

fn run_silesia_bench() {
    let num_threads = rayon::current_num_threads();
    println!("=========================================================================================================");
    println!("             OFFICIAL SILESIA COMPRESSION CORPUS BENCHMARK (AVX-512 + MULTI-CORE)");
    println!("             Hardware: AMD Ryzen 9 7950X3D (Zen 4, 16-Core / 32-Thread) | Rayon Threads: {}", num_threads);
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

    let mut results = Vec::new();
    let mut total_orig = 0usize;
    let mut total_simd_size = 0usize;
    let mut total_lz4_size = 0usize;
    let mut total_snap_size = 0usize;
    let mut total_zstd_size = 0usize;

    for file_name in &files {
        let path = corpus_dir.join(file_name);
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).expect("Failed to read corpus file");
        let len = data.len();
        total_orig += len;

        // 1. SIMD Stream Codec (Sequential)
        let simd_comp = simd_stream_codec::compress(&data);
        total_simd_size += simd_comp.len();

        let decomp_iters = (600 * 1024 * 1024 / len).max(5);
        let mut decomp_dest = vec![0u8; len + 128];

        // 1A. 1-Core RAW
        let start = Instant::now();
        for _ in 0..decomp_iters {
            simd_stream_codec::decompress_into_raw(&simd_comp, &mut decomp_dest).expect("1C Raw failed");
        }
        let simd_decomp_1core_raw_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 1B. 1-Core VERIFIED (AVX2 Checksum)
        let start = Instant::now();
        for _ in 0..decomp_iters {
            simd_stream_codec::decompress_into(&simd_comp, &mut decomp_dest).expect("1C Ver failed");
        }
        let simd_decomp_1core_ver_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 2. SIMD Stream Codec Multi-Core (Pre-allocated, Rayon Threads)
        let simd_comp_par = simd_stream_codec::compress_parallel(&data);
        let par_iters = (1200 * 1024 * 1024 / len).max(10);
        let mut par_dest = vec![0u8; len + 128];

        // 2A. Multi-Core RAW
        let start = Instant::now();
        for _ in 0..par_iters {
            simd_stream_codec::decompress_parallel_into_raw(&simd_comp_par, &mut par_dest).expect("Par Raw failed");
        }
        let simd_decomp_par_raw_gb = throughput_gb(len * par_iters, start.elapsed().as_secs_f64());

        // 2B. Multi-Core VERIFIED (AVX2 Checksum)
        let start = Instant::now();
        for _ in 0..par_iters {
            simd_stream_codec::decompress_parallel_into(&simd_comp_par, &mut par_dest).expect("Par Ver failed");
        }
        let simd_decomp_par_ver_gb = throughput_gb(len * par_iters, start.elapsed().as_secs_f64());

        // 3. LZ4 Baseline
        let lz4_comp = lz4_flex::compress(&data);
        total_lz4_size += lz4_comp.len();
        let mut lz4_dest = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..decomp_iters {
            let _ = lz4_flex::decompress_into(&lz4_comp, &mut lz4_dest);
        }
        let lz4_decomp_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 4. Google Snappy Baseline
        let mut snap_encoder = snap::raw::Encoder::new();
        let snap_comp = snap_encoder.compress_vec(&data).unwrap();
        total_snap_size += snap_comp.len();
        let mut snap_decoder = snap::raw::Decoder::new();
        let mut snap_dest = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..decomp_iters {
            let _ = snap_decoder.decompress(&snap_comp, &mut snap_dest);
        }
        let snap_decomp_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 5. Meta Zstandard Level 1 Baseline
        let zstd_comp = zstd::encode_all(&data[..], 1).unwrap();
        total_zstd_size += zstd_comp.len();
        let zstd_iters = decomp_iters.min(15);
        let start = Instant::now();
        for _ in 0..zstd_iters {
            let _ = zstd::decode_all(&zstd_comp[..]);
        }
        let zstd_decomp_gb = throughput_gb(len * zstd_iters, start.elapsed().as_secs_f64());

        results.push(SilesiaResult {
            file_name: file_name.to_string(),
            orig_size: len,
            simd_size: simd_comp.len(),
            simd_decomp_1core_raw_gb,
            simd_decomp_1core_ver_gb,
            simd_decomp_par_raw_gb,
            simd_decomp_par_ver_gb,
            lz4_size: lz4_comp.len(),
            lz4_decomp_gb,
            snap_size: snap_comp.len(),
            snap_decomp_gb,
            zstd_size: zstd_comp.len(),
            zstd_decomp_gb,
        });

        println!("Tested {:<10} ({:>5.1} MB) | 1-Core: {:>5.1} GB/s | 16-Core RAW: {:>5.1} GB/s (Ver: {:>5.1} GB/s) | LZ4: {:>5.1} GB/s",
                 file_name, len as f64 / (1024.0 * 1024.0), simd_decomp_1core_raw_gb, simd_decomp_par_raw_gb, simd_decomp_par_ver_gb, lz4_decomp_gb);
    }

    println!("\n=========================================================================================================");
    println!("                                   COMPRESSION RATIO COMPARISON");
    println!("=========================================================================================================");
    println!("{:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
             "File", "Orig Size", "SIMD (Ours)", "LZ4", "Snappy", "Zstd-1");
    println!("-------------+------------+------------+------------+------------+-----------");
    for r in &results {
        println!("{:<12} | {:>7.2} MB | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x",
                 r.file_name,
                 r.orig_size as f64 / (1024.0 * 1024.0),
                 r.orig_size as f64 / r.simd_size as f64,
                 r.orig_size as f64 / r.lz4_size as f64,
                 r.orig_size as f64 / r.snap_size as f64,
                 r.orig_size as f64 / r.zstd_size as f64);
    }
    println!("-------------+------------+------------+------------+------------+-----------");
    println!("{:<12} | {:>7.2} MB | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x",
             "TOTAL SILESIA",
             total_orig as f64 / (1024.0 * 1024.0),
             total_orig as f64 / total_simd_size as f64,
             total_orig as f64 / total_lz4_size as f64,
             total_orig as f64 / total_snap_size as f64,
             total_orig as f64 / total_zstd_size as f64);

    println!("\n=================================================================================================================================");
    println!("                                     DECOMPRESSION THROUGHPUT (GB/s) - SINGLE VS MULTI-CORE");
    println!("=================================================================================================================================");
    println!("{:<10} | {:>14} | {:>14} | {:>14} | {:>14} | {:>11} | {:>11} | {:>11}",
             "File", "SIMD 1C Raw", "SIMD 1C Ver", "SIMD 16C Raw", "SIMD 16C Ver", "LZ4 (1C)", "Snap (1C)", "Zstd-1 (1C)");
    println!("-----------+----------------+----------------+----------------+----------------+-------------+-------------+------------");
    for r in &results {
        println!("{:<10} | {:>11.2} GB/s | {:>11.2} GB/s | {:>11.2} GB/s | {:>11.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 r.file_name,
                 r.simd_decomp_1core_raw_gb,
                 r.simd_decomp_1core_ver_gb,
                 r.simd_decomp_par_raw_gb,
                 r.simd_decomp_par_ver_gb,
                 r.lz4_decomp_gb,
                 r.snap_decomp_gb,
                 r.zstd_decomp_gb);
    }
    println!("=================================================================================================================================");
}

fn run_workload_bench() {
    println!("\n=================================================================================================================================");
    println!("                      REAL-WORLD APPLICATION WORKLOAD BENCHMARK (25 MB PER WORKLOAD)");
    println!("                      Hardware: AMD Ryzen 9 7950X3D (Zen 4, 16C / 32T) | Rayon Threads: {}", rayon::current_num_threads());
    println!("=================================================================================================================================");

    let target_size = 25 * 1024 * 1024; // 25 MB

    let workloads: Vec<(&str, &str, Vec<u8>)> = vec![
        ("JSON Logs", "Kubernetes / CloudWatch JSON structured logs", generate_json_logs(target_size)),
        ("Columnar DB", "Parquet / ClickHouse timestamp & metric columns", generate_columnar_db(target_size)),
        ("Binary RPC", "Protobuf / gRPC microservice packed payloads", generate_binary_rpc(target_size)),
        ("Source Code", "Codebase repositories, ASTs, and syntax trees", generate_source_code(target_size)),
    ];

    println!("{:<14} | {:<42} | {:>6} | {:>14} | {:>14} | {:>10} | {:>10} | {:>8}",
             "Workload", "Application Domain", "Ratio", "SIMD 1C Raw", "SIMD 16C Ver", "LZ4 1C", "Snap 1C", "Speedup");
    println!("---------------+--------------------------------------------+--------+----------------+----------------+------------+------------+---------");

    for (name, domain, data) in &workloads {
        let len = data.len();
        let simd_par = simd_stream_codec::compress_parallel(data);
        let simd_ratio = len as f64 / simd_par.len() as f64;

        let iters = 20;
        let mut dst = vec![0u8; len + 128];

        // 1. SIMD 1C Raw
        let start = Instant::now();
        for _ in 0..iters {
            simd_stream_codec::decompress_into_raw(&simd_par, &mut dst).unwrap();
        }
        let simd_1c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        // 2. SIMD 16C Ver
        let start = Instant::now();
        for _ in 0..iters {
            simd_stream_codec::decompress_parallel_into(&simd_par, &mut dst).unwrap();
        }
        let simd_16c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        // 3. LZ4 1C
        let lz4_comp = lz4_flex::compress(data);
        let mut lz4_dst = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..iters {
            lz4_flex::decompress_into(&lz4_comp, &mut lz4_dst).unwrap();
        }
        let lz4_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        // 4. Snappy 1C
        let mut snap_enc = snap::raw::Encoder::new();
        let snap_comp = snap_enc.compress_vec(data).unwrap();
        let mut snap_dec = snap::raw::Decoder::new();
        let mut snap_dst = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..iters {
            snap_dec.decompress(&snap_comp, &mut snap_dst).unwrap();
        }
        let snap_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let speedup = simd_16c_gb / lz4_gb;
        println!("{:<14} | {:<42} | {:>5.2}x | {:>11.2} GB/s | {:>11.2} GB/s | {:>7.2} GB/s | {:>7.2} GB/s | {:>6.1}x",
                 name, domain, simd_ratio, simd_1c_gb, simd_16c_gb, lz4_gb, snap_gb, speedup);
    }
    println!("=================================================================================================================================\n");
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--silesia") {
        run_silesia_bench();
    } else if args.iter().any(|a| a == "--workload") {
        run_workload_bench();
    } else {
        run_silesia_bench();
        run_workload_bench();
    }
}
