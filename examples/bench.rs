use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn throughput_gb(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / secs
}

#[derive(Clone, Debug)]
pub struct BenchRecord {
    pub category: String,
    pub name: String,
    pub orig_size: usize,
    pub simd_size: usize,
    pub simd_ratio: f64,
    pub lz4_size: usize,
    pub lz4_ratio: f64,
    pub snap_size: usize,
    pub snap_ratio: f64,
    pub zstd_size: usize,
    pub zstd_ratio: f64,
    pub ratio_deficit_vs_lz4_pct: f64,

    // Compression throughput (GB/s)
    pub simd_comp_1c_gb: f64,
    pub simd_comp_16c_gb: f64,
    pub lz4_comp_1c_gb: f64,
    pub snap_comp_1c_gb: f64,
    pub zstd_comp_1c_gb: f64,

    // Decompression throughput (GB/s)
    pub simd_decomp_1c_raw_gb: f64,
    pub simd_decomp_1c_ver_gb: f64,
    pub simd_decomp_16c_raw_gb: f64,
    pub simd_decomp_16c_ver_gb: f64,
    pub lz4_decomp_1c_gb: f64,
    pub snap_decomp_1c_gb: f64,
    pub zstd_decomp_1c_gb: f64,
}

fn run_silesia_bench(records: &mut Vec<BenchRecord>) {
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

        let comp_iters = (300 * 1024 * 1024 / len).max(3);
        let decomp_iters = (600 * 1024 * 1024 / len).max(5);
        let par_iters = (1200 * 1024 * 1024 / len).max(10);

        // 1. SIMD Compression (1-Core)
        let mut simd_comp = Vec::new();
        let start = Instant::now();
        for _ in 0..comp_iters {
            simd_comp = simd_stream_codec::compress(&data);
        }
        let simd_comp_1c_gb = throughput_gb(len * comp_iters, start.elapsed().as_secs_f64());
        total_simd_size += simd_comp.len();

        // 2. SIMD Parallel Compression (16-Core)
        let mut simd_comp_par = Vec::new();
        let start = Instant::now();
        for _ in 0..comp_iters {
            simd_comp_par = simd_stream_codec::compress_parallel(&data);
        }
        let simd_comp_16c_gb = throughput_gb(len * comp_iters, start.elapsed().as_secs_f64());

        // 3. LZ4 Compression (1-Core)
        let mut lz4_comp = Vec::new();
        let start = Instant::now();
        for _ in 0..comp_iters {
            lz4_comp = lz4_flex::compress(&data);
        }
        let lz4_comp_1c_gb = throughput_gb(len * comp_iters, start.elapsed().as_secs_f64());
        total_lz4_size += lz4_comp.len();

        // 4. Snappy Compression (1-Core)
        let mut snap_comp = Vec::new();
        let mut snap_encoder = snap::raw::Encoder::new();
        let start = Instant::now();
        for _ in 0..comp_iters {
            snap_comp = snap_encoder.compress_vec(&data).unwrap();
        }
        let snap_comp_1c_gb = throughput_gb(len * comp_iters, start.elapsed().as_secs_f64());
        total_snap_size += snap_comp.len();

        // 5. Zstd Level 1 Compression (1-Core)
        let mut zstd_comp = Vec::new();
        let zstd_c_iters = comp_iters.min(10);
        let start = Instant::now();
        for _ in 0..zstd_c_iters {
            zstd_comp = zstd::encode_all(&data[..], 1).unwrap();
        }
        let zstd_comp_1c_gb = throughput_gb(len * zstd_c_iters, start.elapsed().as_secs_f64());
        total_zstd_size += zstd_comp.len();

        // --- Decompression ---
        let mut decomp_dest = vec![0u8; len + 128];

        // 1A. SIMD 1-Core RAW
        let start = Instant::now();
        for _ in 0..decomp_iters {
            simd_stream_codec::decompress_into_raw(&simd_comp, &mut decomp_dest).expect("1C Raw failed");
        }
        let simd_decomp_1core_raw_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 1B. SIMD 1-Core VERIFIED
        let start = Instant::now();
        for _ in 0..decomp_iters {
            simd_stream_codec::decompress_into(&simd_comp, &mut decomp_dest).expect("1C Ver failed");
        }
        let simd_decomp_1core_ver_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 2A. SIMD 16-Core RAW
        let mut par_dest = vec![0u8; len + 128];
        let start = Instant::now();
        for _ in 0..par_iters {
            simd_stream_codec::decompress_parallel_into_raw(&simd_comp_par, &mut par_dest).expect("Par Raw failed");
        }
        let simd_decomp_par_raw_gb = throughput_gb(len * par_iters, start.elapsed().as_secs_f64());

        // 2B. SIMD 16-Core VERIFIED
        let start = Instant::now();
        for _ in 0..par_iters {
            simd_stream_codec::decompress_parallel_into(&simd_comp_par, &mut par_dest).expect("Par Ver failed");
        }
        let simd_decomp_par_ver_gb = throughput_gb(len * par_iters, start.elapsed().as_secs_f64());

        // 3. LZ4 Decompression (1-Core)
        let mut lz4_dest = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..decomp_iters {
            let _ = lz4_flex::decompress_into(&lz4_comp, &mut lz4_dest);
        }
        let lz4_decomp_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 4. Snappy Decompression (1-Core)
        let mut snap_decoder = snap::raw::Decoder::new();
        let mut snap_dest = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..decomp_iters {
            let _ = snap_decoder.decompress(&snap_comp, &mut snap_dest);
        }
        let snap_decomp_gb = throughput_gb(len * decomp_iters, start.elapsed().as_secs_f64());

        // 5. Zstd Decompression (1-Core)
        let zstd_d_iters = decomp_iters.min(15);
        let start = Instant::now();
        for _ in 0..zstd_d_iters {
            let _ = zstd::decode_all(&zstd_comp[..]);
        }
        let zstd_decomp_gb = throughput_gb(len * zstd_d_iters, start.elapsed().as_secs_f64());

        let simd_ratio = len as f64 / simd_comp.len() as f64;
        let lz4_ratio = len as f64 / lz4_comp.len() as f64;
        let snap_ratio = len as f64 / snap_comp.len() as f64;
        let zstd_ratio = len as f64 / zstd_comp.len() as f64;
        let ratio_deficit_vs_lz4_pct = ((simd_ratio - lz4_ratio) / lz4_ratio) * 100.0;

        let record = BenchRecord {
            category: "silesia".to_string(),
            name: file_name.to_string(),
            orig_size: len,
            simd_size: simd_comp.len(),
            simd_ratio,
            lz4_size: lz4_comp.len(),
            lz4_ratio,
            snap_size: snap_comp.len(),
            snap_ratio,
            zstd_size: zstd_comp.len(),
            zstd_ratio,
            ratio_deficit_vs_lz4_pct,
            simd_comp_1c_gb,
            simd_comp_16c_gb,
            lz4_comp_1c_gb,
            snap_comp_1c_gb,
            zstd_comp_1c_gb,
            simd_decomp_1c_raw_gb: simd_decomp_1core_raw_gb,
            simd_decomp_1c_ver_gb: simd_decomp_1core_ver_gb,
            simd_decomp_16c_raw_gb: simd_decomp_par_raw_gb,
            simd_decomp_16c_ver_gb: simd_decomp_par_ver_gb,
            lz4_decomp_1c_gb: lz4_decomp_gb,
            snap_decomp_1c_gb: snap_decomp_gb,
            zstd_decomp_1c_gb: zstd_decomp_gb,
        };

        println!("Tested {:<10} ({:>5.1} MB) | Comp 1C: {:>4.2} GB/s (16C: {:>4.1} GB/s) | Decomp 1C: {:>4.1} GB/s (16C: {:>4.1} GB/s) | LZ4 1C: {:>4.1} GB/s",
                 file_name, len as f64 / (1024.0 * 1024.0), simd_comp_1c_gb, simd_comp_16c_gb, simd_decomp_1core_raw_gb, simd_decomp_par_ver_gb, lz4_decomp_gb);

        records.push(record);
    }

    println!("\n=========================================================================================================");
    println!("                                   COMPRESSION RATIO COMPARISON");
    println!("=========================================================================================================");
    println!("{:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>14}",
             "File", "Orig Size", "SIMD (Ours)", "LZ4", "Snappy", "Zstd-1", "Deficit vs LZ4");
    println!("-------------+------------+------------+------------+------------+------------+---------------");
    for r in records.iter().filter(|r| r.category == "silesia") {
        println!("{:<12} | {:>7.2} MB | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>13.1}%",
                 r.name,
                 r.orig_size as f64 / (1024.0 * 1024.0),
                 r.simd_ratio,
                 r.lz4_ratio,
                 r.snap_ratio,
                 r.zstd_ratio,
                 r.ratio_deficit_vs_lz4_pct);
    }
    let tot_simd_r = total_orig as f64 / total_simd_size as f64;
    let tot_lz4_r = total_orig as f64 / total_lz4_size as f64;
    let tot_snap_r = total_orig as f64 / total_snap_size as f64;
    let tot_zstd_r = total_orig as f64 / total_zstd_size as f64;
    let tot_def = ((tot_simd_r - tot_lz4_r) / tot_lz4_r) * 100.0;
    println!("-------------+------------+------------+------------+------------+------------+---------------");
    println!("{:<12} | {:>7.2} MB | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>8.2}x | {:>13.1}%",
             "TOTAL SILESIA",
             total_orig as f64 / (1024.0 * 1024.0),
             tot_simd_r,
             tot_lz4_r,
             tot_snap_r,
             tot_zstd_r,
             tot_def);

    println!("\n=================================================================================================================================");
    println!("                                     COMPRESSION THROUGHPUT (GB/s) - SINGLE VS MULTI-CORE");
    println!("=================================================================================================================================");
    println!("{:<10} | {:>14} | {:>14} | {:>11} | {:>11} | {:>11}",
             "File", "SIMD 1C Comp", "SIMD 16C Comp", "LZ4 (1C)", "Snap (1C)", "Zstd-1 (1C)");
    println!("-----------+----------------+----------------+-------------+-------------+------------");
    for r in records.iter().filter(|r| r.category == "silesia") {
        println!("{:<10} | {:>11.2} GB/s | {:>11.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 r.name, r.simd_comp_1c_gb, r.simd_comp_16c_gb, r.lz4_comp_1c_gb, r.snap_comp_1c_gb, r.zstd_comp_1c_gb);
    }

    println!("\n=================================================================================================================================");
    println!("                                     DECOMPRESSION THROUGHPUT (GB/s) - SINGLE VS MULTI-CORE");
    println!("=================================================================================================================================");
    println!("{:<10} | {:>14} | {:>14} | {:>14} | {:>14} | {:>11} | {:>11} | {:>11}",
             "File", "SIMD 1C Raw", "SIMD 1C Ver", "SIMD 16C Raw", "SIMD 16C Ver", "LZ4 (1C)", "Snap (1C)", "Zstd-1 (1C)");
    println!("-----------+----------------+----------------+----------------+----------------+-------------+-------------+------------");
    for r in records.iter().filter(|r| r.category == "silesia") {
        println!("{:<10} | {:>11.2} GB/s | {:>11.2} GB/s | {:>11.2} GB/s | {:>11.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s | {:>8.2} GB/s",
                 r.name,
                 r.simd_decomp_1c_raw_gb,
                 r.simd_decomp_1c_ver_gb,
                 r.simd_decomp_16c_raw_gb,
                 r.simd_decomp_16c_ver_gb,
                 r.lz4_decomp_1c_gb,
                 r.snap_decomp_1c_gb,
                 r.zstd_decomp_1c_gb);
    }
    println!("=================================================================================================================================");
}

fn run_workload_bench(records: &mut Vec<BenchRecord>) {
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

    println!("{:<14} | {:<42} | {:>6} | {:>14} | {:>14} | {:>10} | {:>10} | {:>14}",
             "Workload", "Application Domain", "Ratio", "SIMD 1C Raw", "SIMD 16C Ver", "LZ4 1C", "Snap 1C", "Deficit vs LZ4");
    println!("---------------+--------------------------------------------+--------+----------------+----------------+------------+------------+---------------");

    for (name, domain, data) in &workloads {
        let len = data.len();
        let iters = 15;

        // Compression
        let mut _simd_comp = Vec::new();
        let start = Instant::now();
        for _ in 0..iters {
            _simd_comp = simd_stream_codec::compress(data);
        }
        let simd_comp_1c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let mut simd_par = Vec::new();
        let start = Instant::now();
        for _ in 0..iters {
            simd_par = simd_stream_codec::compress_parallel(data);
        }
        let simd_comp_16c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let mut lz4_comp = Vec::new();
        let start = Instant::now();
        for _ in 0..iters {
            lz4_comp = lz4_flex::compress(data);
        }
        let lz4_comp_1c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let mut snap_enc = snap::raw::Encoder::new();
        let mut snap_comp = Vec::new();
        let start = Instant::now();
        for _ in 0..iters {
            snap_comp = snap_enc.compress_vec(data).unwrap();
        }
        let snap_comp_1c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let zstd_c_iters = 5;
        let mut zstd_comp = Vec::new();
        let start = Instant::now();
        for _ in 0..zstd_c_iters {
            zstd_comp = zstd::encode_all(&data[..], 1).unwrap();
        }
        let zstd_comp_1c_gb = throughput_gb(len * zstd_c_iters, start.elapsed().as_secs_f64());

        // Decompression
        let mut dst = vec![0u8; len + 128];

        let start = Instant::now();
        for _ in 0..iters {
            simd_stream_codec::decompress_into_raw(&simd_par, &mut dst).unwrap();
        }
        let simd_1c_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let start = Instant::now();
        for _ in 0..iters {
            simd_stream_codec::decompress_into(&simd_par, &mut dst).unwrap();
        }
        let simd_1c_ver_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let start = Instant::now();
        for _ in 0..iters {
            simd_stream_codec::decompress_parallel_into_raw(&simd_par, &mut dst).unwrap();
        }
        let simd_16c_raw_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let start = Instant::now();
        for _ in 0..iters {
            simd_stream_codec::decompress_parallel_into(&simd_par, &mut dst).unwrap();
        }
        let simd_16c_ver_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let mut lz4_dst = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..iters {
            lz4_flex::decompress_into(&lz4_comp, &mut lz4_dst).unwrap();
        }
        let lz4_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let mut snap_dec = snap::raw::Decoder::new();
        let mut snap_dst = vec![0u8; len];
        let start = Instant::now();
        for _ in 0..iters {
            snap_dec.decompress(&snap_comp, &mut snap_dst).unwrap();
        }
        let snap_gb = throughput_gb(len * iters, start.elapsed().as_secs_f64());

        let mut _zstd_dst = Vec::new();
        let zstd_d_iters = 10;
        let start = Instant::now();
        for _ in 0..zstd_d_iters {
            _zstd_dst = zstd::decode_all(&zstd_comp[..]).unwrap();
        }
        let zstd_d_gb = throughput_gb(len * zstd_d_iters, start.elapsed().as_secs_f64());

        let simd_ratio = len as f64 / simd_par.len() as f64;
        let lz4_ratio = len as f64 / lz4_comp.len() as f64;
        let snap_ratio = len as f64 / snap_comp.len() as f64;
        let zstd_ratio = len as f64 / zstd_comp.len() as f64;
        let ratio_deficit_vs_lz4_pct = ((simd_ratio - lz4_ratio) / lz4_ratio) * 100.0;

        println!("{:<14} | {:<42} | {:>5.2}x | {:>11.2} GB/s | {:>11.2} GB/s | {:>7.2} GB/s | {:>7.2} GB/s | {:>13.1}%",
                 name, domain, simd_ratio, simd_1c_gb, simd_16c_ver_gb, lz4_gb, snap_gb, ratio_deficit_vs_lz4_pct);

        records.push(BenchRecord {
            category: "workload".to_string(),
            name: name.to_string(),
            orig_size: len,
            simd_size: simd_par.len(),
            simd_ratio,
            lz4_size: lz4_comp.len(),
            lz4_ratio,
            snap_size: snap_comp.len(),
            snap_ratio,
            zstd_size: zstd_comp.len(),
            zstd_ratio,
            ratio_deficit_vs_lz4_pct,
            simd_comp_1c_gb,
            simd_comp_16c_gb,
            lz4_comp_1c_gb,
            snap_comp_1c_gb,
            zstd_comp_1c_gb,
            simd_decomp_1c_raw_gb: simd_1c_gb,
            simd_decomp_1c_ver_gb: simd_1c_ver_gb,
            simd_decomp_16c_raw_gb: simd_16c_raw_gb,
            simd_decomp_16c_ver_gb: simd_16c_ver_gb,
            lz4_decomp_1c_gb: lz4_gb,
            snap_decomp_1c_gb: snap_gb,
            zstd_decomp_1c_gb: zstd_d_gb,
        });
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

fn write_csv(path: &str, records: &[BenchRecord]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "category,name,orig_bytes,simd_comp_bytes,simd_ratio,lz4_ratio,snap_ratio,zstd_ratio,ratio_deficit_pct,simd_comp_1c_gb,simd_comp_16c_gb,lz4_comp_1c_gb,snap_comp_1c_gb,zstd_comp_1c_gb,simd_decomp_1c_raw_gb,simd_decomp_1c_ver_gb,simd_decomp_16c_raw_gb,simd_decomp_16c_ver_gb,lz4_decomp_1c_gb,snap_decomp_1c_gb,zstd_decomp_1c_gb")?;
    for r in records {
        writeln!(
            file,
            "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            r.category,
            r.name,
            r.orig_size,
            r.simd_size,
            r.simd_ratio,
            r.lz4_ratio,
            r.snap_ratio,
            r.zstd_ratio,
            r.ratio_deficit_vs_lz4_pct,
            r.simd_comp_1c_gb,
            r.simd_comp_16c_gb,
            r.lz4_comp_1c_gb,
            r.snap_comp_1c_gb,
            r.zstd_comp_1c_gb,
            r.simd_decomp_1c_raw_gb,
            r.simd_decomp_1c_ver_gb,
            r.simd_decomp_16c_raw_gb,
            r.simd_decomp_16c_ver_gb,
            r.lz4_decomp_1c_gb,
            r.snap_decomp_1c_gb,
            r.zstd_decomp_1c_gb,
        )?;
    }
    println!("Wrote benchmark results to CSV: {}", path);
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut csv_path = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--csv" && i + 1 < args.len() {
            csv_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }

    let run_silesia = !args.iter().any(|a| a == "--workload");
    let run_workload = !args.iter().any(|a| a == "--silesia");

    let mut records = Vec::new();

    if run_silesia {
        run_silesia_bench(&mut records);
    }
    if run_workload {
        run_workload_bench(&mut records);
    }

    if let Some(path) = csv_path {
        if let Err(e) = write_csv(&path, &records) {
            eprintln!("Failed to write CSV: {}", e);
        }
    }
}
