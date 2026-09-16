use std::env;
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::Instant;

fn print_usage() {
    eprintln!(r#"Alatirok High-Performance SIMD Stream Codec (AVX-512 / AVX2 / GPU)
Usage: alatirok [OPTIONS] [INPUT] [-o OUTPUT]

Options:
    -c, --compress         Compress input (default if output is .alk)
    -d, --decompress       Decompress input (default if input is .alk)
    -m, --multi-core       Use multi-core parallel engine (default)
    -1, --single-core      Force single-core sequential engine
    -o, --output <FILE>    Specify destination output file (defaults to stdout if piped)
    -b, --bench            Benchmark compression and decompression throughput
    -v, --version          Print version and CPU SIMD hardware capabilities
    -h, --help             Print this help message

Examples:
    alatirok input.tar -o input.tar.alk
    alatirok -d input.tar.alk -o input.tar
    cat large.json | alatirok -c > large.json.alk
    cat large.json.alk | alatirok -d > large.json
    alatirok -b dataset.bin
"#);
}

fn print_version() {
    println!("alatirok 0.1.0");
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
        let avx2 = is_x86_feature_detected!("avx2");
        let bmi2 = is_x86_feature_detected!("bmi2");
        println!("Hardware SIMD: AVX-512={}, AVX2={}, BMI2={}", avx512, avx2, bmi2);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("Hardware SIMD: Scalar Fallback");
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut mode_compress = None;
    let mut multi_core = true;
    let mut benchmark_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "-v" | "--version" => {
                print_version();
                return Ok(());
            }
            "-c" | "--compress" => mode_compress = Some(true),
            "-d" | "--decompress" => mode_compress = Some(false),
            "-m" | "--multi-core" => multi_core = true,
            "-1" | "--single-core" => multi_core = false,
            "-b" | "--bench" => benchmark_mode = true,
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: -o requires an output filename");
                    std::process::exit(1);
                }
            }
            other => {
                if !other.starts_with('-') && input_path.is_none() {
                    input_path = Some(other.to_string());
                } else {
                    eprintln!("Unknown option: {}", other);
                    print_usage();
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }

    // Benchmark mode
    if benchmark_mode {
        let path = match input_path {
            Some(ref p) => p,
            None => {
                eprintln!("Error: Benchmark requires an input file");
                std::process::exit(1);
            }
        };
        run_benchmark(path);
        return Ok(());
    }

    // Read input data
    let input_data = match input_path {
        Some(ref p) if p != "-" => std::fs::read(p)?,
        _ => {
            let mut data = Vec::new();
            io::stdin().read_to_end(&mut data)?;
            data
        }
    };

    // Determine compress vs decompress
    let should_compress = mode_compress.unwrap_or_else(|| {
        if let Some(ref p) = input_path {
            !p.ends_with(".alk")
        } else {
            true
        }
    });

    let output_data = if should_compress {
        if multi_core && input_data.len() > simd_stream_codec::format::MAX_BLOCK_SIZE {
            simd_stream_codec::compress_parallel(&input_data)
        } else {
            simd_stream_codec::compress(&input_data)
        }
    } else {
        let result = if multi_core && input_data.len() > simd_stream_codec::format::MAX_BLOCK_SIZE {
            simd_stream_codec::decompress_parallel(&input_data)
        } else {
            simd_stream_codec::decompress(&input_data)
        };

        match result {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Decompression failed: {:?}", e);
                std::process::exit(1);
            }
        }
    };

    // Determine output destination
    match output_path {
        Some(ref p) if p != "-" => {
            std::fs::write(p, &output_data)?;
        }
        _ => {
            io::stdout().write_all(&output_data)?;
            io::stdout().flush()?;
        }
    }

    Ok(())
}

fn run_benchmark(path_str: &str) {
    let path = Path::new(path_str);
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read {}: {}", path_str, e);
            return;
        }
    };

    let len = data.len();
    println!("Benchmarking: {} ({:.2} MB)", path_str, len as f64 / (1024.0 * 1024.0));

    // Sequential compression
    let start = Instant::now();
    let comp_seq = simd_stream_codec::compress(&data);
    let seq_comp_sec = start.elapsed().as_secs_f64();
    let seq_comp_gb = (len as f64 / (1024.0 * 1024.0 * 1024.0)) / seq_comp_sec;

    // Parallel compression
    let start = Instant::now();
    let comp_par = simd_stream_codec::compress_parallel(&data);
    let par_comp_sec = start.elapsed().as_secs_f64();
    let par_comp_gb = (len as f64 / (1024.0 * 1024.0 * 1024.0)) / par_comp_sec;

    let ratio = len as f64 / comp_seq.len() as f64;
    println!("  Compression ratio: {:.2}x ({:.2} MB -> {:.2} MB)",
             ratio, len as f64 / (1024.0 * 1024.0), comp_seq.len() as f64 / (1024.0 * 1024.0));
    println!("  1-Core Compression:   {:>7.2} GB/s", seq_comp_gb);
    println!("  Multi-Core Compress:  {:>7.2} GB/s", par_comp_gb);

    // Decompression benchmark
    let iters = (300 * 1024 * 1024 / len).max(5);
    let mut dst = vec![0u8; len + 128];

    let start = Instant::now();
    for _ in 0..iters {
        simd_stream_codec::decompress_into_raw(&comp_seq, &mut dst).unwrap();
    }
    let raw_1c_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        simd_stream_codec::decompress_into(&comp_seq, &mut dst).unwrap();
    }
    let ver_1c_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        simd_stream_codec::decompress_parallel_into_raw(&comp_par, &mut dst).unwrap();
    }
    let raw_par_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        simd_stream_codec::decompress_parallel_into(&comp_par, &mut dst).unwrap();
    }
    let ver_par_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    println!("  1-Core Decompress (Raw):  {:>7.2} GB/s", raw_1c_gb);
    println!("  1-Core Decompress (Ver):  {:>7.2} GB/s", ver_1c_gb);
    println!("  Multi-Core Decomp (Raw):  {:>7.2} GB/s", raw_par_gb);
    println!("  Multi-Core Decomp (Ver):  {:>7.2} GB/s", ver_par_gb);
}
