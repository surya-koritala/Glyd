use std::env;
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::Instant;

fn print_usage() {
    eprintln!(r#"Glyd High-Performance SIMD Stream Codec (AVX-512 / AVX2 / GPU)
Usage: glyd [OPTIONS] [INPUT] [-o OUTPUT]

Options:
    -c, --compress         Compress input (default if output is .glyd)
    -1, --fast             Fast level: LZ4-class compression speed, ratio ~2.10
    -t, --turbo            Turbo level: fastest decode, ~6% less ratio
    -9, --max              Max level: entropy coded, ratio above zstd -3
    -19, --ultra           Ultra level: optimal parse, ratio above zstd -16; slow to compress
    -r, --records          Record mode: logs and table dumps as typed columns before the level
    -d, --decompress       Decompress input (default if input is .glyd)
    -m, --multi-core       Use multi-core parallel engine (default)
    -s, --single-core      Force single-core sequential engine
    -o, --output <FILE>    Specify destination output file (defaults to stdout if piped)
    -b, --bench            Benchmark compression and decompression throughput
    -v, --version          Print version and CPU SIMD hardware capabilities
    -h, --help             Print this help message

Examples:
    glyd input.tar -o input.tar.glyd
    glyd -d input.tar.glyd -o input.tar
    cat large.json | glyd -c > large.json.glyd
    cat large.json.glyd | glyd -d > large.json
    glyd -b dataset.bin
"#);
}

fn print_version() {
    println!("glyd 0.1.0");
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
    let mut fast = false;
    let mut turbo = false;
    let mut max = false;
    let mut ultra = false;
    let mut records = false;

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
            "-1" | "--fast" => fast = true,
            "-t" | "--turbo" => turbo = true,
            "-9" | "--max" => max = true,
            "-19" | "--ultra" => ultra = true,
            "-r" | "--records" => records = true,
            "-d" | "--decompress" => mode_compress = Some(false),
            "-m" | "--multi-core" => multi_core = true,
            "-s" | "--single-core" => multi_core = false,
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
            !p.ends_with(".glyd")
        } else {
            true
        }
    });

    let output_data = if should_compress {
        let mut out = Vec::with_capacity(input_data.len() / 2 + 1024);
        let mc = multi_core && input_data.len() > glyd::format::MAX_BLOCK_SIZE;
        let level: fn(&[u8], &mut Vec<u8>) = match (mc, fast, turbo, max) {
            (true, _, _, _) if ultra => glyd::compress_parallel_into_ultra,
            (false, _, _, _) if ultra => glyd::compress_into_ultra,
            (true, _, _, true) => glyd::compress_parallel_into_max,
            (false, _, _, true) => glyd::compress_into_max,
            (true, true, _, _) => glyd::compress_parallel_into_fast,
            (true, _, true, _) => glyd::compress_parallel_into_turbo,
            (true, _, _, _) => glyd::compress_parallel_into,
            (false, true, _, _) => glyd::compress_into_fast,
            (false, _, true, _) => glyd::compress_into_turbo,
            (false, _, _, _) => glyd::compress_into,
        };
        if records {
            // Record mode parallelises over its own units; the level
            // inside a unit is the sequential one.
            let unit_level: fn(&[u8], &mut Vec<u8>) = match (fast, turbo, max, ultra) {
                (_, _, _, true) => glyd::compress_into_ultra,
                (_, _, true, _) => glyd::compress_into_max,
                (true, _, _, _) => glyd::compress_into_fast,
                (_, true, _, _) => glyd::compress_into_turbo,
                _ => glyd::compress_into,
            };
            glyd::compress_records_with(&input_data, &mut out, unit_level);
        } else {
            level(&input_data, &mut out);
        }
        out
    } else {
        // Decoded a batch of units at a time into one reused buffer and
        // written as it goes: the memory is a batch, not the file.
        let mut out: Box<dyn Write> = match output_path {
            Some(ref p) if p != "-" => Box::new(std::fs::File::create(p)?),
            _ => Box::new(io::stdout().lock()),
        };
        let result = if multi_core && input_data.len() > glyd::format::MAX_BLOCK_SIZE {
            glyd::decompress_stream(&input_data, |batch| out.write_all(batch))
        } else {
            match glyd::decompress(&input_data) {
                Ok(data) => out.write_all(&data),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }
        };
        if let Err(e) = result {
            if e.kind() == io::ErrorKind::InvalidData {
                eprintln!("Decompression failed: {}", e);
                std::process::exit(1);
            }
            return Err(e);
        }
        out.flush()?;
        return Ok(());
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
    let comp_seq = glyd::compress(&data);
    let seq_comp_sec = start.elapsed().as_secs_f64();
    let seq_comp_gb = (len as f64 / (1024.0 * 1024.0 * 1024.0)) / seq_comp_sec;

    // Parallel compression
    let start = Instant::now();
    let comp_par = glyd::compress_parallel(&data);
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
        glyd::decompress_into_raw(&comp_seq, &mut dst).unwrap();
    }
    let raw_1c_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_into(&comp_seq, &mut dst).unwrap();
    }
    let ver_1c_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_parallel_into_raw(&comp_par, &mut dst).unwrap();
    }
    let raw_par_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_parallel_into(&comp_par, &mut dst).unwrap();
    }
    let ver_par_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    println!("  1-Core Decompress (Raw):  {:>7.2} GB/s", raw_1c_gb);
    println!("  1-Core Decompress (Ver):  {:>7.2} GB/s", ver_1c_gb);
    println!("  Multi-Core Decomp (Raw):  {:>7.2} GB/s", raw_par_gb);
    println!("  Multi-Core Decomp (Ver):  {:>7.2} GB/s", ver_par_gb);
}
