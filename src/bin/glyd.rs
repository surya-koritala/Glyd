use std::env;
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::Instant;

fn print_usage() {
    eprintln!(r#"Glyd: compression for cloud storage
Usage: glyd [OPTIONS] [INPUT] [-o OUTPUT]
       (the store, which compresses across objects, is the glyd-store command)

Options:
    -c, --compress         Compress input (default if output is .glyd)
    -1, --fast             Fast level: LZ4-class compression speed, ratio ~2.10
    -t, --turbo            Turbo level: fastest decode, ~6% less ratio
    -9, --max              Max level: entropy coded, ratio above zstd -3
    -19, --ultra           Ultra level: optimal parse, ratio above zstd -16; slow to compress
    -C, --cold             Cold level: context mixing, the smallest output, 1-2 MB/s per core
                           each way; for what is stored for years and read rarely
    -r, --records          Record mode: logs and table dumps as typed columns before the level
    -B, --base <FILE>      Base mode: compress a new version against this old one (--max, or
                           --ultra); decoding needs the same file
    -S, --shape <FILE>     Shape dictionary: compress or decompress a small object (an event, a
                           small log or CSV) with a dictionary trained on a sample of such data
        --shape-train      Train a shape dictionary on the input (a few MB) and write it to -o
    -P, --pack             Pack the input files (many small objects) into one stream with an
                           index, record mode where it pays; any one object is read back alone
    -U, --unpack <DIR>     Write a pack's objects into DIR as 000000, 000001, ...
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
    glyd --base dump-monday.sql dump-tuesday.sql -o tuesday.glyd
    glyd -d --base dump-monday.sql tuesday.glyd -o dump-tuesday.sql
    glyd --pack events/*.json -o events.glyd
    glyd --unpack out/ events.glyd
    glyd --shape-train sample.log -o events.shape
    glyd --shape events.shape event.log -o event.glyd
    glyd -d --shape events.shape event.glyd -o event.log
    cat large.json | glyd -c > large.json.glyd
    cat large.json.glyd | glyd -d > large.json
    glyd -b dataset.bin
"#);
}

fn print_version() {
    println!("glyd {}", env!("CARGO_PKG_VERSION"));
    // The kernels are AVX2 (x86-64) and NEON (aarch64); everything else runs scalar.
    #[cfg(target_arch = "x86_64")]
    println!("SIMD: {}", if is_x86_feature_detected!("avx2") { "AVX2" } else { "scalar" });
    #[cfg(target_arch = "aarch64")]
    println!("SIMD: NEON");
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    println!("SIMD: scalar");
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
    let mut cold = false;
    let mut records = false;
    let mut base_path: Option<String> = None;
    let mut shape_path: Option<String> = None;
    let mut shape_train = false;
    let mut pack = false;
    let mut unpack_dir: Option<String> = None;
    let mut inputs: Vec<String> = Vec::new();

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
            "-C" | "--cold" => cold = true,
            "-r" | "--records" => records = true,
            "-B" | "--base" => {
                if i + 1 < args.len() {
                    base_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --base requires a file");
                    std::process::exit(1);
                }
            }
            "-S" | "--shape" => {
                if i + 1 < args.len() {
                    shape_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --shape requires a file");
                    std::process::exit(1);
                }
            }
            "--shape-train" => shape_train = true,
            "-P" | "--pack" => pack = true,
            "-U" | "--unpack" => {
                if i + 1 < args.len() {
                    unpack_dir = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --unpack requires a directory");
                    std::process::exit(1);
                }
            }
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
                    inputs.push(other.to_string());
                } else if !other.starts_with('-') && pack {
                    inputs.push(other.to_string());
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

    if pack {
        let files: Vec<Vec<u8>> = inputs.iter().map(std::fs::read).collect::<io::Result<_>>()?;
        let refs: Vec<&[u8]> = files.iter().map(|v| v.as_slice()).collect();
        let level: fn(&[u8], &mut Vec<u8>) = if cold { glyd::compress_into_cold } else if ultra { glyd::compress_into_ultra } else { glyd::compress_into_max };
        let mut out = Vec::new();
        glyd::compress_pack(&refs, &mut out, level);
        return write_out(&output_path, &out);
    }
    if let Some(dir) = unpack_dir {
        let data = match input_path {
            Some(ref p) => std::fs::read(p)?,
            None => {
                eprintln!("Error: --unpack needs the pack file");
                std::process::exit(1);
            }
        };
        let objects = match glyd::decompress_pack(&data) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Decompression failed: {}", e);
                std::process::exit(1);
            }
        };
        std::fs::create_dir_all(&dir)?;
        for (i, o) in objects.iter().enumerate() {
            std::fs::write(Path::new(&dir).join(format!("{:06}", i)), o)?;
        }
        return Ok(());
    }

    // Read input data
    // A file is mapped, not copied: the read then costs page faults
    // spread over the compressing threads instead of a pass before.
    let mapped;
    let owned;
    let input_data: &[u8] = match input_path {
        Some(ref p) if p != "-" => {
            mapped = glyd::mmap::Mapping::read_only(Path::new(p))?;
            mapped.bytes()
        }
        _ => {
            let mut data = Vec::new();
            io::stdin().read_to_end(&mut data)?;
            owned = data;
            &owned
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

    let base = match base_path {
        Some(ref p) => Some(std::fs::read(p)?),
        None => None,
    };
    if shape_train {
        let dict = match glyd::ShapeDict::train(&input_data) {
            Some(d) => d,
            None => {
                eprintln!("Error: the input is not record-shaped (delimited lines, JSON lines or a log)");
                std::process::exit(1);
            }
        };
        return write_out(&output_path, &dict.to_bytes());
    }
    if let Some(ref p) = shape_path {
        let dict = match glyd::ShapeDict::from_bytes(&std::fs::read(p)?) {
            Some(d) => d,
            None => {
                eprintln!("Error: {} is not a shape dictionary", p);
                std::process::exit(1);
            }
        };
        let out = if should_compress {
            let mut out = Vec::new();
            dict.compress(&input_data, &mut out);
            out
        } else {
            match dict.decompress(&input_data) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Decompression failed: {}", e);
                    std::process::exit(1);
                }
            }
        };
        return write_out(&output_path, &out);
    }
    let output_data = if should_compress {
        let mut out = Vec::with_capacity(input_data.len() / 2 + 1024);
        if let Some(ref base) = base {
            glyd::compress_with_base(base, &input_data, &mut out, ultra);
            match output_path {
                Some(ref p) if p != "-" => std::fs::write(p, &out)?,
                _ => {
                    io::stdout().write_all(&out)?;
                    io::stdout().flush()?;
                }
            }
            return Ok(());
        }
        let mc = multi_core && input_data.len() > glyd::format::MAX_BLOCK_SIZE;
        let level: fn(&[u8], &mut Vec<u8>) = match (mc, fast, turbo, max) {
            (true, _, _, _) if cold => glyd::compress_parallel_into_cold,
            (false, _, _, _) if cold => glyd::compress_into_cold,
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
        let stream_level: Option<(fn(&[u8], &mut Vec<u8>), usize)> = match (mc, records, cold, ultra, max, fast, turbo) {
            (true, false, false, false, true, _, _) => Some((glyd::compress_into_max, glyd::format::PARALLEL_UNIT_MAX)),
            (true, false, false, true, _, _, _) => Some((glyd::compress_into_ultra, glyd::format::PARALLEL_UNIT_ULTRA)),
            _ => None,
        };
        if let Some((unit_level, smallest)) = stream_level {
            // Units written as they finish: the output never sits whole
            // in memory, and the write overlaps the compressing.
            let mut sink: Box<dyn Write + Send> = match output_path {
                Some(ref p) if p != "-" => Box::new(std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(p)?)),
                _ => Box::new(io::stdout()),
            };
            // A gzip object: opened, its envelope first, its plain text streamed.
            let opened = if glyd::gz::is_gzip(input_data) { glyd::gz::open(input_data) } else { None };
            let data: &[u8] = match &opened {
                Some(o) => {
                    let mut head = Vec::new();
                    glyd::gz::envelope(input_data.len(), &o.recipe, &mut head);
                    sink.write_all(&head)?;
                    &o.plain
                }
                None => input_data,
            };
            glyd::compress_stream(data, unit_level, smallest, |unit| sink.write_all(unit))?;
            sink.flush()?;
            return Ok(());
        }
        if records && cold {
            glyd::compress_records_into_cold(&input_data, &mut out);
        } else if records {
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
        let result = if glyd::needs_base(&input_data) {
            match base {
                Some(ref base) => glyd::decompress_stream_with_base(base, &input_data, |batch| out.write_all(batch)),
                None => Err(io::Error::new(io::ErrorKind::InvalidData, "this file was compressed against a base: pass it with --base")),
            }
        } else if multi_core && input_data.len() > glyd::format::MAX_BLOCK_SIZE {
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

fn write_out(output_path: &Option<String>, data: &[u8]) -> io::Result<()> {
    match output_path {
        Some(p) if p != "-" => std::fs::write(p, data),
        _ => {
            io::stdout().write_all(data)?;
            io::stdout().flush()
        }
    }
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
