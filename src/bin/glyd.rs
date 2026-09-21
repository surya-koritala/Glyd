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
    -C, --cold             Cold level: context mixing, the smallest output, 1-2 MB/s per core
                           each way; for what is stored for years and read rarely
    -r, --records          Record mode: logs and table dumps as typed columns before the level
    -B, --base <FILE>      Base mode: compress a new version against this old one (--max, or
                           --ultra); decoding needs the same file
    -S, --shape <FILE>     Shape dictionary: compress or decompress a small object (an event, a
                           small log or CSV) with a dictionary trained on a sample of such data
        --shape-train      Train a shape dictionary on the input (a few MB) and write it to -o
        --store <DIR>      A store that compresses across its objects: with --put FILE..., each
                           file is stored as a delta against the stored object it most resembles
                           (found by fingerprints) when that pays, small files in packs; --get ID
                           writes an object to -o; --find NAME prints its id; --delete ID;
                           --compact frees what no live object needs; --verify reads every
                           object back; --rebase ID stores an object alone again (one decode to
                           read); --stats lists the objects and the bytes. --ultra or --cold
                           sets the level objects stored alone take. --s3 s3://bucket/prefix
                           keeps the objects in S3 through the AWS CLI (metadata stays in DIR).
        --audit <PATH>     What the store would save on a directory or an s3://bucket/prefix:
                           a sample of its objects (--sample N, default 200; spread over the
                           listing, in name order) goes through a store, against zstd -3 per
                           object when the zstd CLI is installed, and the result is scaled to
                           the whole listing with a yearly cost at S3 Standard's list price
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
    glyd --store bucket/ --put snapshot-mon.tar snapshot-tue.tar
    glyd --store bucket/ --get 1 -o snapshot-tue.tar
    glyd --store bucket/ --stats
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
    let mut cold = false;
    let mut records = false;
    let mut base_path: Option<String> = None;
    let mut shape_path: Option<String> = None;
    let mut shape_train = false;
    let mut pack = false;
    let mut unpack_dir: Option<String> = None;
    let mut inputs: Vec<String> = Vec::new();
    let mut store_dir: Option<String> = None;
    let mut store_put = false;
    let mut store_get: Option<u32> = None;
    let mut store_stats = false;
    let mut store_find: Option<String> = None;
    let mut store_delete: Option<u32> = None;
    let mut store_compact = false;
    let mut store_verify = false;
    let mut store_rebase: Option<u32> = None;
    let mut store_s3: Option<String> = None;
    let mut audit: Option<String> = None;
    let mut sample: usize = 200;

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
            "--store" => {
                if i + 1 < args.len() {
                    store_dir = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --store requires a directory");
                    std::process::exit(1);
                }
            }
            "--put" => store_put = true,
            "--get" => {
                match args.get(i + 1).and_then(|a| a.parse().ok()) {
                    Some(id) => {
                        store_get = Some(id);
                        i += 1;
                    }
                    None => {
                        eprintln!("Error: --get requires an object id");
                        std::process::exit(1);
                    }
                }
            }
            "--stats" => store_stats = true,
            "--find" => {
                if i + 1 < args.len() {
                    store_find = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --find requires a name");
                    std::process::exit(1);
                }
            }
            "--delete" => {
                match args.get(i + 1).and_then(|a| a.parse().ok()) {
                    Some(id) => {
                        store_delete = Some(id);
                        i += 1;
                    }
                    None => {
                        eprintln!("Error: --delete requires an object id");
                        std::process::exit(1);
                    }
                }
            }
            "--compact" => store_compact = true,
            "--rebase" => {
                match args.get(i + 1).and_then(|a| a.parse().ok()) {
                    Some(id) => {
                        store_rebase = Some(id);
                        i += 1;
                    }
                    None => {
                        eprintln!("Error: --rebase requires an object id");
                        std::process::exit(1);
                    }
                }
            }
            "--s3" => {
                if i + 1 < args.len() {
                    store_s3 = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --s3 requires an s3://bucket/prefix url");
                    std::process::exit(1);
                }
            }
            "--verify" => store_verify = true,
            "--audit" => {
                if i + 1 < args.len() {
                    audit = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --audit requires a directory or an s3:// url");
                    std::process::exit(1);
                }
            }
            "--sample" => {
                match args.get(i + 1).and_then(|a| a.parse().ok()) {
                    Some(n) => {
                        sample = n;
                        i += 1;
                    }
                    None => {
                        eprintln!("Error: --sample requires a count");
                        std::process::exit(1);
                    }
                }
            }
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
                } else if !other.starts_with('-') && (pack || store_put) {
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

    if let Some(target) = audit {
        return run_audit(&target, sample);
    }
    if let Some(dir) = store_dir {
        let mut store = match store_s3 {
            Some(url) => glyd::Store::open_with(&dir, Box::new(glyd::store::S3Cli::new(&url)?))?,
            None => glyd::Store::open(&dir)?,
        };
        if cold {
            store.set_level(glyd::store::Level::Cold);
        } else if ultra {
            store.set_level(glyd::store::Level::Ultra);
        }
        if store_put {
            for f in &inputs {
                let data = std::fs::read(f)?;
                let t = Instant::now();
                let id = store.put(f, &data)?;
                let e = &store.entries()[id as usize];
                eprintln!("{:>6}  {:>10} -> {:>10} B  {}  {:.0} MB/s  {}", id, e.raw_len, e.stored_len, e.base.map_or("alone".to_string(), |b| format!("delta against {} (depth {})", b, e.depth)), data.len() as f64 / t.elapsed().as_secs_f64() / 1e6, f);
            }
        }
        if let Some(id) = store_get {
            let data = store.get(id)?;
            write_out(&output_path, &data)?;
        }
        if let Some(name) = store_find {
            match store.id_of(&name) {
                Some(id) => println!("{id}"),
                None => {
                    eprintln!("not in the store: {name}");
                    std::process::exit(1);
                }
            }
        }
        if let Some(id) = store_delete {
            store.delete(id)?;
        }
        if let Some(id) = store_rebase {
            store.rebase(id)?;
        }
        if store_compact {
            let freed = store.compact()?;
            eprintln!("compacted: {freed} B freed");
        }
        if store_verify {
            let (ok, failed) = store.verify();
            for (id, e) in &failed {
                eprintln!("object {id}: {e}");
            }
            eprintln!("verified: {ok} objects ok, {} failed", failed.len());
            if !failed.is_empty() {
                std::process::exit(1);
            }
        }
        if store_stats {
            for e in store.entries() {
                let how = if e.deleted { "deleted".to_string() } else if let Some((p, i)) = e.pack { format!("in pack {p} at {i}") } else { e.base.map_or("alone".to_string(), |b| format!("delta against {} (depth {})", b, e.depth)) };
                println!("{:>6}  {:>12}  {:>12}  {:<28} {}", e.id, e.raw_len, e.stored_len, how, e.name);
            }
            let (raw, stored) = store.stats();
            let live = store.entries().iter().filter(|e| !e.deleted).count();
            println!("{} objects ({} live): {} B raw, {} B on disk ({:.2}x)", store.entries().len(), live, raw, stored, raw as f64 / stored.max(1) as f64);
        }
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
            mapped = glyd::store::Mapping::read_only(Path::new(p))?;
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
            glyd::compress_stream(input_data, unit_level, smallest, |unit| sink.write_all(unit))?;
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

/// The objects under `target` (a directory, or s3://bucket/prefix through
/// the AWS CLI): (name, bytes), in name order.
fn list_objects(target: &str) -> io::Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    if let Some(rest) = target.strip_prefix("s3://") {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let text = std::process::Command::new("aws").args(["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix, "--query", "Contents[].[Key,Size]", "--output", "text"]).output()?;
        if !text.status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, format!("aws: {}", String::from_utf8_lossy(&text.stderr).trim())));
        }
        for line in String::from_utf8_lossy(&text.stdout).lines() {
            if let Some((key, size)) = line.rsplit_once('\t') {
                if let Ok(n) = size.trim().parse::<u64>() {
                    if n > 0 {
                        out.push((format!("s3://{bucket}/{key}"), n));
                    }
                }
            }
        }
    } else {
        fn walk(dir: &Path, out: &mut Vec<(String, u64)>) -> io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out)?;
                } else if let Ok(m) = entry.metadata() {
                    if m.len() > 0 {
                        out.push((path.to_string_lossy().to_string(), m.len()));
                    }
                }
            }
            Ok(())
        }
        walk(Path::new(target), &mut out)?;
    }
    out.sort();
    Ok(out)
}

fn read_object(name: &str) -> io::Result<Vec<u8>> {
    if name.starts_with("s3://") {
        let out = std::process::Command::new("aws").args(["s3", "cp", "--quiet", name, "-"]).output()?;
        if !out.status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, format!("aws: {}", String::from_utf8_lossy(&out.stderr).trim())));
        }
        Ok(out.stdout)
    } else {
        std::fs::read(name)
    }
}

/// `zstd -3` bytes of `data` through the zstd CLI, when it is installed.
fn zstd3_len(data: &[u8]) -> Option<usize> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("zstd").args(["-3", "-q", "-c", "-T0"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut stdin = child.stdin.take()?;
    let data = data.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&data));
    let out = child.wait_with_output().ok()?;
    writer.join().ok()?.ok()?;
    if out.status.success() { Some(out.stdout.len()) } else { None }
}

/// The store on a sample of the objects under `target`, scaled to all
/// of them, priced at S3 Standard's list rate.
fn run_audit(target: &str, sample: usize) -> io::Result<()> {
    const PRICE_TB_YEAR: f64 = 0.023 * 12.0 * 1000.0;
    let all = list_objects(target)?;
    if all.is_empty() {
        eprintln!("nothing under {target}");
        std::process::exit(1);
    }
    let total: u64 = all.iter().map(|(_, n)| n).sum();
    // The sample: runs of consecutive objects (versions of a thing sit
    // together by name) at eight places spread over the listing.
    let n = sample.min(all.len());
    let picked: Vec<&(String, u64)> = if n >= all.len() {
        all.iter().collect()
    } else {
        let runs = 8.min(n);
        let per = n / runs;
        let mut v = Vec::with_capacity(n);
        for r in 0..runs {
            let start = r * (all.len() - per) / (runs - 1).max(1);
            v.extend(all[start..start + per].iter());
        }
        v
    };
    let dir = std::env::temp_dir().join(format!("glyd-audit-{}", std::process::id()));
    let mut store = glyd::Store::open(&dir)?;
    let zstd = zstd3_len(b"probe").is_some();
    let (mut raw, mut alone_z, mut alone_g) = (0u64, 0u64, 0u64);
    eprintln!("{target}: {} objects, {:.1} GB; sampling {} of them{}", all.len(), total as f64 / 1e9, picked.len(), if zstd { " against zstd -3" } else { " (no zstd CLI: Glyd --max alone stands in for today's codec)" });
    let t = Instant::now();
    for (i, (name, _)) in picked.iter().enumerate() {
        let data = match read_object(name) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping {name}: {e}");
                continue;
            }
        };
        raw += data.len() as u64;
        if zstd {
            alone_z += zstd3_len(&data).unwrap_or(data.len()) as u64;
        }
        let mut c = Vec::new();
        glyd::compress_parallel_into_max(&data, &mut c);
        alone_g += c.len() as u64;
        store.put(name, &data)?;
        if (i + 1) % 20 == 0 {
            eprintln!("  {} of {} ({:.0} MB/s)", i + 1, picked.len(), raw as f64 / t.elapsed().as_secs_f64() / 1e6);
        }
    }
    store.flush()?;
    let (_, stored) = store.stats();
    let deltas = store.entries().iter().filter(|e| e.base.is_some()).count();
    let packs = store.entries().iter().filter(|e| e.name.starts_with("pack of ")).count();
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
    let today = if zstd { alone_z } else { alone_g };
    let scale = total as f64 / raw.max(1) as f64;
    let tb = |b: u64| b as f64 * scale / 1e12;
    let size = |t: f64| if t >= 1.0 { format!("{t:.2} TB") } else { format!("{:.1} GB", t * 1000.0) };
    let money = |d: f64| if d >= 100.0 { format!("${d:.0}") } else { format!("${d:.2}") };
    println!("{target}");
    println!("  {} objects, {} raw; sample of {} objects, {}", all.len(), size(total as f64 / 1e12), picked.len(), size(raw as f64 / 1e12));
    println!("  {:<34} {:>10} stored  {:>10} a year", if zstd { "zstd -3, each object alone:" } else { "Glyd --max, each object alone:" }, size(tb(today)), money(tb(today) * PRICE_TB_YEAR));
    if zstd {
        println!("  {:<34} {:>10} stored  {:>10} a year", "Glyd --max, each object alone:", size(tb(alone_g)), money(tb(alone_g) * PRICE_TB_YEAR));
    }
    println!("  {:<34} {:>10} stored  {:>10} a year", "Glyd store (across the objects):", size(tb(stored)), money(tb(stored) * PRICE_TB_YEAR));
    println!("  {:.2}x fewer bytes than {}; {} of {} sampled objects stored as deltas, {} packs of small ones", today as f64 / stored.max(1) as f64, if zstd { "zstd -3" } else { "Glyd alone" }, deltas, picked.len(), packs);
    println!("  saving about {} a year at S3 Standard's list price ($23/TB-month), scaled from the sample", money((tb(today) - tb(stored)) * PRICE_TB_YEAR));
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
