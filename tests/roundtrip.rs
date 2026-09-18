use simd_stream_codec::{compress, decompress};

#[test]
fn test_roundtrip_empty() {
    let input = b"";
    let compressed = compress(input);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_roundtrip_short() {
    let input = b"Hello, World!";
    let compressed = compress(input);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_roundtrip_rle_offset_1() {
    let input = vec![b'A'; 2048];
    let compressed = compress(&input);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_roundtrip_repeating_patterns() {
    for period in 1..=20 {
        let pattern: Vec<u8> = (0..period).map(|i| b'a' + (i as u8 % 26)).collect();
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(&pattern);
        }
        let compressed = compress(&input);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "Failed on period {}", period);
    }
}

#[test]
fn test_roundtrip_multiblock() {
    // 200 KB spanning multiple 64 KB blocks
    let base = b"{\"user_id\": 12345, \"name\": \"Antigravity\", \"status\": \"active\", \"score\": 99.8}, \n";
    let mut input = Vec::new();
    while input.len() < 200_000 {
        input.extend_from_slice(base);
    }
    let compressed = compress(&input);
    println!("Multiblock 200KB: compressed to {} bytes (ratio: {:.2}x)", 
             compressed.len(), input.len() as f64 / compressed.len() as f64);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);

    // Parallel multi-core test
    let comp_par = simd_stream_codec::compress_parallel(&input);
    let decomp_par = simd_stream_codec::decompress_parallel(&comp_par).unwrap();
    assert_eq!(decomp_par, input);
}

#[test]
fn test_roundtrip_random() {
    // Pseudo-random bytes with periodic repeats
    let mut input = Vec::with_capacity(65536);
    let mut state = 0x12345678u64;
    for _ in 0..65536 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        input.push((state >> 33) as u8);
    }
    let compressed = compress(&input);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_roundtrip_100mb_scale() {
    // 100 MB stream spanning ~1,600 blocks
    let base = b"EventRecord(id=987654321, metric=123.456, tag='telemetry-prod', status='SUCCESS')\n";
    let mut input = Vec::with_capacity(100 * 1024 * 1024);
    while input.len() < 100 * 1024 * 1024 {
        input.extend_from_slice(base);
    }

    // 1. Sequential Chained
    let compressed = compress(&input);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed.len(), input.len());
    assert_eq!(decompressed, input);

    // 2. Parallel Independent Blocks
    let comp_par = simd_stream_codec::compress_parallel(&input);
    let decomp_par = simd_stream_codec::decompress_parallel(&comp_par).unwrap();
    assert_eq!(decomp_par.len(), input.len());
    assert_eq!(decomp_par, input);
}

#[test]
fn test_roundtrip_all_silesia_corpus() {
    let corpus_dir = std::path::Path::new("corpus");
    if !corpus_dir.exists() {
        return;
    }
    let files = [
        "dickens", "mozilla", "mr", "nci", "ooffice", "osdb",
        "reymont", "samba", "sao", "webster", "xml", "x-ray",
    ];
    for file_name in &files {
        let p = corpus_dir.join(file_name);
        if !p.exists() { continue; }
        let data = std::fs::read(&p).unwrap();
        println!("Testing corpus file: {}", file_name);

        // Sequential test
        let comp = compress(&data);
        let decomp = decompress(&comp).unwrap();
        assert_eq!(decomp, data, "Sequential roundtrip mismatch on {}", file_name);

        // Parallel test
        let comp_p = simd_stream_codec::compress_parallel(&data);
        let decomp_p = simd_stream_codec::decompress_parallel(&comp_p).unwrap();
        assert_eq!(decomp_p, data, "Parallel roundtrip mismatch on {}", file_name);
    }
}

#[test]
fn test_roundtrip_extended_literals_10kb() {
    // Generate 10 KB with low repeat probability so extended literal runs are created
    let mut input = Vec::with_capacity(10_000);
    let mut state = 0x87654321u64;
    for _ in 0..10_000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        input.push((state >> 32) as u8);
    }

    let compressed = compress(&input);
    let decompressed = decompress(&compressed).expect("Decompression of 10KB random failed");
    assert_eq!(decompressed, input);

    let comp_p = simd_stream_codec::compress_parallel(&input);
    let decomp_p = simd_stream_codec::decompress_parallel(&comp_p).expect("Parallel decompression failed");
    assert_eq!(decomp_p, input);
}

#[test]
fn test_roundtrip_extended_literals_max_block() {
    // Exactly 65,535 bytes
    let mut input = Vec::with_capacity(65535);
    let mut state = 0xCAFEBABE12345678u64;
    for _ in 0..65535 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        input.push((state >> 32) as u8);
    }

    let compressed = compress(&input);
    let decompressed = decompress(&compressed).expect("Decompression of max-block random failed");
    assert_eq!(decompressed, input);

    let comp_p = simd_stream_codec::compress_parallel(&input);
    let decomp_p = simd_stream_codec::decompress_parallel(&comp_p).expect("Parallel decompression failed");
    assert_eq!(decomp_p, input);
}

#[test]
fn test_roundtrip_mixed_literal_runs() {
    let mut input = Vec::new();
    let mut state = 0xDEADBEEF00112233u64;
    let run_sizes = [5, 16, 31, 32, 33, 64, 100, 256, 1024, 4096];

    for &size in &run_sizes {
        // High entropy literal run of exact size
        for _ in 0..size {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            input.push((state >> 32) as u8);
        }
        // Repetitive match sequence to trigger match encoding
        let pattern = b"0123456789ABCDEF";
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
    }

    let compressed = compress(&input);
    let decompressed = decompress(&compressed).expect("Decompression of mixed runs failed");
    assert_eq!(decompressed, input);

    let comp_p = simd_stream_codec::compress_parallel(&input);
    let decomp_p = simd_stream_codec::decompress_parallel(&comp_p).expect("Parallel decompression failed");
    assert_eq!(decomp_p, input);
}

#[test]
fn test_parallel_pipeline_256k_chunking_and_chaining() {
    // 1 MB payload spanning multiple 256 KB parallel chunks and multiple 64 KB blocks per chunk
    let base = b"EventRecord(user='antigravity-ai', cluster='us-east-1', status='PROD_ACTIVE', ts=1720000000)\n";
    let mut input = Vec::with_capacity(1024 * 1024);
    while input.len() < 1024 * 1024 {
        input.extend_from_slice(base);
    }

    // 1. Parallel compress -> Sequential decompress
    let comp_parallel = simd_stream_codec::compress_parallel(&input);
    let decomp_seq = simd_stream_codec::decompress(&comp_parallel).expect("Sequential decompress of parallel stream failed");
    assert_eq!(decomp_seq, input);

    // 2. Parallel compress -> Parallel decompress
    let decomp_par = simd_stream_codec::decompress_parallel(&comp_parallel).expect("Parallel decompress of parallel stream failed");
    assert_eq!(decomp_par, input);

    // 3. Sequential compress (cross-block chained across 1MB) -> Parallel decompress (must safely fall back to sequential)
    let comp_seq = simd_stream_codec::compress(&input);
    let decomp_par_fallback = simd_stream_codec::decompress_parallel(&comp_seq).expect("Parallel decompress of sequential chained stream failed");
    assert_eq!(decomp_par_fallback, input);
}




/// A literal run at the format's maximum (MAX_LIT_LEN = 65797) inside a
/// compressed block. The chunked AVX2 decoder keeps lengths in u16 arrays and
/// once wrapped this to 261; the parallel Silesia stream happened to contain
/// one, the sequential one did not. The block needs a compressible prefix so
/// it is not stored raw, then an incompressible run longer than the maximum.
#[test]
fn test_roundtrip_literal_run_at_max_len() {
    use simd_stream_codec::format::MAX_LIT_LEN;
    let mut input = vec![b'A'; 100 * 1024];
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..(MAX_LIT_LEN + 5000) {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        input.push((st >> 40) as u8);
    }
    for f in [simd_stream_codec::compress, simd_stream_codec::compress_parallel] {
        let c = f(&input);
        assert_eq!(simd_stream_codec::decompress(&c).unwrap(), input);
        assert_eq!(simd_stream_codec::decompress_parallel(&c).unwrap(), input);
    }
}

/// Fast level: every shape the default tests cover, sequential and parallel.
#[test]
fn test_roundtrip_fast_level() {
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut rnd = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let mut inputs: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"Hello, World!".to_vec(),
        vec![b'A'; 2048],
        rnd(300_000),                      // incompressible: raw blocks
        (0..2_000_000u32).map(|i| (i / 7) as u8).collect(), // long runs, long matches
    ];
    for period in [1usize, 3, 7, 13, 64, 300, 5000, 70_000] {
        let pat = rnd(period);
        inputs.push(pat.iter().cycle().take(700_000).copied().collect());
    }
    let mut text = Vec::new();
    while text.len() < 1_500_000 { text.extend_from_slice(b"the quick brown fox jumps over the lazy dog "); text.extend_from_slice(&rnd(3)); }
    inputs.push(text);
    for input in &inputs {
        let mut c = Vec::new();
        simd_stream_codec::compress_into_fast(input, &mut c);
        assert_eq!(&decompress(&c).unwrap(), input, "fast sequential, len {}", input.len());
        let mut p = Vec::new();
        simd_stream_codec::compress_parallel_into_fast(input, &mut p);
        assert_eq!(&simd_stream_codec::decompress_parallel(&p).unwrap(), input, "fast parallel, len {}", input.len());
    }
}
