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
