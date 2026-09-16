use simd_stream_codec::{compress, decompress, fallback};

#[test]
fn test_fallback_parity() {
    let input = b"The quick brown fox jumps over the lazy dog. 1234567890! Repeating pattern: ABCDEFABCDEFABCDEF";
    let mut tokens = Vec::new();
    let mut offsets = Vec::new();
    let mut literals = Vec::new();

    fallback::compress_fallback(input, &mut tokens, &mut offsets, &mut literals);

    let mut decomp_buf = vec![0u8; input.len()];
    let written = fallback::decompress_fallback(&tokens, &offsets, &literals, &mut decomp_buf, 0, input.len()).unwrap();
    assert_eq!(written, input.len());
    assert_eq!(&decomp_buf[..written], input);
}

#[test]
fn test_corruption_truncation_safety() {
    let input = b"Structured payload testing memory safety under extreme stream truncations. \
                  {\"id\": 101, \"status\": \"ACTIVE\", \"tokens\": [1,2,3,4,5,6,7,8,9]}";
    let compressed = compress(input);

    // Truncate at every possible non-empty length
    for len in 1..compressed.len() {
        let truncated = &compressed[..len];
        let result = decompress(truncated);
        assert!(result.is_err(), "Truncated stream of len {} should error cleanly", len);
    }
}

#[test]
fn test_corruption_bitflip_safety() {
    let input = b"Structured payload testing bitflip resilience and memory safety. \
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.";
    let original_compressed = compress(input);

    // Systematically flip bits across the bitstream
    let mut state = 0xDEADBEEFu64;
    for _ in 0..1000 {
        let mut corrupted = original_compressed.clone();
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let byte_idx = (state as usize) % corrupted.len();
        let bit_mask = 1 << ((state >> 8) % 8);

        corrupted[byte_idx] ^= bit_mask;

        // Decompression must NEVER panic or segfault
        let _ = decompress(&corrupted);
    }
}
