//! Decoding into a `dst` of exactly the uncompressed length (no slack):
//! the v6 decoder's wild copies must never land past it.
use glyd::*;

type Decode = fn(&[u8], &mut [u8]) -> error::Result<usize>;

/// Decode into `input.len()` bytes in front of a 64-byte sentinel.
fn check(what: &str, compressed: &[u8], input: &[u8], decode: Decode) {
    let mut buf = vec![0xEEu8; input.len() + 64];
    let (dst, tail) = buf.split_at_mut(input.len());
    let n = decode(compressed, dst).unwrap();
    assert!(tail.iter().all(|&b| b == 0xEE), "{what}: wrote past dst");
    assert_eq!(n, input.len(), "{what}: length");
    assert!(dst == input, "{what}: output");
}

fn inputs() -> Vec<Vec<u8>> {
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut rnd = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let mut text = Vec::new();
    while text.len() < 600_000 {
        text.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        text.extend_from_slice(&rnd(3));
    }
    let periodic = rnd(300).iter().cycle().take(600_000).copied().collect();
    let runs = (0..600_000u32).map(|i| (i / 40) as u8).collect();
    vec![text, periodic, rnd(600_000), runs]
}

/// A v6 block built so the fast path's last store overshoots the block:
/// 32 literals, then 32 matches of 40 at offset 32 (one 32-token chunk,
/// copied with the fixed 3x32 tails: the last reaches 88 past the chunk),
/// then a 70-byte literal tail, which is past the old 64-byte guard.
fn crafted_v6_block() -> (Vec<u8>, Vec<u8>) {
    use format::*;
    let (mut tokens, mut offsets, mut extras) = (Vec::new(), Vec::new(), Vec::new());
    let mut literals: Vec<u8> = (0..32).collect();
    for i in 0..32 {
        let lit = if i == 0 { 32 } else { 0 };
        tokens.push(Token::from_codes(encode_lit(lit, &mut extras), encode_match(40, &mut extras, MIN_MATCH_LEN), 0).0);
        push_offset(&mut offsets, 32);
    }
    literals.extend((0..70).map(|i| 0x80 | i));
    tokens.push(Token::from_codes(encode_lit(70, &mut extras), 0, 0).0);

    let mut expected = literals[..32].to_vec();
    for _ in 0..32 * 40 {
        expected.push(expected[expected.len() - 32]);
    }
    expected.extend_from_slice(&literals[32..]);
    let header = BlockHeader {
        magic: MAGIC,
        version: CURRENT_VERSION,
        flags: FLAG_CHAIN_RESET,
        checksum: compute_checksum(&expected),
        uncompressed_len: expected.len() as u32,
        token_count: tokens.len() as u32,
        token_bytes: tokens.len() as u32,
        offset_bytes: offsets.len() as u32,
        extras_bytes: extras.len() as u32,
        literal_len: literals.len() as u32,
    };
    let mut block = unsafe { std::slice::from_raw_parts(&header as *const BlockHeader as *const u8, HEADER_SIZE) }.to_vec();
    block.extend([tokens, offsets, extras, literals].concat());
    (block, expected)
}

#[test]
fn exact_dst_crafted_block() {
    let (block, expected) = crafted_v6_block();
    for (what, decode) in [
        ("decompress_into_raw", decompress_into_raw as Decode),
        ("decompress_into", decompress_into),
        ("decompress_parallel_into_raw", decompress_parallel_into_raw),
        ("decompress_parallel_into", decompress_parallel_into),
    ] {
        check(what, &block, &expected, decode);
    }
}

#[test]
fn exact_dst_sequential() {
    let levels: [(&str, fn(&[u8], &mut Vec<u8>)); 3] =
        [("default", compress_into), ("fast", compress_into_fast), ("turbo", compress_into_turbo)];
    for input in inputs() {
        for (name, level) in levels {
            let mut c = Vec::new();
            level(&input, &mut c);
            check(&format!("{name} decompress_into_raw"), &c, &input, decompress_into_raw);
            check(&format!("{name} decompress_into"), &c, &input, decompress_into);
        }
    }
}

/// Parallel-compressed (> 256 KB: several units), so every block decodes
/// into its own exact-size slice.
#[test]
fn exact_dst_parallel() {
    let levels: [(&str, fn(&[u8], &mut Vec<u8>)); 3] =
        [("default", compress_parallel_into), ("fast", compress_parallel_into_fast), ("turbo", compress_parallel_into_turbo)];
    for input in inputs() {
        for (name, level) in levels {
            let mut c = Vec::new();
            level(&input, &mut c);
            check(&format!("{name} decompress_parallel_into_raw"), &c, &input, decompress_parallel_into_raw);
            check(&format!("{name} decompress_parallel_into"), &c, &input, decompress_parallel_into);
        }
    }
}
