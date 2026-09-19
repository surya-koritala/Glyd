//! Corruption must never panic, hang, or read out of bounds; it must
//! return an error or, when the checksum still matches, the right data.
//!
//! `tests/fuzz_safety.rs` is the G1 gate (1M mutations, every level and
//! path, sentinels past `dst`); this is the cheaper v7-only gate: 200,000
//! mutations by default, `V7_FUZZ=1000000 cargo test --release --test
//! v7_fuzz` for the full run, whose count lands in `.v7-fuzz-status`.
use simd_stream_codec::{compress_into_max, decompress, decompress_into};

fn seed_inputs() -> Vec<Vec<u8>> {
    let mut x = 0xC0FFEEu64;
    let mut rnd = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let mut v = vec![b"hello hello hello world".to_vec(), vec![0u8; 5000], rnd(20_000)];
    let pat = rnd(37);
    v.push(pat.iter().cycle().take(300_000).copied().collect());
    let mut text = Vec::new();
    while text.len() < 400_000 { text.extend_from_slice(b"lorem ipsum dolor sit amet "); text.extend_from_slice(&rnd(3)); }
    v.push(text);
    v
}

#[test]
fn v7_mutation_fuzz() {
    let inputs = seed_inputs();
    let streams: Vec<Vec<u8>> = inputs.iter().map(|i| { let mut c = Vec::new(); compress_into_max(i, &mut c); c }).collect();
    let mut x = 0x5EEDu64;
    let mut mutations = 0u64;
    let target: u64 = std::env::var("V7_FUZZ").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let mut m = Vec::new();
    while mutations < target {
        for (input, c) in inputs.iter().zip(&streams) {
            // `dst` is exactly the input's size: every block's last 96
            // bytes take the exact copy path, which `decompress` (padded
            // output) never reaches. Nothing may be written past it.
            let mut dst = vec![0u8; input.len()];
            for _ in 0..50 {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                m.clear();
                m.extend_from_slice(c);
                match x % 4 {
                    0 => { let i = (x >> 8) as usize % m.len(); m[i] ^= 1 << ((x >> 40) & 7); }
                    1 => { let i = (x >> 8) as usize % m.len(); m[i] = (x >> 40) as u8; }
                    2 => { let n = (x >> 8) as usize % m.len(); m.truncate(n); }
                    _ => { let i = (x >> 8) as usize % m.len(); let j = (x >> 32) as usize % m.len(); m.swap(i, j); }
                }
                if let Ok(n) = decompress_into(&m, &mut dst) {
                    assert_eq!(&dst[..n], &input[..n], "checksum passed but data differs");
                }
                if let Ok(out) = decompress(&m) {
                    assert!(out.len() <= input.len() && out[..] == input[..out.len()], "checksum passed but data differs");
                }
                mutations += 1;
                if mutations >= target { break; }
            }
            if mutations >= target { break; }
        }
    }
    std::fs::write(".v7-fuzz-status", format!("{}", mutations)).unwrap();
}

/// The wild copy pass (NEON on aarch64, 32-byte scalar copies elsewhere)
/// and the exact scalar pass must produce identical bytes: a padded
/// output (`decompress`) copies nearly every sequence wild, an exact-size
/// one (`decompress_into`) routes each block's tail through the exact
/// pass, and both must equal the input.
#[test]
fn v7_scalar_and_simd_agree() {
    for input in seed_inputs() {
        let mut c = Vec::new();
        compress_into_max(&input, &mut c);
        assert_eq!(decompress(&c).unwrap(), input);
        let mut exact = vec![0u8; input.len()];
        assert_eq!(decompress_into(&c, &mut exact).unwrap(), input.len());
        assert_eq!(exact, input);
    }
}
