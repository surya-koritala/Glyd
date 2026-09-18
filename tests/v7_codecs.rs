use simd_stream_codec::bits::{BitReader, BitWriter, PAD};

#[test]
fn bits_roundtrip_mixed_widths() {
    let mut w = BitWriter::new();
    let mut expect = Vec::new();
    let mut x = 0x2545F4914F6CDD1Du64;
    for i in 0..10_000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let n = (i % 32) + 1; // 1..=32 bits
        let v = x & ((1u64 << n) - 1);
        w.put(v, n);
        expect.push((v, n));
    }
    let bytes = w.finish();
    assert_eq!(&bytes[bytes.len() - PAD..], &[0u8; PAD]);
    let mut r = BitReader::new(&bytes);
    for (v, n) in expect {
        assert_eq!(r.get(n), v);
    }
    assert!(!r.overrun());
}

#[test]
fn bits_reader_clamps_at_end() {
    let bytes = BitWriter::new().finish(); // 8 pad bytes only
    let mut r = BitReader::new(&bytes);
    for _ in 0..1000 { let _ = r.get(32); } // far past the end: must not fault
    assert!(r.overrun());
}

use simd_stream_codec::huff8;

fn skewed_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let r = x & 0xFFFF; if r < 40000 { b'e' } else if r < 55000 { (x >> 20) as u8 & 15 } else { (x >> 24) as u8 } }).collect()
}

#[test]
fn huff8_roundtrip_and_size() {
    for n in [0usize, 1, 7, 8, 9, 31, 32, 33, 1000, 100_003] {
        let data = skewed_bytes(n, 99);
        let mut hist = [0u64; 256];
        for &b in &data { hist[b as usize] += 1; }
        let lengths = huff8::lengths_for(&hist);
        let streams = huff8::encode(&data, &lengths);
        assert_eq!(streams.len(), huff8::STREAMS);
        let refs: [&[u8]; 8] = std::array::from_fn(|k| streams[k].as_slice());
        let table = huff8::Table::build(&lengths).unwrap();
        let mut out = vec![0u8; n];
        huff8::decode(&table, &refs, n, &mut out).unwrap();
        assert_eq!(out, data, "n = {}", n);
        if n >= 1000 {
            let coded: usize = streams.iter().map(|s| s.len()).sum();
            assert!(coded < n * 8 / 10, "skewed data must compress: {} vs {}", coded, n);
            assert_eq!(huff8::coded_size(&hist, &lengths), (0..256).map(|s| hist[s] as usize * lengths[s] as usize).sum::<usize>() / 8 + 128);
        }
    }
}

#[test]
fn huff8_rejects_invalid_code() {
    let mut lengths = [1u8; 256]; // Kraft sum 128 > 1
    assert!(huff8::Table::build(&lengths).is_none());
    lengths = [0u8; 256];
    lengths[0] = 1; lengths[1] = 1; // valid
    assert!(huff8::Table::build(&lengths).is_some());
}

#[test]
fn huff8_overrun_is_an_error() {
    let data = skewed_bytes(5000, 7);
    let mut hist = [0u64; 256];
    for &b in &data { hist[b as usize] += 1; }
    let lengths = huff8::lengths_for(&hist);
    let streams = huff8::encode(&data, &lengths);
    let refs: [&[u8]; 8] = std::array::from_fn(|k| streams[k].as_slice());
    let table = huff8::Table::build(&lengths).unwrap();
    let mut out = vec![0u8; 50_000];
    assert!(huff8::decode(&table, &refs, 50_000, &mut out).is_err());
}

/// Ignored perf check, not part of the pristine test run. Reads the Silesia
/// corpus files this repo ships with, extracts every non-raw, non-dense
/// block's literal section exactly as examples/huff_spike.rs does, then
/// times huff8::decode over all of them (best of 5). Run with:
///   RUSTFLAGS="-C target-cpu=native" cargo test --release --test v7_codecs huff8_speed -- --ignored --nocapture
#[test]
#[ignore]
fn huff8_speed_silesia() {
    use simd_stream_codec::format::*;
    use std::time::Instant;

    let files = ["dickens", "mozilla"];
    let mut lits: Vec<Vec<u8>> = Vec::new();
    for f in &files {
        let d = match std::fs::read(format!("corpus/{}", f)) {
            Ok(d) => d,
            Err(_) => continue, // skip silently if the corpus isn't present
        };
        let c = simd_stream_codec::compress(&d);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            if h.flags & (FLAG_RAW_UNCOMPRESSED | FLAG_DENSE) == 0 {
                let lb = cur + HEADER_SIZE + h.token_bytes as usize + h.offset_bytes as usize + h.extras_bytes as usize;
                lits.push(c[lb..lb + h.literal_len as usize].to_vec());
            }
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    if lits.is_empty() {
        return; // corpus not present: nothing to measure
    }

    struct Blk {
        lit: Vec<u8>,
        table: huff8::Table,
        streams: Vec<Vec<u8>>,
    }
    let blks: Vec<Blk> = lits
        .iter()
        .map(|l| {
            let mut hist = [0u64; 256];
            for &b in l {
                hist[b as usize] += 1;
            }
            let lengths = huff8::lengths_for(&hist);
            let streams = huff8::encode(l, &lengths);
            let table = huff8::Table::build(&lengths).unwrap();
            Blk { lit: l.clone(), table, streams }
        })
        .collect();

    let total: usize = blks.iter().map(|b| b.lit.len()).sum();
    let mut out = vec![0u8; blks.iter().map(|b| b.lit.len()).max().unwrap_or(0)];

    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for b in &blks {
            let refs: [&[u8]; huff8::STREAMS] = std::array::from_fn(|k| b.streams[k].as_slice());
            huff8::decode(&b.table, &refs, b.lit.len(), &mut out[..b.lit.len()]).unwrap();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    let ns_per_symbol = best * 1e9 / total as f64;
    println!(
        "huff8_speed_silesia: {} blocks, {} MB literals, {:.3} ns/symbol",
        blks.len(),
        total >> 20,
        ns_per_symbol
    );
}
