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
