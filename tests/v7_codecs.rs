use glyd::bits::{BitReader, BitWriter, Stream, PAD};

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

/// The safe writer fronts a raw cursor reserved for one put of at most
/// MAX_PUT bits: more must panic in release, not write past the reservation.
#[test]
#[should_panic(expected = "MAX_PUT")]
fn bits_writer_rejects_oversized_put() {
    let mut w = BitWriter::new();
    w.put(0, glyd::bits::MAX_PUT + 1);
}

use glyd::huff8;

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
        let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));
        let table = huff8::Table::build(&lengths).unwrap();
        let mut out = vec![0u8; n];
        huff8::decode(&table, &refs, n, &mut out).unwrap();
        assert_eq!(out, data, "n = {}", n);
        if n >= 1000 {
            let coded: usize = streams.iter().map(|s| s.len()).sum();
            assert!(coded < n * 8 / 10, "skewed data must compress: {} vs {}", coded, n);
            assert_eq!(huff8::coded_size(&hist, &lengths), (0..256).map(|s| hist[s] as usize * lengths[s] as usize).sum::<usize>() / 8 + glyd::huffman::packed_lengths_v8_size(&lengths));
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
    let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));
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
    use glyd::format::*;
    use std::time::Instant;

    let files = ["dickens", "mozilla"];
    let mut lits: Vec<Vec<u8>> = Vec::new();
    for f in &files {
        let d = match std::fs::read(format!("corpus/{}", f)) {
            Ok(d) => d,
            Err(_) => continue, // skip silently if the corpus isn't present
        };
        let c = glyd::compress(&d);
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
            let refs: [Stream; huff8::STREAMS] = std::array::from_fn(|k| Stream::whole(b.streams[k].as_slice()));
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

/// Exercises the outer loop's re-evaluation in huff8::decode's fast path:
/// only 3 distinct byte values means short codes (~2 bits each), so the
/// unclamped reader's pointer creeps forward slowly and `safe_refills`
/// (which assumes a worst-case 7 bytes/refill) undershoots badly, forcing
/// many small batches instead of one big one.
#[test]
fn huff8_short_codes_roundtrip() {
    let n = 200_000usize;
    let mut x = 12345u64;
    let data: Vec<u8> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            match x % 3 {
                0 => b'a',
                1 => b'b',
                _ => b'c',
            }
        })
        .collect();
    let mut hist = [0u64; 256];
    for &b in &data {
        hist[b as usize] += 1;
    }
    let lengths = huff8::lengths_for(&hist);
    let streams = huff8::encode(&data, &lengths);
    let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));
    let table = huff8::Table::build(&lengths).unwrap();
    let mut out = vec![0u8; n];
    huff8::decode(&table, &refs, n, &mut out).unwrap();
    assert_eq!(out, data);
}

use glyd::tans;

fn skewed_codes(n: usize, nsym: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; let r = (x >> 16) & 0xFF; (if r < 128 { 0 } else if r < 200 { 1 + (x & 3) } else { x % nsym as u64 }) as u8 }).collect()
}

#[test]
fn tans_normalize_sums_to_l_and_keeps_present() {
    let hist: Vec<u32> = vec![1_000_000, 1, 0, 3, 500];
    let c = tans::normalize(&hist, 5);
    assert_eq!(c.iter().map(|&v| v as u32).sum::<u32>(), tans::L as u32);
    assert!(c[1] >= 1 && c[3] >= 1 && c[4] >= 1 && c[2] == 0);
}

#[test]
fn tans_single_stream_roundtrip() {
    for (n, nsym) in [(0usize, 4usize), (1, 4), (2, 36), (1000, 36), (77_777, 64)] {
        let data = skewed_codes(n, nsym, 5);
        let mut hist = vec![0u32; nsym];
        for &s in &data { hist[s as usize] += 1; }
        if n == 0 { hist[0] = 1; }
        let counts = tans::normalize(&hist, nsym);
        let et = tans::EncodeTable::build(&counts).unwrap();
        let dt = tans::DecodeTable::build(&counts).unwrap();
        let mut enc = tans::Encoder::new(&et);
        for &s in &data { enc.push(s); }
        let stream = enc.finish();
        let mut dec = tans::Decoder::new(&dt, &stream);
        let out: Vec<u8> = (0..n).map(|_| dec.next()).collect();
        assert_eq!(out, data, "n={} nsym={}", n, nsym);
        assert!(!dec.overrun());
    }
}

#[test]
fn tans_rejects_bad_counts() {
    assert!(tans::DecodeTable::build(&[512, 511]).is_none()); // sums to 1023
    assert!(tans::DecodeTable::build(&vec![16u16; 65]).is_none()); // too many symbols
}

#[test]
fn tans8_roundtrip() {
    for n in [0usize, 1, 5, 8, 9, 64, 65, 12_345] {
        let data = skewed_codes(n, 36, 11);
        let mut hist = vec![0u32; 36];
        for &s in &data { hist[s as usize] += 1; }
        if n == 0 { hist[0] = 1; }
        let counts = tans::normalize(&hist, 36);
        let et = tans::EncodeTable::build(&counts).unwrap();
        let dt = tans::DecodeTable::build(&counts).unwrap();
        let streams = tans::encode8(&data, &et);
        let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));
        let mut out = vec![0u8; n];
        tans::decode8(&dt, &refs, n, &mut out).unwrap();
        assert_eq!(out, data, "n={}", n);
    }
}

/// Streams of different lengths: symbols at i % 8 == 0 are always a rare
/// symbol (long code, few bytes consumed per symbol) and the rest are the
/// common symbol (short code), so the 8 sub-streams end up with very
/// different byte lengths. Exercises `safe_refills` picking a low common
/// `iters` across streams whose pointers creep at different rates.
#[test]
fn tans8_streams_of_different_lengths() {
    let n = 20_000usize;
    let nsym = 36usize;
    let data: Vec<u8> = (0..n).map(|i| if i % 8 == 0 { (nsym - 1) as u8 } else { 0u8 }).collect();
    let mut hist = vec![0u32; nsym];
    for &s in &data { hist[s as usize] += 1; }
    let counts = tans::normalize(&hist, nsym);
    let et = tans::EncodeTable::build(&counts).unwrap();
    let dt = tans::DecodeTable::build(&counts).unwrap();
    let streams = tans::encode8(&data, &et);
    let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));
    let mut out = vec![0u8; n];
    tans::decode8(&dt, &refs, n, &mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn tans8_overrun_is_an_error() {
    let n = 5000usize;
    let nsym = 36usize;
    let data = skewed_codes(n, nsym, 7);
    let mut hist = vec![0u32; nsym];
    for &s in &data { hist[s as usize] += 1; }
    let counts = tans::normalize(&hist, nsym);
    let et = tans::EncodeTable::build(&counts).unwrap();
    let dt = tans::DecodeTable::build(&counts).unwrap();
    let streams = tans::encode8(&data, &et);
    let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));
    let mut out = vec![0u8; n * 10];
    assert!(tans::decode8(&dt, &refs, n * 10, &mut out).is_err());
}

/// Ignored perf check, not part of the pristine test run. Generates 14M
/// skewed symbols, encodes with `encode8`, times `decode8` best of 5. Run
/// with:
///   RUSTFLAGS="-C target-cpu=native" cargo test --release --test v7_codecs tans8_speed -- --ignored --nocapture
#[test]
#[ignore]
fn tans8_speed() {
    use std::time::Instant;

    let n = 14_000_000usize;
    let nsym = 36usize;
    let data = skewed_codes(n, nsym, 42);
    let mut hist = vec![0u32; nsym];
    for &s in &data { hist[s as usize] += 1; }
    let counts = tans::normalize(&hist, nsym);
    let et = tans::EncodeTable::build(&counts).unwrap();
    let dt = tans::DecodeTable::build(&counts).unwrap();
    let streams = tans::encode8(&data, &et);
    let refs: [Stream; 8] = std::array::from_fn(|k| Stream::whole(streams[k].as_slice()));

    let mut out = vec![0u8; n];
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        tans::decode8(&dt, &refs, n, &mut out).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    assert_eq!(out, data);
    let ns_per_symbol = best * 1e9 / n as f64;
    println!("tans8_speed: {} symbols, {:.3} ns/symbol", n, ns_per_symbol);
}

use glyd::v7_format::{self, Reps, SubHeader};

#[test]
fn v7_length_and_offset_codes_roundtrip() {
    for v in (0u32..70_000).step_by(7).chain([0, 15, 16, 17, 31, 32, 262_143].into_iter()) {
        let (c, nb, e) = v7_format::ll_code(v);
        assert!(c < v7_format::LL_SYMBOLS as u8);
        assert_eq!(v7_format::extra_bits_of_code(v7_format::Kind::Ll, c), nb);
        assert_eq!(v7_format::ll_value(c, e), v);
        let m = v + v7_format::MIN_MATCH;
        let (c, nb, e) = v7_format::ml_code(m);
        assert!(c < v7_format::ML_SYMBOLS as u8);
        assert_eq!(v7_format::extra_bits_of_code(v7_format::Kind::Ml, c), nb);
        assert_eq!(v7_format::ml_value(c, e), m);
    }
    for o in (1u32..v7_format::MAX_WINDOW).step_by(997).chain([1, 2, 3, 4, 65535, 65536, v7_format::MAX_WINDOW - 1].into_iter()) {
        let (c, nb, e) = v7_format::off_code(o);
        assert!(c >= 3 && c < v7_format::OFF_SYMBOLS as u8);
        assert_eq!(v7_format::extra_bits_of_code(v7_format::Kind::Off, c), nb);
        assert_eq!(v7_format::off_value(c, e), o);
    }
}

#[test]
fn v7_repeat_offsets_encoder_and_decoder_agree() {
    let offsets = [100u32, 100, 7, 100, 7, 7, 300, 100, 300, 1];
    // Both with and without literals before the match: after none the
    // rep symbols are shifted (`FLAG_LL0_REP`) and must shift back.
    for ll0 in [false, true] {
        let mut enc = Reps::new();
        let mut dec = Reps::new();
        let mut rep_hits = 0;
        for (i, &o) in offsets.iter().enumerate() {
            let ll0 = ll0 && i % 2 == 1;
            let (code, nb, extra) = enc.code_for(o, ll0);
            if code < 3 { rep_hits += 1; assert_eq!(nb, 0); }
            assert_eq!(dec.resolve(code, extra, ll0), o);
        }
        assert!(rep_hits >= 5, "repeats must be found: {}", rep_hits);
    }
    assert_eq!(glyd::v7_format::rep_symbol(1, true), 0);
    assert_eq!(glyd::v7_format::rep_symbol(0, true), 2);
    assert_eq!(glyd::v7_format::rep_of_symbol(0, true), 1);
    assert_eq!(glyd::v7_format::rep_of_symbol(2, true), 0);
    assert_eq!(glyd::v7_format::rep_of_symbol(5, true), 5);
}

#[test]
fn v7_subheader_roundtrip() {
    let h = SubHeader { coded: 0b01011, reuse: 0b10, dict_id: 0xDEADBEEF, sizes: [1, 2, 3, 4, 5] };
    let mut out = Vec::new();
    h.write(&mut out);
    assert_eq!(out.len(), SubHeader::BYTES);
    let p = SubHeader::parse(&out).unwrap();
    assert_eq!((p.coded, p.reuse, p.dict_id, p.sizes), (h.coded, h.reuse, h.dict_id, h.sizes));
    assert!(SubHeader::parse(&out[..10]).is_none());
}
