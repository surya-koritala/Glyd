use simd_stream_codec::huffman::*;

fn roundtrip(data: &[u8]) {
    let mut hist = [0u64; 256];
    for &b in data { hist[b as usize] += 1; }
    let lengths = build_lengths(&hist);
    for (i, &l) in lengths.iter().enumerate() {
        assert!(l as u32 <= MAX_CODE_LEN, "symbol {} length {} exceeds cap", i, l);
        if hist[i] > 0 { assert!(l > 0, "used symbol {} got no code", i); }
    }
    // Kraft sum must not exceed 1 for a valid prefix code.
    let kraft: f64 = lengths.iter().filter(|&&l| l > 0)
        .map(|&l| 2f64.powi(-(l as i32))).sum();
    assert!(kraft <= 1.0 + 1e-9, "Kraft sum {} > 1", kraft);

    let codes = build_codes(&lengths);
    let mut packed = Vec::new();
    let mut w = BitWriter::new();
    for &b in data { w.put(codes[b as usize], lengths[b as usize], &mut packed); }
    w.finish(&mut packed);

    let mut lenbuf = Vec::new();
    pack_lengths(&lengths, &mut lenbuf);
    assert_eq!(lenbuf.len(), LENGTHS_BYTES);
    let lengths2 = unpack_lengths(&lenbuf);
    assert_eq!(&lengths2[..], &lengths[..]);

    let table = DecodeTable::build(&lengths2).expect("table");
    let mut out = vec![0u8; data.len()];
    decode_into(&table, &packed, data.len(), &mut out).expect("decode");
    assert_eq!(&out[..], data, "huffman roundtrip mismatch");
}

#[test]
fn huff_uniform() {
    let d: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    roundtrip(&d);
}

#[test]
fn huff_skewed() {
    let mut d = Vec::new();
    for i in 0..20000 { d.push(if i % 10 == 0 { (i % 251) as u8 } else { 7u8 }); }
    roundtrip(&d);
}

#[test]
fn huff_single_symbol() {
    roundtrip(&vec![42u8; 1000]);
}

#[test]
fn huff_two_symbols() {
    let d: Vec<u8> = (0..1000).map(|i| if i % 3 == 0 { 1u8 } else { 2u8 }).collect();
    roundtrip(&d);
}

#[test]
fn huff_extreme_skew_forces_length_limit() {
    // Weights that grow faster than the golden ratio build a fully skewed
    // Huffman tree: natural depth is (symbols - 1). With more than
    // MAX_CODE_LEN + 1 symbols that exceeds the cap, so the rescaling path
    // must engage and still produce a valid prefix code.
    //
    // Total input is capped at BUDGET bytes. An earlier version grew 60
    // symbols by 1.7x each, roughly 1.7^60 bytes, and got the WSL VM
    // OOM-killed. 21 symbols under 200 KB is enough to push depth to 20.
    const BUDGET: usize = 200_000;
    let mut weights: Vec<usize> = Vec::new();
    let mut total = 0usize;
    let mut w = 1usize;
    while total + w <= BUDGET {
        weights.push(w);
        total += w;
        w = (w as f64 * 1.7) as usize + 1;
    }
    // Chain condition: the weight two ahead outweighs the running sum, so the
    // Huffman builder always merges the accumulated node with the next leaf
    // and the tree degenerates into a chain of depth (symbols - 1).
    let mut sum = 0usize;
    for k in 0..weights.len() {
        sum += weights[k];
        if k + 2 < weights.len() {
            assert!(weights[k + 2] >= sum, "weights not skewed enough at {}", k);
        }
    }
    assert!(
        weights.len() > MAX_CODE_LEN as usize + 1,
        "{} symbols cannot exceed the {}-bit cap",
        weights.len(),
        MAX_CODE_LEN
    );

    let mut d = Vec::with_capacity(total);
    for (s, &w) in weights.iter().enumerate() {
        d.extend(std::iter::repeat(s as u8).take(w));
    }
    assert!(d.len() <= BUDGET, "test input {} bytes exceeds budget", d.len());
    roundtrip(&d);
}

#[test]
fn huff_real_token_shape() {
    // Mimics the measured token distribution: a few very common symbols.
    let mut d = Vec::new();
    let mut st = 12345u64;
    for _ in 0..50000 {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (st >> 33) % 100;
        d.push(if r < 60 { 0u8 } else if r < 80 { 9u8 } else if r < 92 { 17u8 }
               else { ((st >> 20) % 256) as u8 });
    }
    roundtrip(&d);
}
