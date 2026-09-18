use simd_stream_codec::v7_encode::{encode_block, payload_layout, Sequence, Tables};
use simd_stream_codec::v7_format::{SubHeader, S_EXTRA, S_LIT, S_LL, S_ML, S_OFF};

fn sample_sequences() -> (Vec<Sequence>, Vec<u8>) {
    // "abcabcabc..." style: literal runs then matches at small offsets.
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut x = 42u64;
    for i in 0..5000u32 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let ll = (x % 20) as u32;
        for _ in 0..ll { lits.push((x >> 8) as u8); }
        let ml = 3 + (x >> 16) as u32 % 40;
        let offset = if i % 3 == 0 { 100 } else { 1 + (x >> 24) as u32 % 5000 };
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset });
    }
    lits.extend_from_slice(b"tail literals");
    seqs.push(Sequence { lit_len: 13, match_len: 0, offset: 0 });
    (seqs, lits)
}

#[test]
fn v7_encode_block_layout_is_self_describing() {
    let (seqs, lits) = sample_sequences();
    let mut prev = Tables::none();
    let mut out = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut out);
    let layout = payload_layout(&out).unwrap();
    assert_eq!(layout.sub.dict_id, 0);
    let total: usize = SubHeader::BYTES + layout.sub.sizes.iter().map(|&s| s as usize).sum::<usize>();
    assert_eq!(total, out.len());
    for s in [S_LIT, S_LL, S_ML, S_OFF, S_EXTRA] {
        assert_eq!(layout.sections[s].len(), layout.sub.sizes[s] as usize);
    }
    // Skewed sequence codes must have been coded, not stored raw.
    assert!(layout.sub.coded & (1 << S_LL) != 0);
    assert!(layout.sub.coded & (1 << S_OFF) != 0);
    // A second block with the same statistics reuses the sequence tables.
    let mut out2 = Vec::new();
    encode_block(&seqs, &lits, 0, &mut prev, &mut out2);
    let l2 = payload_layout(&out2).unwrap();
    assert!(l2.sub.reuse & 0b10 != 0);
    assert!(out2.len() < out.len());
}

/// A symbol absent from the previous block's table (count 0) but present,
/// even rarely, in this block's data must not pass the reuse check: a
/// reused table that assigns it zero probability sends the tANS encoder
/// out of bounds (see `close`). Block 2 introduces ll code 20 (lit_len
/// 300, via log2 bucketing), which never appears in `sample_sequences()`.
#[test]
fn v7_encode_reuse_with_new_symbol_does_not_panic() {
    let (seqs1, lits1) = sample_sequences();
    let mut seqs2 = seqs1.clone();
    let mut lits2 = lits1.clone();
    let added: u32 = (300 - seqs2[10].lit_len) + (300 - seqs2[11].lit_len);
    seqs2[10].lit_len = 300;
    seqs2[11].lit_len = 300;
    lits2.extend(std::iter::repeat(b'x').take(added as usize)); // keep lit_lens summing to lits.len()

    let mut prev = Tables::none();
    let mut out1 = Vec::new();
    encode_block(&seqs1, &lits1, 0, &mut prev, &mut out1);
    let mut out2 = Vec::new();
    encode_block(&seqs2, &lits2, 0, &mut prev, &mut out2); // must not panic

    let l2 = payload_layout(&out2).expect("second block payload must parse");
    assert!(l2.sub.reuse & 0b10 == 0, "must not reuse ll/ml/off tables when a new symbol appears");
}

/// The up-front reuse decision is optimistic (based on a closeness check
/// against fresh counts); if one of the three streams then decides raw
/// beats coding after all, the other two must not be left with an
/// omitted table under a header that ends up reporting "not reused" --
/// every coded, non-reused section must carry its own table. Block 1 (400
/// sequences) makes coding pay for all three streams; block 2 (192 of the
/// same shape) shrinks the win enough that `ml` alone falls back to raw
/// while reusing `ll`/`off`'s tables still looks attractive up front.
#[test]
fn v7_encode_reuse_falls_back_when_a_stream_goes_raw() {
    fn repeated_matches(n: u32) -> (Vec<Sequence>, Vec<u8>) {
        let mut seqs: Vec<Sequence> =
            (0..n).map(|i| Sequence { lit_len: 0, match_len: 3 + (i % 16), offset: 1 }).collect();
        seqs.push(Sequence { lit_len: 0, match_len: 0, offset: 0 });
        (seqs, Vec::new())
    }
    let (seqs1, lits1) = repeated_matches(400);
    let (seqs2, lits2) = repeated_matches(192);

    let mut prev = Tables::none();
    let mut out1 = Vec::new();
    encode_block(&seqs1, &lits1, 0, &mut prev, &mut out1);
    let mut out2 = Vec::new();
    encode_block(&seqs2, &lits2, 0, &mut prev, &mut out2);

    let l2 = payload_layout(&out2).expect("second block payload must parse");
    for (s, n_symbols) in [(S_LL, 32u8), (S_ML, 32u8), (S_OFF, 24u8)] {
        if l2.sub.coded & (1 << s) != 0 && l2.sub.reuse & 0b10 == 0 {
            let sec = &out2[l2.sections[s].clone()];
            assert_eq!(sec[0], n_symbols, "coded, non-reused stream {} must carry its own table", s);
        }
    }
}
