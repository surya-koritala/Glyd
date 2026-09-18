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
